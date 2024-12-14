import os
import time
import hashlib
from typing import List
import fitz
import pandas as pd
from google.auth import load_credentials_from_file
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import EvaluationDataset

credentials, project_id = load_credentials_from_file("/Users/Apple/secrets/genai-441923-47c3e249f8b8.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Apple/secrets/genai-441923-47c3e249f8b8.json"

llm = VertexAI(
    model_name="gemini-1.5-pro",
    temperature=0.3,
    max_output_tokens=8192,
    max_workers=2,
)

embedding_model = VertexAIEmbeddings(
    model_name="textembedding-gecko"
)


def get_directory_hash(directory_path: str) -> str:
    hasher = hashlib.md5()
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()


def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    documents = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
            if text.strip():
                metadata = {"source": pdf_path, "page": page_num + 1, "total_pages": len(pdf_document)}
                documents.append(Document(page_content=text, metadata=metadata))
        pdf_document.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
    return documents


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            documents.extend(extract_text_from_pdf(file_path))
    return documents


class RAGPipeline:
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.rag_chain = None
        self.retrieve_docs = []
        self.initialize_pipeline()

    def initialize_pipeline(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        should_update = self._should_update_embeddings()
        if not should_update and os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=embedding_model)
        else:
            self._create_new_embeddings()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            doc_contents = [doc.page_content for doc in docs]
            self.retrieve_docs.append(doc_contents)
            return "\n\n".join(doc_contents)

        self.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

    def _should_update_embeddings(self) -> bool:
        if not os.path.exists(self.persist_dir):
            return True
        current_hash = get_directory_hash(self.data_dir)
        hash_file = os.path.join(self.persist_dir, "directory_hash.txt")
        if not os.path.exists(hash_file):
            return True
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        return current_hash != stored_hash

    def _create_new_embeddings(self):
        docs = load_pdfs_from_directory(self.data_dir)
        if not docs:
            raise ValueError(f"No documents were loaded from {self.data_dir}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len,
                                                       is_separator_regex=False)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model,
                                                 persist_directory=self.persist_dir)
        current_hash = get_directory_hash(self.data_dir)
        with open(os.path.join(self.persist_dir, "directory_hash.txt"), 'w') as f:
            f.write(current_hash)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=k)
        self.retrieve_docs.append([doc.page_content for doc in docs])
        return docs

    def query(self, question: str) -> str:
        return self.rag_chain.invoke(question)

    def get_retrieve_history(self) -> List[List[str]]:
        return self.retrieve_docs

    def clear_retrieve_history(self):
        self.retrieve_docs = []

    def retrieve_and_query(self, query: str) -> str:
        self.retrieve(query)
        query_res = self.query(query)
        retrieve_res = self.get_retrieve_history()[0] if self.get_retrieve_history() else []
        self.clear_retrieve_history()
        return query_res, retrieve_res


rag = RAGPipeline(data_dir="./data", persist_dir="./chroma_db")
ans, retrieve_history = rag.retrieve_and_query(
    "How does the number of datasets and templates affect the performance of instruction tuning in the FLAN model?")
test_dataset = pd.read_json('dataset_data/eval_dataset_1.json')
test_dataset.columns = ['user_input', 'reference', 'response', 'retrieved_contexts']
import time

for i in range(len(test_dataset)):
    res, retrieve_data = rag.retrieve_and_query(test_dataset['user_input'][i])
    test_dataset.loc[i, 'response'] = res
    test_dataset.loc[i, 'retrieved_contexts'] = retrieve_data
    time.sleep(1)
test_dataset.to_json('./test_dataset_for_eval.json', orient='records', lines=True)

my_run_config = RunConfig(max_workers=1, timeout=60)
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration


def custom_is_finished_parser(response: LLMResult):
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]
        if resp.generation_info is not None:
            if resp.generation_info.get("finish_reason") is not None:
                is_finished_list.append(resp.generation_info.get("finish_reason") == "STOP")
        elif isinstance(resp, ChatGeneration) and resp.message is not None:
            resp_message: BaseMessage = resp.message
            if resp_message.response_metadata.get("finish_reason") is not None:
                is_finished_list.append(resp_message.response_metadata.get("finish_reason") == "STOP")
        else:
            is_finished_list.append(True)
    return all(is_finished_list)


llm = VertexAI(
    model_name="gemini-1.5-pro",
    temperature=0.01,
    max_output_tokens=8192,
    max_workers=1,
)

embedding_model = VertexAIEmbeddings(
    model_name="textembedding-gecko"
)
evaluator_llm = LangchainLLMWrapper(llm, run_config=my_run_config, is_finished_parser=custom_is_finished_parser)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model, run_config=my_run_config)

eval_test_dataset = pd.read_json('./test_dataset_for_eval.json', orient='records', lines=True)
eval_dataset = EvaluationDataset.from_pandas(eval_test_dataset)

from evalute_fact import FactualCorrectnessRevise1

metrics = [FactualCorrectnessRevise1(llm=evaluator_llm)]
results = evaluate(dataset=eval_dataset, metrics=metrics, run_config=my_run_config)

from evaluate_fact_ver4 import FactualCorrectnessRevise4

for i in range(5):
    metrics = [FactualCorrectnessRevise4(llm=evaluator_llm)]
    results = evaluate(dataset=eval_dataset, metrics=metrics, run_config=my_run_config)

from evaluate_fact_ver5 import FactualCorrectnessRevise5

for i in range(10):
    metrics = [FactualCorrectnessRevise5(llm=evaluator_llm)]
    results = evaluate(dataset=eval_dataset, metrics=metrics, run_config=my_run_config)

from evaluate_fact_beta import FactualCorrectnessReviseBeta

for i in range(10):
    metrics = [FactualCorrectnessReviseBeta(llm=evaluator_llm)]
    results = evaluate(dataset=eval_dataset, metrics=metrics, run_config=my_run_config)
