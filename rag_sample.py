import os
from typing import List
import fitz
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-3.5-turbo")

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    documents = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
            if text.strip():
                metadata = {
                    "source": pdf_path,
                    "page": page_num + 1,
                    "total_pages": len(pdf_document)
                }
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

def initialize_rag_pipeline(data_dir: str = "./data"):
    docs = load_pdfs_from_directory(data_dir)
    if not docs:
        raise ValueError(f"No documents were loaded from {data_dir}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    try:
        rag_chain = initialize_rag_pipeline()
        response = rag_chain.invoke("How does the number of datasets and templates affect the performance of instruction tuning in the FLAN model?")
        print(response)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")