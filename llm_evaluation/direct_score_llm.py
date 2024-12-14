import os
import re
import pandas as pd
from langchain_google_vertexai import VertexAI
from langchain import PromptTemplate, LLMChain

# Set GCP credentials and environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Apple/secrets/genai-441923-47c3e249f8b8.json"

# Initialize the LLM
llm = VertexAI(
    model_name="gemini-1.0-pro",
    temperature=0.3,
    max_output_tokens=1024,
    max_workers=1,
)

# Define the prompt template
# The prompt asks the LLM to compare the LLM output to the reference answer for the given question.
# It should output a JSON with a "Score" field: a value between 0.0 and 1.0 (e.g., 0.00, 0.05, 0.10, ... 1.00)
# You can instruct the model to choose the closest increment. The increments are for you to interpret,
# but this shows how you might ask the model to produce a rating and then parse it.
prompt_template = """
You are a professional evaluator. You will be given:
1. A question.
2. A reference answer to that question.
3. Another LLM's answer to the same question.

Your task is to provide a final numerical rating between 0.0 and 1.0 that reflects how well the LLM's answer matches the reference answer in terms of correctness, completeness, and relevance. The rating should be in increments of 0.01, for example: 0.00, 0.01, 0.02, ..., 0.55, ..., 0.66 ... up to 1.00.
Return your answer in the following JSON format only:
{{
  "Score": <your_score>
}}

### Input
Question:
{question}

Reference Answer:
{reference_answer}

LLM Output Answer:
{llm_output}

### Output
Please respond ONLY with the JSON object containing the "Score" field.
"""

template = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "reference_answer", "llm_output"]
)

chain = LLMChain(llm=llm, prompt=template)

# Sample inputs
question = "What is the Zero-shot-CoT method and how does it elicit chain of thought from large language models?"
reference_answer = "Zero-shot-CoT is a zero-shot template-based prompting method that elicits chain of thought reasoning from large language models. It does not require step-by-step few-shot examples and instead uses a single fixed prompt to prompt the models. This method encourages the discovery of broad cognitive abilities in LLMs rather than narrow task-specific skills."
llm_output = "The Zero-shot-CoT method is a way to elicit chain of thought from large language models. It does this by adding the prompt \"Let's think step by step\" before each answer. This method has been shown to be effective in eliciting complex multi-step reasoning from large language models."

results = []

# Perform 20 evaluations
for i in range(30):
    response = chain.run(
        question=question,
        reference_answer=reference_answer,
        llm_output=llm_output
    )
    # Parse the JSON response using regex
    match = re.search(r'"Score"\s*:\s*(0\.\d+|1\.0+)', response)
    if match:
        score = float(match.group(1))
    else:

        score = 0.0

    results.append({
        "Question": question,
        "Reference_Answer": reference_answer,
        "LLM_Output_Answer": llm_output,
        "Score": score
    })

df = pd.DataFrame(results)
df.to_excel("direct_llm_evaluations_0_to_1.xlsx", index=False)
print("Evaluations completed. Results saved to llm_evaluations.xlsx")
