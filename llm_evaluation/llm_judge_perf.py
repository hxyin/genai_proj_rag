import os
import pandas as pd
import re
from langchain_google_vertexai import VertexAI
from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

llm = VertexAI(
    model_name="gemini-1.5-pro",
    temperature=0.01,
    max_output_tokens=8192,
    max_workers=2,
)

prompt_template = """
You are a professional text evaluator. You are given one original source text and two different segmented versions (Version A and Version B).

You must evaluate these versions according to the previously established criteria. The criteria and their weights are as follows:
1. Information Completeness (40%):  
   - Does the version retain all the essential information present in the original text without omissions or distortions?
   - Are any critical details missing or misrepresented?

2. Sentence Structure (30%):  
   - Are the sentences well-structured, coherent, and clear?
   - Does the segmentation improve readability and logical flow without changing meaning?

3. Semantic Accuracy (30%):  
   - Does each segmented sentence accurately reflect the meaning of the corresponding part of the original text?
   - Are there any semantic shifts or misunderstandings introduced?

Your task:
1. Internally evaluate both Version A and Version B according to these criteria, and compute a weighted score for each version.  
   (You do NOT need to show these internal computations in the final answer, just do it internally.)

2. Compare the two final weighted scores (Score A and Score B).

3. If Version A's score is higher than Version B's score, return "TRUE". If Version B's score is higher than Version A's score, return "FALSE". If both scores are equal, return "EVEN".

4. After the decision, provide a short explanation (one to two sentences) justifying why one version is considered better than the other, or why they are considered equal, according to the criteria above.

Important:  
- Do NOT output the detailed breakdown of scores or the full reasoning steps.  
- Only output the final decision and a brief explanation.  
- The output format must be strictly JSON with only the two fields "FinalDecision" and "Explanation".

### Input
Original Text:
{original_text}

Version A:
{version_a}

Version B:
{version_b}

### Output Format
Please respond in JSON with the following fields only:
{{
  "FinalDecision": "<TRUE_or_FALSE_or_EVEN>",
  "Explanation": "<your_explanation>"
}}
"""

template = PromptTemplate(
    template=prompt_template,
    input_variables=["original_text", "version_a", "version_b"]
)

chain = LLMChain(llm=llm, prompt=template)

input_df = pd.read_excel("./merged_unique_responses.xlsx")

results = []
for idx, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Evaluating Responses"):
    original_text = row["response_original"]
    version_a = row["response_claims_file1"]
    version_b = row["response_claims_file2"]

    response = chain.run(original_text=original_text, version_a=version_a, version_b=version_b)

    if '"FinalDecision": "TRUE"' in response:
        final_decision = "TRUE"
    elif '"FinalDecision": "FALSE"' in response:
        final_decision = "FALSE"
    elif '"FinalDecision": "EVEN"' in response:
        final_decision = "EVEN"
    else:
        final_decision = "UNKNOWN"

    explanation_match = re.search(r'"Explanation":\s*"([^"]+)"', response)
    if explanation_match:
        explanation = explanation_match.group(1)
    else:
        explanation = ""

    results.append({
        "Original_Text": original_text,
        "Version_A": version_a,
        "Version_B": version_b,
        "LLM_Response": response,
        "FinalDecision": final_decision,
        "Explanation": explanation
    })

output_df = pd.DataFrame(results)
output_df.to_excel("evaluation_results_ver5.xlsx", index=False)

print("Evaluation completed. Results saved to evaluation_results.xlsx")
