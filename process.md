First, we need to get an optimal evaluation system.
- Determine difference between different evaluation methods:
    - Evaluate the metric prompts of Context Precision, Context Recall, Factual Correctness, and Sematic Similarity using t-validation(each for 3 times).
    - We need to iterate the prompt to stabilize the score of each metric.
    - We can also use another prompt to evaluate the different prompt.
- For different task, we need to combine different metrics to get different final scores. So we will construct a formula to calculate the final score.

Second, we can use different retrieval methods to get results.
- Embedding Search
- BM25
- Hybrid
- GraphRag
- Agent


We need to test the performance of each method using different datasets.

The conclusion of our paper will include:
The effects of parameters and prompts on the metric scores.
The formula to calculate the final score.
The performance of different retrieval methods.
