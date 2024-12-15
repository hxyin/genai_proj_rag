
# README

## Introduction
This project introduces an advanced evaluation framework for RAG Systems that significantly improves upon existing open-source methodologies. Our framework achieves:

- **Superior Accuracy**: Consistently outperforms standard open-source benchmarks
- **Reduced Variance**: Significantly lower variance in evaluation metrics compared to baseline methods
- **Enhanced Reproducibility**: Structured approach ensuring consistent results across evaluations


## Prerequisites

### Vertex AI Setup
1. **Registration**:  
   Sign up for Vertex AI and ensure that your account has the necessary permissions to execute tasks. This may include enabling APIs and setting IAM roles.

2. **GCP Authentication JSON File**:  
   Download your Google Cloud authentication JSON file from the GCP Console and save it to a secure location on your local machine. This file is required for accessing Vertex AI programmatically.

### Environment Requirements
- **Python Version**:  
   The scripts are tested with **Python 3.12.7**. Ensure your environment matches this version to avoid compatibility issues.

- **Required Libraries**:  
   The project uses several Python libraries, including:
   - `langchain`
   - `llama-index`
   - `ragas`  
   A complete list of dependencies is provided in the `requirements.txt` file. Install all dependencies with the following command:

   ```bash
   pip install -r requirements.txt
   ```

## File Structure

- `retrieval_evalulation.ipynb`:
  The implementation of the retrieval system, including Vector-based Retrieval, BM25 Retrieval and Hybrid Retrieval.

- `./llm_evaluation/llm_result_evaluation.ipynb`:  
   The main evaluation script that integrates all components of the framework. This notebook orchestrates the evaluation workflow.

- `./llm_evaluation/factualcorrectness_revise.py`:  
   The core script that defines custom metrics and evaluation logic. Significant modifications have been made to existing open-source frameworks to improve accuracy and flexibility.

- `./llm_evaluation/evaluation_plots`:
   The t-validation metrics (factual correctness, faithfulness, semantic similarity) of the three different retrieval strategies.

- Other supporting files:  
   Additional scripts in the directory provide utility functions and support for various evaluation tasks. All files are referenced within the notebook and should remain in the same directory.

## How to Use

### Step 1: Configure Your Environment
1. Install the required Python version (3.12.7).
2. Set up your Python environment using the following commands:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

### Step 2: Set Up GCP Authentication
1. Place the downloaded GCP authentication JSON file in a known location.
2. Export the path to your environment:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/auth.json"
   ```
   Replace `path/to/your/auth.json` with the actual path to your JSON file.

### Step 3: Run the Evaluation
1. Open `./llm_evaluation/llm_result_evaluation.ipynb` in a Jupyter Notebook environment.
2. Execute the notebook cells sequentially. The notebook will automatically load and use the supporting files in the directory.
3. The evaluation results, including metrics and comparisons, will be generated within the notebook.
4. Open and execute `retrieval_evalulation.ipynb` in a Jupyter Notebook environment to run and test the three different retrieval strategies and generate the corresponding results.

### Custom Metrics
The evaluation framework introduces enhancements to key metrics, including:
- **Improved In-context learning**:  
   Optimized prompts and examples for more accurate evaluations.

- **Few-shot learning**:  
   - Adjustments to sampling and scoring methods for enhanced benchmarking
   - More consistent performance across diverse test cases
   - Better handling of edge cases and complex scenarios

These modifications are implemented in `factualcorrectness_revise.py` and automatically integrated into the main notebook.

## Notes
- Ensure your environment is correctly configured before running the scripts.
- For detailed implementation of metrics and logic, refer to the comments within `factualcorrectness_revise.py`.
- The evaluation results demonstrate consistently lower variance compared to existing frameworks, making our approach more reliable for production use.
- Our framework's improved stability makes it particularly suitable for continuous evaluation pipelines where consistent results are crucial.
