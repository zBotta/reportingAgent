# AI Reporting Agent
An AI reporting agent that uses a Language Model to generate reports

# Description

In many industries, as part of a highly performing Quality Assurance and Customer Support functions, reporting unusual events, anomalies, or process deviations requires clear, structured documentation that captures the full context: what happened, when, where, who was involved, why it occurred, and what actions were taken… This project proposes the development of a chatbot assistant that interacts with the user to collect all relevant information through a series of structured questions. The assistant then automatically generates a well-written, standardized report based on the input. Here some of the tasks carried out:

- Data creation: generate a data set of reports to use them as ground truth (reference) to train the model. This was made by automating prompts through an API.

- Model selection and experimentation plan : Test different Hugging Face models to evaluate its performance. Play with the generation parameters (temperature, top_k, top_p, presence_penalty, etc) and compare the results

- Evaluation metrics : Text-to-text comparison, through tokenization and the attention mechanism: Bert score, bleu/rouge score, jaccard, sentence similarity

- Model training : Fine-tune the parameters of an already-trained tiny model (less than 1 billion parameters) from Hugging face in order to specialize it in the reporting task

# Model selection
Two model sizes were studied for generating one-paragraph reports: 
- Meidum size models (< 3B parameters)
- Small size models (< 1B parameters)

A grid search has been done to select the best generation parameters. Here below you can find the results of the grid search.  

**Medium size models Grid Search**  

![photo](assets/GridSearchMediumModels.png)

**Small size models Grid Search**  

![photo](assets/GridSearchSmallModels.png)

## Best models
The best models giving the highest cross encoder similarity score were:
-	Best small size model: **Qwen2.5-0.5B-Instruct**
-	Best medium size model: **Llama-3.2-3B-Instruct**

# Evaluation metrics
Several evaluation metrics have been calculated in the grid search:
- **BLEU**: similar to precision on n-grams
- **ROUGE**: similar to recall on n-grams
- **BERTscore**: precision, recall and f1
- **Bi-encoder similarity** (cosine similarity between outputs sequences)
- **Cross-encoder similarity** (best score for evaluating texts with several sentences)

# Getting started

Clone the repository and install the dependencies.

``` pip install -r requirements.txt ```

## App execution

To open the app, execute the following command in the root folder.

``` streamlit run app/reportingAgent.py --server.address=0.0.0.0 --server.port=8501 ```

## Grid search execution

To call the grid search, you can use the script `reportParamGridSearch.py` found in the `app` folder.
An example for calling it is as follows
```
python app/reportParamGridSearch.py --model_id openai-community/gpt2-xl --non-threaded --prompt_method B C --max_workers 4 --dataset_filename pharma_dev_reports_collection.xlsx --start_idx 1 --end_idx 80  --temperature 0.7 1.0 1.3 --top_p 0.3 0.6 0.9 --top_k 30 50 70 --max_new_tokens 300 --do_sample True
```
Here we specify:
- The model_id: any model id existing in HF
- The prompt method: ```A, B, C or D```
- The generation parameters grid: temperature, top_k, max_new_tokens, do_sample, etc
- The start and end index on the test set. The test set is found in HF or in an excel file locally copied when cloned.
- The data set filename: `pharma_dev_reports_collection.xlsx` or `traffic_accident_reports_collection.xlsx`


# Data sets and models availability
The models and data sets can be found in the HuggingFace [DSTI Community](https://huggingface.co/DSTI)

