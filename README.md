# AI Reporting Agent
An AI reporting agent that uses a Language Model to generate reports

# Description

In many industries, as part of a highly performing Quality Assurance and Customer Support functions, reporting unusual events, anomalies, or process deviations requires clear, structured documentation that captures the full context: what happened, when, where, who was involved, why it occurred, and what actions were taken… This project proposes the development of a chatbot assistant that interacts with the user to collect all relevant information through a series of structured questions. The assistant then automatically generates a well-written, standardized report based on the input. Here some of the tasks carried out:

- Data creation: generate a data set of reports to use them as ground truth (reference) to train the model. This was made by automating prompts through an API.

- Model selection and experimentation plan : Test different Hugging Face models to evaluate its performance. Play with the generation parameters (temperature, top_k, top_p, presence_penalty, etc) and compare the results

- Evaluation metrics : Text-to-text comparison, through tokenization and the attention mechanism: Bert score, bleu/rouge score, jaccard, sentence similarity

- Model training : Fine-tune the parameters of an already-trained tiny model (less than 1 billion parameters) from Hugging face in order to specialize it in the reporting task
