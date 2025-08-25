# Test data sets
The excels files *_collection.xlsx can be seen as the TEST sets for doing a Grid Search on the HyperParameters

# Training and eval data sets
For the training task, we have 700 rows and we generate files train and evaluation files: train.jsonl and eval.jsonl.

For the training we have used **Lora + PEFT** to optimize the training runtime and resources used.

The model that we have trained is a **SmolLM2-360M-Instruct** to adhere to a one paragraph report output.