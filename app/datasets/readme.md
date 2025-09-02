# Test data sets
The excel files `*`_collection.xlsx can be seen as the TEST sets for doing a Grid Search on the HyperParameters
The excel files training_`*`.xslx can be seen as the TRAINING and EVAL sets for the training task.

# Training and eval data sets
For the training task, we have 5k rows and we have generated a train and evaluation dataset.

For the training we have used **Quantization + PEFT** to optimize the training runtime and resources used.

The model that we have trained is a **SmolLM2-360M-Instruct** to adhere to a one paragraph report output.

## Example for loading dataset (python)
```
from datasets import load_dataset
ds = load_dataset("zBotta/traffic-accidents-reports-5k")
print(ds)
print(ds["train"][0])
print(ds["eval"][0])
print(ds["test"][0])
```

More information in the [Hugging Face repo](https://huggingface.co/zBotta/datasets)
