# Large Language Model (LLM) Project

## Introduction

This repository contains the code for a Large Language Model (LLM) project, inspired by models like GPT. It includes two versions of the project: a new, object-oriented (OOP) version under the `new` directory, and an older, procedural version in the `old` directory. The new version is for understanding and educational purposes, while the old version is fully functional.

## File Architecture

- `new/`: Contains the OOP-based implementation of the LLM project. The code is well-commented and organized, though it might have some errors.
- `old/`: Houses the original, procedural version of the project. This version is stable and can be used for running the model.

## OpenWebText Data Source

The LLM project uses the OpenWebText Corpus, a large-scale text dataset. Due to its size (12 GB compressed, expanding to 38 GB), the data isn't included in this repository.

### Downloading OpenWebText

1. Visit OpenWebText Archive.
2. Download the compressed dataset to your local machine.
3. Extract the dataset in a directory accessible to the project.

## Running the Project - New Version

### Setup

Ensure all file paths in the scripts are correctly set to match your local environment, particularly the paths for the OpenWebText data.

### Execution Steps

1. Data Extraction: Run `data_extraction.py` to process the OpenWebText data.
2. Data Handling: Execute `data_handling.py` to prepare the data for training.
3. Training: Run `training.py` to train the model.
4. Chatbot: Finally, execute `chatbot.py` to interact with the trained model.

### Expected Outcomes

- Data Extraction: Generates processed text files and vocabulary.
- Data Handling: Prepares batches of data for the model.
- Training: Trains the model and saves its state.
- Chatbot: Provides an interactive chat interface using the trained model.

## Running the Project - Old Version

### Execution Steps

1. Run `data_extraction.py` to prepare the OpenWebText data.
2. Execute `training.py` to train the model.
3. Use `chatbot.py` for an interactive demonstration.

## Model Configuration and Expectations

### Recommended Settings

```python
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
```
## Hardware Recommendations

- GPU: Training is resource-intensive. A robust GPU is recommended for faster processing.
- CPU: Training on CPU is possible but will be significantly slower.

## Additional Notes
Error Handling: In the new version, be mindful of potential errors and use the old version as a stable fallback.
