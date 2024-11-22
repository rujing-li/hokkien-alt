# Hokkien ASR Model Fine-Tuning

This project focuses on fine-tuning an Automatic Speech Recognition (ASR) model for Hokkien song lyrics transcription using the Whisper model.

## Files

- **`data.zip`**  
  Contains all the relevant datasets for the project. Once unzipped, it provides the necessary data for fine-tuning and evaluation. Ensure that this file is extracted before running any scripts.

- **`whisper_fine_tuning.ipynb`**  
  Jupyter Notebook used for fine-tuning the Whisper model. This file includes the entire pipeline: loading datasets, preprocessing, fine-tuning the model, and evaluating results.

- **`process_suisiann.py`**  
  A Python script designed to preprocess the SuíSiann dataset by converting tonal information into number tones for better compatibility with the ASR model.

- **`test_hokkien.mp3`**  
  A sample audio file in Hokkien used for testing the fine-tuned model. This can be used to verify the model's performance.

- **`results/`**  
  A folder containing the outputs from the model training process, including training logs, evaluation metrics, and saved model checkpoints.

## Usage Instructions

### Extract Data  
Unzip `data.zip` to make the relevant datasets available. Ensure that all files are properly extracted into the current working directory.

### Process SuíSiann Dataset  
Run `process_suisiann.py` to preprocess the SuíSiann dataset. This step is necessary to ensure that the data is in the required format for training. `data.zip` already contains the processed csv file.

### Train the Model
Open whisper_fine_tuning.ipynb in Jupyter Notebook and follow the steps outlined in the notebook. This will load the data, preprocess it further, and fine-tune the Whisper model. 

