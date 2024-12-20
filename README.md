# Entity-Recognition-Address
This project focuses on recognizing entities within address data using fine-tuned models (SpaCy and BERT). The project includes a Streamlit application that provides a user-friendly interface for:
  Inputting address text (either directly or via a text file).
  Processing the input with fine-tuned SpaCy and BERT models to extract entities.
  Downloading the results as a CSV file when a text file is provided as input.

Features
Streamlit App:
  Accepts input as text or a file.
  Displays extracted entities in an intuitive format.
  Allows downloading the output as a CSV file.

Fine-tuned Models:
Fine-tuned SpaCy and BERT models specifically for entity recognition in address data.
  

Codebase:
  Python scripts for fine-tuning both SpaCy and BERT models.



Repository Structure

.
├── fine_tuning
│   ├── fine_tune_spacy.py
│   ├── fine_tune_bert.py
├── streamlit_app
│   ├── app.py
├── data
│   ├── sample_input.txt
│   ├── sample_output.csv
├── requirements.txt
├── README.md

How to Run the Project

Prerequisites
  1. Install the required dependencies:
       pip install -r requirements.txt
     
  2. Fine-Tuning the Models
     he repository includes Python scripts to fine-tune the SpaCy and BERT models. Before running these scripts, ensure you have the necessary data for fine-tuning
     Data Preprocessing the is big work for fine tuning the model
  In this project I choose sample data and annotate those manualy for BERT model and for the spacy model i use open source "NER Annotator for Spacy" which available in this git repo "https://github.com/tecoholic/ner-annotator?tab=readme-ov-file" with all the details

Run the following command to fine-tune the SpaCy model:

python fine_tuning/fine_tune_spacy.py

Fine-Tune BERT

Run the following command to fine-tune the BERT model:


Running the Streamlit App

Start the Streamlit application with the following command:

Using the Streamlit App

Input Text: You can either type/paste text directly into the input box or upload a text file.

View Results: The app displays the recognized entities using the selected model (SpaCy or BERT).

Download Output: If a text file is uploaded, the output entities can be downloaded as a CSV file

# Contributing

Feel free to contribute to this project. Fork the repository, create a new branch, make your changes, and submit a pull request.


