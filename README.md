# Entity-Recognition-Address
This project focuses on recognizing entities within address data using fine-tuned models (SpaCy and BERT). The project includes a Streamlit application that provides a user-friendly interface for:
  * Inputting address text (either directly or via a text file).
  * Processing the input with fine-tuned SpaCy and BERT models to extract entities.
  * Downloading the results as a CSV file when a text file is provided as input.

## Features
- **Streamlit App:**
    - Accepts input as text or a file.
    - Displays extracted entities in an intuitive format.
    - Allows downloading the output as a CSV file.

- **Fine-tuned Models:**
    - Fine-tuned SpaCy and BERT models specifically for entity recognition in address data.
  

- **Codebase:**
    - Python scripts for fine-tuning both SpaCy and BERT models.


## How to Run the Project

### Prerequisites
  1. Install the required dependencies:
       pip install -r requirements.txt
     
  2. Fine-Tuning the Models
     he repository includes Python scripts to fine-tune the SpaCy and BERT models. Before running these scripts, ensure you have the necessary data for fine-tuning
    
#### Fine-Tune Spacy
Run the following command to fine-tune the SpaCy model:
    python src/BERT_model_training.py

#### Fine-Tune BERT
  Run the following command to fine-tune the BERT model:
  python src/spacy_model_training.py


### Running the Streamlit App
Start the Streamlit application with the following command:
  streamlit run  src/app.py
  
### Using the Streamlit App
  
1. Input Text: You can either type/paste text directly into the input box or upload a text file.

2. View Results: The app displays the recognized entities using the selected model (SpaCy or BERT).

3. Download Output: If a text file is uploaded, the output entities can be downloaded as a CSV file

## Contributing

Feel free to contribute to this project. Fork the repository, create a new branch, make your changes, and submit a pull request.


