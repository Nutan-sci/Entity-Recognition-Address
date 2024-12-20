import streamlit as st
from pyprojroot import here
import yaml
from dotenv import load_dotenv
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
import pandas as pd
from utils import predict
from utils import bilou_to_chunks, merge_tokens
import spacy

with open(
    here(r"C:\Users\nutan\nutan\interview\LAT_LAG_Project_FINAL\sample.json")
) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)


def read_txt_file(file_path):
    file_bytes = file_path.read()
    file_data = file_bytes.decode("utf-8").split("\n")
    return file_data


@st.cache_data
def load_models():
    bert_model = BertForTokenClassification.from_pretrained(
        r"C:\Users\nutan\nutan\interview\LAT_LAG_Project_FINAL\saved_models\bert_models\Lat_Lag_NER"
    )
    tokenizer_ = BertTokenizerFast.from_pretrained(
        r"C:\Users\nutan\nutan\interview\LAT_LAG_Project_FINAL\saved_models\bert_models\tokenizer"
    )
    spacy_model = spacy.load(
        r"C:\Users\nutan\nutan\interview\LAT_LAG_Project_FINAL\saved_models\spacy_models\model-last"
    )
    return bert_model, tokenizer_, spacy_model


def bert_text_output(bert_model, tokenizer_, text_input):
    tokens, labels = predict(bert_model, tokenizer_, text_input)
    chunks = bilou_to_chunks(tokens, labels)
    output_list = merge_tokens(chunks)
    return output_list


def spacy_text_output(spacy_model, text_input):
    output = spacy_model(text_input)
    output_list = [(ent.text, ent.label_) for ent in output.ents]
    return output_list


def bert_output_df(bert_model, tokenizer_, text):
    columns = ["Point", "Road", "City", "State", "Pincode", "Country"]
    data = []
    for i in text:
        tokens, labels = predict(bert_model, tokenizer_, i)
        chunks = bilou_to_chunks(tokens, labels)
        output_list = merge_tokens(chunks)

        # print(output_list)
        result = {col: None for col in columns}
        for value, label in output_list:
            if label in result:
                if result[label] is None:
                    result[label] = value
                else:
                    result[label] += f" ,{value}"
        result["Address"] = i
        data.append(result)
    output_df = pd.DataFrame(
        data,
        columns=["Address", "Point", "Road", "City", "State", "Pincode", "Country"],
    )

    return output_df


def spacy_output_df(spacy_model, text):
    columns = ["POINT", "ROAD", "CITY", "STATE", "PINCODE", "COUNTRY", "O"]
    data = []
    for row in text:
        output = spacy_model(row)
        output_list = [(ent.text, ent.label_) for ent in output.ents]
        result = {col: None for col in columns}
        for value, label in output_list:
            if label in result:
                if result[label] is None:
                    result[label] = value
                else:
                    result[label] += f", {value}"
        result["ADDRESS"] = row
        data.append(result)
    output_df = pd.DataFrame(
        data,
        columns=[
            "ADDRESS",
            "POINT",
            "ROAD",
            "CITY",
            "STATE",
            "PINCODE",
            "COUNTRY",
            "O",
        ],
    )
    return output_df


def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def main():
    load_dotenv()
    # st.set_page_config(page_title="Entities Recogintion", page_icon=":home:")
    st.header("Entities Recogintion from Address ")
    input_text = st.text_input("Enter The Address")
    st.write("OR")
    uploaded_file = st.file_uploader("upload your file here", type=["txt", "pdf"])
    # st.write("Select the models")

    # options = st.radio("Select the model  👉", options=])

    if input_text is not None:
        output = spacy_text_output(spacy_model, input_text)
        output_text = bert_text_output(bert_model, tokenizer_, input_text)

    if uploaded_file is not None:
        rows = read_txt_file(uploaded_file)
        output_file = bert_output_df(bert_model, tokenizer_, rows)
        bert_csv = convert_df(output_file)

        output_file_spacy = spacy_output_df(spacy_model, rows)
        spacy_csv = convert_df(output_file_spacy)

    options = st.selectbox(
        "Select the model below",
        ("BERT", "SPACY"),
        index=None,
        placeholder="Select model to run...",
    )

    if options == "BERT":
        if output_text:
            st.write_stream(output_text)

        if bert_csv:
            st.write(output_file.head())

            st.download_button(
                "Press to Download CSV File",
                bert_csv,
                "file.csv",
                "text/csv",
                key="download-BERT csv",
            )

    if options == "SPACY":
        if output:
            st.write_stream(output)

        if spacy_csv:
            st.write(output_file_spacy.tail())
            st.download_button(
                "Press to Download spacy output CSV File",
                spacy_csv,
                "file.csv",
                "text/csv",
                key="download- spacy csv",
            )


bert_model, tokenizer_, spacy_model = load_models()
if __name__ == "__main__":
    main()
