from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from pyprojroot import here
import yaml
from utils import predict
from utils import bilou_to_chunks, merge_tokens
import spacy
import pandas as pd


with open(
    here("C:\\Users\\nutan\\nutan\\interview\\Lat_Log_NER_Project\\config.yml")
) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)


# if uploaded_file is not None:
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


def read_txt_file(file_path):
    file_bytes = open(file_path, "r")
    file_data = file_bytes.read().split("\n")
    return file_data


bert_model, tokenizer_, spacy_model = load_models()
rows = read_txt_file(
    r"C:\Users\nutan\nutan\interview\LAT_LAG_Project_FINAL\data\sample.txt"
)
output_file = spacy_output_df(spacy_model, rows)
print(output_file)
# spacy_csv = convert_df(output_file)
# st.write(output_file.head())
# st.download_button(
#     "Press to Download spacy output CSV File",
#     spacy_csv,
#     "file.csv",
#     "text/csv",
#     key="download- spacy csv",
# )
