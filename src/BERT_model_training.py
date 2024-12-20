import pandas as pd
from pyprojroot import here
import yaml
import json
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from transformers import TrainingArguments, Trainer
from utils import align_labels_tokens
from utils import prepare_datase

with open(r"/content/drive/MyDrive/Colab Notebooks/Lat_Log_NER_Project/config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

app_config["data_dir"]["annotated_data"]
# load the Annotated json file
def main(app_config):
    print(app_config["data_dir"]["annotated_data"])
    file_ = open(app_config["data_dir"]["annotated_data"],'r')
    annot_data = json.load(file_)

    train_data = pd.DataFrame(data=annot_data)
    label_list = [
        "O",
        "B-Point",
        "I-Point",
        "L-Point",
        "U-Point",
        "B-Road",
        "L-Road",
        "U-Pincode",
        "U-Country",
        "B-State",
        "L-State",
        "B-City",
        "I-City",
        "L-City",
    ]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_inputs = align_labels_tokens(tokenizer, train_data)

    num_labes = len(label_list)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labes
    )
    train_dataset = prepare_datase(tokenized_inputs)
    train_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=4,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer
    )
    trainer.train()

    model.save_pretrained(app_config["bert_mode_dir"]["FineTuned_Bert_model"])
    tokenizer.save_pretrained(app_config["bert_mode_dir"]["bert_model_tokenize"])


if __name__ == "__main__":
    main(app_config)
