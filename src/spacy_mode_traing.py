from pyprojroot import here
import yaml
import spacy
from spacy.tokens import DocBin
import json

from pathlib import Path
from spacy.cli.download import download
from spacy.cli.init_config import fill_config
from spacy.cli.train import train

with open("/content/drive/MyDrive/Colab Notebooks/Lat_Log_NER_Project/config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)


def main(app_config):
    # Load The spacy Model
    nlp = spacy.blank("en")
    database = DocBin()

    # Load the annotated json file data
    sample_file = open(app_config["data_dir"]["spacy_annotated"])
    train_data = json.load(sample_file)

    for text, annot in train_data["annotations"]:
        docs = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = docs.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                continue
                # print("skipped the Entity")
            else:
                ents.append(span)

        docs.ents = ents
        database.add(docs)

    database.to_disk(app_config["data_dir"]["spacy_train_data"])

    train_data_path = app_config["data_dir"]["spacy_train_data"]
    config_file_path = app_config["spacy_model_dir"]["spcay_config"]
    model_path = app_config["spacy_model_dir"]["spacy_model_path"]
    download("en_core_web_lg")
    fill_config(Path("config.cfg"), Path(config_file_path))
    train(
        Path("config.cfg"),
        Path(model_path),
        overrides={"paths.train": train_data_path, "paths.dev": train_data_path},
    )


if __name__ == "__main__":
    main(app_config)
