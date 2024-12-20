from datasets import Dataset

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

id2label = {i: label for i, label in enumerate(label_list)}


def align_labels_tokens(tokenizer, data, label_all_tokens=True):
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    data (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.

    label_all_tokens (bool): A flag to indicate whether all tokens should have labels.
                             If False, only the first token of a word will have a label,
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(
        list(data["tokens"]), truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(data["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# preparing data for training
def prepare_datase(tokenized_data):
    # input_ids = [item["input_ids"] for item in tokenized_data]
    # attention_mask = [item['attention_mask'] for item in tokenized_data]
    # labels= [item['labels'] for item in tokenized_data]
    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": tokenized_data["labels"],
        }
    )

    return dataset


## inference
def predict(model, tokenizer, address):
    tokenized = tokenizer(
        address, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    # with torch.no_grad():
    outputs = model(**tokenized)
    predictions = outputs.logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"].squeeze())
    # print(predictions)
    labels = [id2label.get(p.item(), "O") for p in predictions.squeeze()]
    return tokens, labels


def bilou_to_chunks(tokens, labels):
    """
    Convert BILOU-tagged tokens into entity chunks.

    Args:
        tokens (list of str): List of tokens.
        labels (list of str): List of BILOU labels corresponding to the tokens.

    Returns:
        List of tuples: Each tuple contains a chunk (string) and its entity type.
    """
    chunks = []  # Final list of entity chunks
    current_chunk = []  # Temporary list to hold tokens for the current chunk
    current_label = None  # Current entity label

    for token, label in zip(tokens, labels):
        if label == "O":  # Outside any entity
            if current_chunk:  # If a chunk was being built, finalize it
                chunks.append((" ".join(current_chunk), current_label))
                current_chunk = []
                current_label = None
        elif label.startswith("B-"):  # Beginning of a new entity
            if current_chunk:  # Finalize the previous chunk
                chunks.append((" ".join(current_chunk), current_label))
            current_chunk = [token]
            current_label = label[2:]  # Extract entity type
        elif label.startswith("I-"):  # Inside the current entity
            current_chunk.append(token)
        elif label.startswith("L-"):  # Last token of the current entity
            current_chunk.append(token)
            chunks.append((" ".join(current_chunk), label[2:]))
            current_chunk = []
            current_label = None
        elif label.startswith("U-"):
            # Unit-length entity
            chunks.append((token, label[2:]))
            current_chunk = []
            current_label = None
        elif label.startswith("##"):  # Handles subwords (continuations)
            if current_chunk:  # Add continuation to the current chunk
                current_chunk[-1] += token[2:]
        else:
            raise ValueError(f"Unexpected label: {label}")

    # Finalize any remaining chunk
    if current_chunk:
        chunks.append(("".join(current_chunk), current_label))

    return chunks


def merge_tokens(input_list):
    result = []  # Final processed list
    temp_token = ""  # Temporary token to merge substrings
    current_tag = None  # Keeps track of the current tag

    for token, tag in input_list:
        if token.startswith("##"):  # If the token is a continuation (starts with ##)
            temp_token += token[2:]  # Merge the substring without '##'
        else:
            # Append the previous token and tag to the result if a new tag starts
            if temp_token and current_tag is not None:
                result.append((temp_token, current_tag))
            temp_token = token  # Start a new token
            current_tag = tag  # Update the current tag

    # Append the last token in the list
    if temp_token and current_tag is not None:
        result.append((temp_token, current_tag))

    final_check = []
    for token, tag in result:
        if "##" in token:
            temp_word = ""
            tokens = []
            for word in token.split():
                if word.startswith("##"):
                    temp_word += word[2:]
                else:
                    tokens.append(temp_word)
                    temp_word = word
            tokens.append(temp_word)
            tokens = " ".join(tokens)
            final_check.append((tokens, tag))
        else:
            final_check.append((token, tag))

    return final_check
