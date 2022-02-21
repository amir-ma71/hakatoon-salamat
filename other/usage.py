import pandas as pd
from flask import Flask, request
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from langdetect import detect, DetectorFactory


def prepare_input(sent):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    tokenizer = AutoTokenizer.from_pretrained("./src/bert-fa-base-uncased")

    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=250,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids = encoded_dict['input_ids']

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks = encoded_dict['attention_mask']

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler)

    return prediction_dataloader


def percent(arrey):
    min_num = -min(arrey)
    new = []
    for n in arrey:
        new.append(n + min_num + 0.01)

    sum_num = sum(new)
    percen = []
    for m in new:
        percen.append((100 * m) / sum_num)

    return percen


model = torch.load("./src/Output_models/BERT_model-max250-0.85.model", map_location=torch.device('cpu'))
model = model.to("cpu")
model.eval()

c = 0
def predict_text(sent):

    # Prepare the text
    prediction_dataloader = prepare_input(sent)

    b_input_ids, b_input_mask = prediction_dataloader.dataset.tensors

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.numpy()

    pred_percent = percent(logits[0])


    label = pred_percent[1]

    global c
    print(c)
    c +=1
    return label


df = pd.read_csv("final_test.csv", quoting=1, encoding="utf-8")
final_df = pd.DataFrame()
final_df["unique_conversation_id"] = df["unique_conversation_id"]

final_df["predictions"] = df["text"].apply(lambda x: predict_text(x))

final_df.to_csv("export.csv",index=False)
print(len(final_df))
print(len(df))

