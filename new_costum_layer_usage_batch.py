import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoConfig
# import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers.modeling_outputs import TokenClassifierOutput


class CustomModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model =  AutoModel.from_pretrained("./src/bert-fa-base-uncased",
                                                       config=AutoConfig.from_pretrained("./src/bert-fa-base-uncased",
                                                                                         output_attentions=True,
                                                                                         output_hidden_states=True))

        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3)
        # self.relu2 = nn.ReLU()
        self.maxpoling = nn.MaxPool1d(3, stride=2)
        self.linear1 = nn.Linear(62976, 2)
        self. flatten = nn.Flatten()
        self.linear2 = nn.Linear(128, 64)## 3 is the number of classes in this example
        self.softmax = nn.Softmax()
        self.classifier = nn.Linear(64, num_labels)  # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers

        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        conv_out = self.conv2(sequence_output[:, :, :].view(-1,768,250))
        max_out = self.maxpoling(conv_out)
        flatten_out = self.flatten(max_out)

        linear1_output = self.linear1(flatten_out)  ## extract the 1st token's embeddings
        # linear2_output = self.linear2(linear1_output)


        # logits = self.classifier(linear2_output[:, :])  # calculate losses
        logits = self.softmax(linear1_output)


        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

df = pd.read_csv("./data/final_test.csv", encoding="utf-8", quoting=1)
# df['unique_id'] = pd.factorize(df['label'])[0]
# df = df[df.unique_id != -1]

sentences = df.text.values
# labels = df.unique_id.values[-10:-1]
ids = df.unique_conversation_id.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
tokenizer = AutoTokenizer.from_pretrained("./src/bert-fa-base-uncased")

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=250,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Set the batch size.
batch_size = 32

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model = torch.load("./src/Output_models/BERT_model0.51_costum.model")
model.eval()
model.cuda()
device = torch.device("cuda:0")

# Tracking variables
predictions, true_labels = [], []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    # label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.extend(logits)
    # true_labels.append(label_ids)
    print(len(predictions))
print('    DONE.')


def percent(arrey):
    min_num = -min(arrey)
    new = []
    for n in arrey:
        new.append(n + min_num + 0.0001)

    sum_num = sum(new)
    percen = []
    for m in new:
        percen.append((100 * m) / sum_num)

    return percen


pred_label = []
for p in predictions:
    pred = percent(p)
    pred_label.append(pred[1] / 100)

final_df = pd.DataFrame()
final_df["unique_conversation_id"] = df["unique_conversation_id"]
final_df["predictions"] = pred_label

final_df.to_csv("./data/output/export_train_full_costum2.csv", index=False)
print(len(df))
print(len(final_df))
