import pandas as pd
import json
import re


char_fa = [ "ء", "آ", "ی", "ه", "و", "ن", "م", "ل", "گ", "ک", "ق", "ف", "غ", "ع", "ظ", "ط", "ض", "ص", "ش", "س", "ژ",
          "ز", "ر", "ذ", "د", "خ", "ح", "چ", "ج", "ث", "ت", "پ", "ب", "ا","ى"," " ]

def cleaner_for_key2(text):
    r = ""
    # جهت پاک کردن متون چینی اول و آخر خط     #
    f = list(text.replace("ي","ی").replace("ى","ی").replace("ة","ه").replace("ۀ","ه").replace("ؤ","و").replace("إ","ا").replace("أ","ا").replace("ك","ک").replace("ـ","").replace("ئ","ی").replace("ّ","").replace("ً","").replace("ٌ","").replace("ٍ","").replace("َ","").replace("ُ","").replace("ِ",""))
    for j in f:
        if j not in char_fa:
            j = " "
        r = r + j
    return (re.sub(' +',' ',r).strip())


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

df = pd.read_parquet("./data/train.parquet")
text_list = []
label_list = []
id_list = []
for i in range(len(df["messages"])):
    text = ""
    messages = json.loads(df["messages"][i])
    for msg in messages:
        if msg["message_type"] == 1:
            text_message = msg["message_content"]
            text = text + " " + text_message


    # text = clean_str(text)
    text = cleaner_for_key2(text)
    if len(text) == 0:
        text_list.append("متنی موجود نیست")
    else:
        text_list.append(text)
    label_list.append(df["label"][i])
    id_list.append(df["unique_conversation_id"][i])
    print(i)


print(len(text_list))
print(len(label_list))
train_dataset = pd.DataFrame()
train_dataset["text"] = text_list
train_dataset["label"] = label_list
train_dataset["unique_conversation_id"] = id_list

train_dataset.to_csv("final_train_full.csv", quoting=1, index=False, encoding="utf-8")




