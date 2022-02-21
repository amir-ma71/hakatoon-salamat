from keras.models import load_model
import pandas as pd
import numpy as np
from keras import backend as K


df = pd.read_csv("./data/test_data_with_feature_vector.csv", quoting=1)
df = df.sample(frac=1)
print(len(df))
df = df[df['vector'].notna()]
print(len(df))
x_s = df.vector.values
x = []
for i in range(len(x_s)):
    x.append(np.fromstring(x_s[i][1:-1], dtype=np.int, sep=' '))


x = np.array(x)


model = load_model('my_model.h5')

preds = model.predict(x)

pred_list = []
for p in preds:
    pred_list.append(p[1])
export = pd.DataFrame()

export["unique_conversation_id"] = df["unique_conversation_id"]
export["predictions"] = pred_list

export.to_csv("export-count-vec.csv", index=False)
print(9)