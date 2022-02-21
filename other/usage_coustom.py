import pickle
from keras.models import load_model
import numpy as np


with open('feature_vector_test.pickle', 'rb') as handle:
    features = pickle.load(handle)

model = load_model('coustom_model0.88.model')


x = np.array(features[0])
x = x.reshape(x.shape[0], x.shape[1], 1)

probabilities = model.predict_proba(x)

id_list = []
label_list = []
for i in range(len(probabilities)):
    id_list.append(features[1][i])
    label = probabilities[i][1]
    label_list.append(label)
    print(i)

import pandas as pd
export = pd.DataFrame()
export["unique_conversation_id"] = id_list
export["predictions"] = label_list

export.to_csv("./data/output/export_costum.csv",index=False)
print(8)
