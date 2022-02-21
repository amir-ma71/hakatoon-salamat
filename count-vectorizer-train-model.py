

from tensorflow import keras
# import segmentation_models as sm
from keras import backend as K
# from tensorflow import keras
import tensorflow.keras
from keras import Sequential, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Conv1D, Flatten, MaxPool1D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
tf.device("gpu:0")


df = pd.read_csv("./data/data_with_feature_vector.csv", quoting=1)
df = df.sample(frac=1)
print(len(df))
df = df[df['vector'].notna()]
print(len(df))
x_s = df.vector.values
y_s = df.label.values
x = []
y = []
for i in range(len(x_s)):
    x.append(np.fromstring(x_s[i][1:-1], dtype=np.int, sep=' '))
    y.append(np.fromstring(y_s[i][1:-1], dtype=np.int, sep=' '))


x = np.array(x)
y = np.array(y)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, random_state=42)
print(8)

MAX_NB_WORDS = 800000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, 100, input_length=x.shape[1]))
# model.add(SpatialDropout1D(0.2))
model.add(Conv1D(256,3, activation='relu'))
model.add(MaxPool1D())
model.add(Dropout(0.1))
model.add(Conv1D(256,3))
model.add(MaxPool1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.005)


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

epochs = 3
batch_size = 200

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print("loss -->", loss)
print("accuracy -->", accuracy)



y_pred = model.predict(X_test)


def conf(true, pred, label_list):
    true_int = np.argmax(true, axis=1)
    pred_int = np.argmax(pred, axis=1)

    matrix = np.zeros((len(label_list), len(label_list)), dtype=int)
    for i in range(len(true_int)):
        matrix[true_int[i]][pred_int[i]] += 1

    return matrix


def print_matrix(matrix,label_list):
    count = 0
    print("\t\t", end="\t")
    for l in label_list:
        count += 1
        if count != len(label_list):
            print(l, end="\t")
        else:
            print(l)
    for row_label, row in zip(label_list, matrix):
        print('%s \t %s' % (row_label, '\t'.join('\t%03s' % i for i in row)))
    validate(matrix, label_list)

def validate(matrix, label_list):
    # recall
    for i in range(len(label_list)):
        total_recall = 0
        total_prec = 0
        label = label_list[i]
        for j in range(len(label_list)):
            if i == j:
                sorat = matrix[i][j]
                total_recall = total_recall + matrix[i][j]
                total_prec = total_prec + matrix[j][i]
            else:
                total_recall = total_recall + matrix[i][j]
                total_prec = total_prec + matrix[j][i]

        print(label.upper(), "metric: ", end="\t")
        if sorat != 0:
            recall = sorat/total_recall
            precision = sorat/total_prec
            print("Recall= %.2f" % recall , end="\t")
            print("Precision= %.2f" % precision , end="\t")
        else:
            print("Recall= ", "NaN", end="\t")
            print("Precision= ", "NaN", end="\t")

        f1 = 2*((precision*recall)/(precision+recall))
        print("F1 Score= %.2f" % f1)

l_list_conf = ["false","true"]
print_matrix(conf(Y_test, y_pred, l_list_conf),l_list_conf)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
