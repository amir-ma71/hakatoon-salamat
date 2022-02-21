import pickle

from keras import Sequential, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Conv1D, Flatten, MaxPool1D, MaxPool2D
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

print("reading Data . . .")
with open('feature_vector.pickle', 'rb') as handle:
    features = pickle.load(handle)


x = np.array(features[0])
x = x.reshape(x.shape[0], x.shape[1], 1)
y = np.array(features[1])
print("vectorizing Done . . .")

from keras import backend as K




X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, random_state=42)
print(8)

MAX_NB_WORDS = 100000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250

model = Sequential()

# model.add(Embedding(MAX_NB_WORDS, 100, input_length=x.shape[1]))
# model.add(SpatialDropout1D(0.2))
model.add(Conv1D(512,3, activation='sigmoid', input_shape=(8,1)))
model.add(MaxPool1D())
model.add(Dropout(0.3))
model.add(Conv1D(256,3,activation='sigmoid'))
# model.add(MaxPool2D())
# model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 256

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print("loss -->", loss)
print("accuracy -->", accuracy)



y_pred = model.predict(X_test)
print(y_pred)
probabilities = model.predict_proba(X_test)
print(probabilities)
l_list_conf = ["news", "sport", "healt", "enter", "econ", "tech"]
# print_matrix(conf(Y_test, y_pred, l_list_conf),l_list_conf)

model.save("coustom_model{0:.2f}.model".format(accuracy))