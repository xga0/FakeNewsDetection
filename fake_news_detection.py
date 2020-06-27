#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:06:20 2020

@author: seangao
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, Dense, Dropout, GlobalMaxPool1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#DATA LOAD
fakepath = '../input/fake-and-real-news-dataset/Fake.csv'
truepath = '../input/fake-and-real-news-dataset/True.csv'

fake = pd.read_csv(fakepath)
true = pd.read_csv(truepath)

df_fake = fake[['title', 'text']]
df_fake['label'] = len(df_fake.index) * [0]

df_true = true[['title', 'text']]
df_true['label'] = len(df_true.index) * [1]
df_true['text'] = df_true['text'].apply(lambda x: x[x.find('-')+1:]) #REMOVE Reuters INFO

df = df_true.append(df_fake, ignore_index=True)

#PREPROCESSING
df['text'] = df['text'].apply(lambda x: x.strip()) 

df['title'] = df['title'].apply(lambda x: re.sub('\[[^]]*\]', '', x)) 
df['text'] = df['text'].apply(lambda x: re.sub('\[[^]]*\]', '', x)) 

df['title'] = df['title'].apply(lambda x: re.sub('[()]', '', x)) 
df['text'] = df['text'].apply(lambda x: re.sub('[()]', '', x)) 

stpw = stopwords.words('english')

def stpwfix(input_string):
    output_string = []
    for i in input_string.split():
        if i.strip().lower() not in stpw:
            output_string.append(i.strip())
    return ' '.join(output_string)

df['title'] = df['title'].apply(lambda x: stpwfix(x)) 
df['text'] = df['text'].apply(lambda x: stpwfix(x)) 

def addperiod(input_string):
    if not input_string.endswith('.'):
        input_string += '.'
        return input_string
    else:
        return input_string
    
df['title'] = df['title'].apply(lambda x: addperiod(x))    

df['concat'] = df[['title', 'text']].agg(' '.join, axis=1) 
df['concat'] = df['concat'].apply(lambda x: re.sub(' +', ' ', x))

#PREPARE GLOVE EMBEDDING
embed_size = 100
maxlen = 1000

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['concat'].values)

X = tokenizer.texts_to_sequences(df['concat'].values)
X = pad_sequences(X, maxlen = maxlen)

y = df['label'].to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42)

embedding_path = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

embeddings_index = {}
f = open(embedding_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#BUILD AND TRAIN MODEL
model = Sequential()
model.add(Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix]))
model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(GlobalMaxPool1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=100, callbacks=[es])

epochs = [i for i in range(1, 8)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(10,5)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

y_pred = model.predict(X_test, verbose=1)
y_pred_label = y_pred > 0.5

confusion_matrix(y_pred_label, y_test)
tn, fp, fn, tp = confusion_matrix(y_pred_label, y_test).ravel()
(tn, fp, fn, tp)
