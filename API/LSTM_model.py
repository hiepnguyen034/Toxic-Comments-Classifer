import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D,InputLayer
from keras import metrics


data=pd.read_csv('toxic comments.csv').fillna('')
data=data.drop(['id'],1)

x_train,x_test,y_train,y_test=train_test_split(data['comment_text'],data.loc[:,data.columns != 'comment_text'],
                                                   test_size=0.01, random_state = 5)


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(list(x_train))
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

def lstm_model():
    d=0.5
    model=Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1],)))
    model.add(Embedding(max_words,128))
    model.add(LSTM(128,return_sequences=True,name='lstm_layer'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(d))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(6,activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model=lstm_model()
model.fit(x_train,y_train, batch_size=32, epochs=10, validation_data=(x_test,y_test))

if __name__ == '__main__':
	lstm_model()