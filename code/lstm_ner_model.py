# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras import metrics

import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix, classification_report


# data preprocesing 
x, y, maxLen = read_data("../data/eng_data.txt")

# train-test split

def train_test(x, y, train_split = 0.8):
    rand = np.random.rand(len(x))
    split =  rand < (train_split)
    train_x = x[split]
    train_y = y[split]
    test_x = x[~split]
    test_y = y[~split]
    print(rand)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y
    

train_x, train_y, test_x, test_y = train_test(x, y, train_split = 0.8)

# reshape to 3D

train_X = np.reshape(train_x, (1, train_x.shape[0], train_x.shape[1]))


    
def model_fit(x, y):
    model = Sequential()
    model.add(LSTM(150, return_sequences = True, input_shape = (train_x.shape[1], 60)))
    model.add(Dense(5, activation = 'softmax'))
    model.add(Dropout(0.2))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(train_x, train_y, batch_size = 50)
    print(model.summary())
    
model_fit(train_X, train_y)
