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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

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

# method -2

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 101)

# LSTM model 

model = Sequential()
model.add(LSTM(150, return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dense(5, activation = 'softmax'))
model.add(Dropout(0.2))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
    
model.fit(train_x, train_y)

# Prediction

pred = model.predict(test_x)
pred_tags = decode_result(pred)
test_tags = decode_result(test_y)

# Confusion matrix

#def evaluate():
#    test_tags_cm, pred_tags_cm = [], []   
#    for i,j in zip(pred_tags, test_tags):
#        if j != null_class:
#            test_tags_cm.append(j)
#            pred_tags_cm.append(i)
#    return pred_tags_cm, test_tags_cm

def evaluate():
    test_tags_cm, pred_tags_cm = [], []   
    for i,j in zip(pred_tags, test_tags):
        test_tags_cm.append(j)
        pred_tags_cm.append(i)
    return pred_tags_cm, test_tags_cm

t1, t2 = evaluate()

predicted_tags = np.array(t1)
testing_tags = np.array(t2)
conf_matrix = confusion_matrix(testing_tags, predicted_tags)
all_tags = sorted(list(set(testing_tags)))
print(conf_matrix)
print(all_tags)
conf_matrix_test = pd.DataFrame(columns = all_tags, index = all_tags)    
print(conf_matrix_test)
for x, y in zip(conf_matrix, conf_matrix_test):
    conf_matrix_test[y] = x

print(conf_matrix_test)

## Tags prediction for a given sentence

def return_predicted_tags(sentence):
    sent_list = sentence.strip().split()
    len_sent = len(sent_list)
    vec = encode_sentence(sentence)
    tags = model.predict(vec)
    tags = tags[-len_sent:]
    tags_pred_sent = decode_result(tags)
    
    for word, tags in zip(sent_list, tags_pred_sent):
        print(word +"  ============>>>>  "+ tags)
        
sentence = """
    
Fencing Work Perimeter of Fencing 75x75mm & 10 Gauge GI Chain Link
    
    """    
return_predicted_tags(sentence)
    
## remove null arrays

#df = []
#for i in test_x:
#    for j in range(len(i)):
#        if i[j][0] == 0.0:
#            continue
#        else:
#            df.append(i[j])

            
    


            

        
            
                

        
    



