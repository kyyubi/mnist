# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:45:23 2019

@author: KRISHNAN
"""
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,AveragePooling1D
#importing the dataset
sub = pd.read_csv("sample_submission.csv")
sub = sub.iloc[:,0].values
train = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#preprocessing
test_data = test_data.iloc[:,:].values
test_data = test_data.reshape(28000,784,1)
labels = train.iloc[:,0].values
data = train.iloc[:,1:].values
data = data
def values(d):
    for i in range(d.shape[1]):         #for converting the results into ones and zeros
        for j in range(d.shape[0]):
            if d[j,i]<0.5:
                d[j,i]=0
            else:
                d[j,i]=1
    return d

def one_hot_decode(t_d):
    t_decoded=np.zeros([t_d.shape[0],1],int)   #one_hot_decoder
    for i in range(t_d.shape[0]):
        for j in range(t_d.shape[1]):
            if t_d[i,j]==1:
               t_decoded[i,0]=j
    return t_decoded
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X = np.concatenate((X_train, X_test))
Y = np.concatenate((y_train,y_test))
X = X.reshape(70000,784,1)
Y = to_categorical(Y)
model= Sequential()

model.add(Conv1D(filters=6,kernel_size=5,strides=1,padding='valid',activation='relu',input_shape=(784,1)))
model.add(AveragePooling1D(pool_size=2,strides=2))
model.add(Conv1D(16,kernel_size=5,strides=1,padding='valid',activation='relu'))
model.add(AveragePooling1D(pool_size=2,strides=2))
model.add(Conv1D(120,kernel_size=1,strides=1,padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X,Y,batch_size=1000,epochs=100)
test_labels = model.predict(test_data)
test_labesl = values(test_labels)
test_labels = one_hot_decode(test_labesl)
import pandas as pd
test_sub = pd.DataFrame(test_labels,columns=['Label'])
sub = pd.DataFrame(sub,columns=['ImageId'])
pred1 = sub.join(test_sub)
pred1.set_index("ImageId",inplace=True)
pred1.to_csv("test_submission.csv")