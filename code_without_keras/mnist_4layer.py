# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:46:04 2019

@author: KRISHNAN
"""
def one_hot_encoder(new_labels):
    for i in range(new_labels.shape[0]):  #one_hot_encoder
        num = int(labels[i])
        new_labels[i,num] = 1
    return new_labels
def one_hot_decode(t_d):
    t_decoded=np.zeros([t_d.shape[0],1],int)   #one_hot_decoder
    for i in range(t_d.shape[0]):
        for j in range(t_d.shape[1]):
            if t_d[i,j]==1:
               t_decoded[i,0]=j+1
    return t_decoded
def weight_init(X,Y):
    W1=np.random.randn(X.shape[1],n_h1)*np.sqrt(2./X.shape[1])
    b1=np.zeros([1,n_h1],int)
    W2=np.random.randn(n_h1,n_h2)*np.sqrt(2./n_h1)
    b2=np.zeros([1,n_h2],int)
    W3=np.random.randn(n_h2,n_h3)*np.sqrt(2./n_h2)
    b3=np.zeros([1,n_h3],int)
    W4=np.random.randn(n_h3,Y.shape[1])*np.sqrt(1./n_h3)
    b4=np.zeros([1,Y.shape[1]],int)
    return W1,W2,b1,b2,W3,b3,W4,b4

def sigmoid(H):
    v=1/(1+np.exp(-H))
    return v
def sigmoid_der(H):
    return sigmoid(H)*(1-sigmoid(H))
def relu(x):
    z=np.zeros_like(x)
    return np.where(x>z,x,z)
def relu_der(v):
    v1=v>0
    return v1
def tanh(s):
    s1 = np.exp(s)+np.exp(-s)
    s2 = np.exp(s)-np.exp(-s)
    v = s1/s2
    print(v)
    return v
def predict(X):
    H1 = np.dot(X,W1) + b1
    V1 = relu(H1)
    H2 = np.dot(V1,W2) + b2
    V2 = relu(H2)
    H3 = np.dot(V2,W3) +b3
    V3=relu(H3)
    H4= np.dot(V3,W4) + b4
    d=sigmoid(H4)
    d=values(d)
    return d
def forward(W1,W2,W3,W4,b1,b2,b3,b4,X):
    H1 = np.dot(X,W1) + b1
    V1 = relu(H1)
    d1=np.random.rand(V1.shape[0],V1.shape[1])<keep_prob1   #forward propagation with dropout technique
    d1=d1/keep_prob1
    V1_new=np.multiply(d1,V1)
    H2 = np.dot(V1_new,W2) + b2
    V2 = relu(H2)
    d2=np.random.rand(V2.shape[0],V2.shape[1])<keep_prob2
    d2=d2/keep_prob2
    V2_new=np.multiply(d2,V2)
    H3 = np.dot(V2_new,W3) + b3
    V3 = relu(H3)
    d3=np.random.rand(V3.shape[0],V3.shape[1])<keep_prob3
    d3=d3/keep_prob3
    V3_new=np.multiply(d3,V3)
    H4 = np.dot(V3_new,W4) + b4
    d = sigmoid(H4)
    return d,V1_new,V2_new,V3_new

def backpropagation(X,d,V1,V2,V3,W2,W3,W4,Y):
    m=Y.shape[0]
    del4=(d-Y)/m
    dW4 = V3.T.dot(del4)
    db4=np.sum(del4,axis=0,keepdims=True)
    del3=np.dot(del4,(W4.T))*relu_der(V3)
    dW3= V2.T.dot(del3)
    db3=np.sum(del3,axis=0,keepdims=True)
    del2=np.dot(del3,(W3.T))*relu_der(V2)
    dW2= V1.T.dot(del2)
    db2=np.sum(del2,axis=0,keepdims=True)
    del1=np.dot(del2,(W2.T))*relu_der(V1)
    dW1= X.T.dot(del1)
    db1=np.sum(del1,axis=0,keepdims=True)
    return dW1,dW2,dW3,dW4,db1,db2,db3,db4

def weight_updation(W1,W2,W3,W4,b1,b2,b3,b4,dW1,dW2,dW3,dW4,db1,db2,db3,db4,learning_rate):
    W4 = W4 - learning_rate*dW4
    W3 = W3 - learning_rate*dW3
    W2 = W2 - learning_rate*dW2
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    b3 = b3 - learning_rate*db3
    b4 = b4 - learning_rate*db4
    return W1,W2,W3,W4,b1,b2,b3,b4

def values(d):
    for i in range(d.shape[1]):         #for converting the results into ones and zeros
        for j in range(d.shape[0]):
            if d[j,i]<0.5:
                d[j,i]=0
            else:
                d[j,i]=1
    return d

#importing the required libraries
import numpy as np
import pandas as pd
#importing the dataset
sub = pd.read_csv("sample_submission.csv")
sub = sub.iloc[:,0].values
train = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#preprocessing
test_data = test_data.iloc[:,:].values
test_data = test_data/255.0
labels = train.iloc[:,0].values
data = train.iloc[:,1:].values
data = data/255.0
new_labels = np.zeros([42000,10],int)
new_labels = one_hot_encoder(new_labels)
new_labels = new_labels[:,1:]
I = int(new_labels.shape[0]*0.3)
validation_data = data[0:I,:]
validation_data = validation_data
train_data = data[I:,:]
train_data = train_data
train_labels = new_labels[I:,:]
train_labels = train_labels
validation_labels = new_labels[0:I,:]
validation_labels = validation_labels
n_h1 = 256
n_h2 = 128
n_h3 = 64
keep_prob1=0.6
keep_prob2=0.7
keep_prob3=0.8
iters=7000
#initilization
W1,W2,b1,b2,W3,b3,W4,b4=weight_init(train_data,train_labels)
#iterations
for i in range(iters):
    d,V1,V2,V3=forward(W1,W2,W3,W4,b1,b2,b3,b4,train_data)
    dW1,dW2,dW3,dW4,db1,db2,db3,db4=backpropagation(train_data,d,V1,V2,V3,W2,W3,W4,train_labels)
    error = (1./train_labels.shape[0])*np.sum((train_labels*np.log(d) + (1-train_labels)*np.log(1-d)))
    W1,W2,W3,W4,b1,b2,b3,b4=weight_updation(W1,W2,W3,W4,b1,b2,b3,b4,dW1,dW2,dW3,dW4,db1,db2,db3,db4,0.1)
    print(i,error)


#decoding and checking the score of the end results for train dataset
d=values(d)
train_d=one_hot_decode(d)
train_e=one_hot_decode(train_labels)
from sklearn.metrics import accuracy_score
score=accuracy_score(train_e,train_d)
#decoding and checking the score of the end results for validation dataset
valid = predict(validation_data)
valid_d=one_hot_decode(valid)
valid_e=one_hot_decode(validation_labels)
valid_score=accuracy_score(valid_e,valid_d)
#predicting the test dataset
test_labels=predict(test_data)
test_labels=one_hot_decode(test_labels)
#saving it as a .csv file
test_sub = pd.DataFrame(test_labels,columns=['Label'])
sub = pd.DataFrame(sub,columns=['ImageId'])
pred1 = sub.join(test_sub)
pred1.set_index("ImageId",inplace=True)
pred1.to_csv("test_submission.csv")



#saving the parameters
weights1=pd.DataFrame(W1)
weights2=pd.DataFrame(W2)
weights3=pd.DataFrame(W3)
weights4=pd.DataFrame(W4)
b1=pd.DataFrame(b1)
b2=pd.DataFrame(b2)
b3=pd.DataFrame(b3)
b4=pd.DataFrame(b4)
W1=pd.read_csv("w1.csv")
W1=W1.iloc[:,1:].values
W2=pd.read_csv("w2.csv")
W2=W2.iloc[:,1:].values
W3=pd.read_csv("w3.csv")
W3=W3.iloc[:,1:].values
W4=pd.read_csv("w4.csv")
W4=W4.iloc[:,1:].values
b1=pd.read_csv("b1.csv")
b1=b1.iloc[:,1:].values
b2=pd.read_csv("b2.csv")
b2=b2.iloc[:,1:].values
b3=pd.read_csv("b3.csv")
b3=b3.iloc[:,1:].values
b4=pd.read_csv("b4.csv")
b4=b4.iloc[:,1:].values
