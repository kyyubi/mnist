# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
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
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto',dtype='int',handle_unknown='ignore')
labels=labels.reshape(-1,1)
X = onehotencoder.fit_transform(labels).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(data, X, test_size = 0.2, random_state = 0)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100,epochs = 500)

# Part 3 - Making the predictions and evaluating the model
from sklearn.metrics import accuracy_score
valid = classifier.predict(X_valid)
valid = (valid > 0.5)
valid_pred=onehotencoder.inverse_transform(valid)
valid_expected=onehotencoder.inverse_transform(y_valid)
valid_score=accuracy_score(valid_expected,valid_pred)
# Predicting the Test set results
y_pred = classifier.predict(test_data)
y_pred = (y_pred > 0.5)
test_labels=onehotencoder.inverse_transform(y_pred)
#saving it as a .csv file
test_sub = pd.DataFrame(test_labels,columns=['Label'])
sub = pd.DataFrame(sub,columns=['ImageId'])
pred1 = sub.join(test_sub)
pred1.set_index("ImageId",inplace=True)
pred1.to_csv("test_submission1.csv")
