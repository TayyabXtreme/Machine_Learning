

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection

"""

dataset=pd.read_csv('/content/diabetes.csv')

dataset.head()

dataset.shape

dataset.describe()

dataset.Outcome.value_counts()

"""0 --> Non Diabatics

1 --> Diabetics
"""

dataset.groupby('Outcome').mean()

dataset.shape

X=dataset.drop(columns='Outcome',axis=1)
Y=dataset['Outcome']

print(X)

print(Y)

"""Data Standardization"""

scalar=StandardScaler()

scalar.fit(X)

scaled_data=scalar.transform(X)

print(scaled_data)

X=scaled_data
Y=dataset['Outcome']

"""Train Test Split"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

"""Trainning The Model

"""

classifier=svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

"""Accuracy Score"""

X_train_accuracy=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_accuracy,Y_train)

print('Accuracy score of the training data : ',training_data_accuracy)

X_test_accuracy=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_accuracy,Y_test)

print('Accuracy score of the test data : ',test_data_accuracy)

"""Making a Prdicictive System"""

input_data=(9,156,86,28,155,34.3,1.189,42)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scalar.transform(input_data_reshaped)
print(std_data)

prediciton=classifier.predict(std_data)
if (prediciton[0]==0):
  print('The person is not diabetic')

else:
  print('The person is diabetic')

