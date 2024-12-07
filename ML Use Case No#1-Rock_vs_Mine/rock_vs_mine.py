

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

"""# Data Collection and Data Processing"""

dataset=pd.read_csv('/content/sonar data.csv',header=None)

dataset.shape

dataset.head()

dataset.describe()

dataset.isnull().sum().sum()

dataset[60].value_counts()

"""M --> Mine

R --> Rock
"""

dataset.groupby(60).mean()

"""Seperating Data and Label"""

X=dataset.drop(labels=60,axis=1)
Y=dataset[60]

X.head()

Y.head()

Y.tail()

print(X.shape)
print(Y.shape)

"""# Trainning and Test Data

"""

X_TRAIN,X_TEST,Y_TRAIN,Y_TEST=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

print(X.shape,X_TRAIN.shape,X_TEST.shape)

"""# Model Trainning"""

model=LogisticRegression()

"""Trainnig the logistic regression by trainning data"""

model.fit(X_TRAIN,Y_TRAIN)

model.score(X_TRAIN,Y_TRAIN)

"""Accuracy on test data"""

accuracy_score(model.predict(X_TEST),Y_TEST)

"""# Making Predictive System"""

input_data=(0,0,0.0449,0.1096,0.1913,0,0.0761,0.1092,0.0757,0.1006,0.2500,0.3988,0.3809,0.4753,0.6165,0.6464,0.8024,0.9208,0.9832,0.9634,0.8646,0.8325,0.8276,0.8007,0.6102,0.4853,0.4355,0.4307,0.4399,0.3833,0.3032,0.3035,0.3197,0.2292,0.2131,0.2347,0.3201,0.4455,0.3655,0.2715,0.1747,0.1781,0.2199,0.1056,0.0573,0.0307,0.0237,0.0470,0.0102,0.0057,0.0031,0.0163,0.0099,0.0084,0.0270,0.0277,0.0097,0.0054,0.0148,0.0092)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
if prediction[0]=='R':
  print('The object is Rock')
else:
  print('The object is Mine')

