# Importing the libraries
import numpy as np
import pandas as pd
import os

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/Data_PETREL_INPUT.csv'
dataset = pd.read_csv(filepath)
dataset = dataset.loc[dataset['wellName'] == 'A10']
# print(dataset)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Loading model
from sklearn.externals import joblib
filename = "finalized_model.sav"
loaded_model = joblib.load(filename)

# Predicting the result
y_pred = loaded_model.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / np.sum(cm))