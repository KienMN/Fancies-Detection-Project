# LVQ Model

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/SD-3X_rocktype.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Training the LVQ
from detection.competitive_learning_network import LvqNetwork
lvq = LvqNetwork(n_subclass = 100, learning_rate = 0.5, decay_rate = 1)
lvq.sample_weights_init(X_train)
lvq.fit(X_train, y_train, num_iteration = 10000, epoch_size = 500)

# Predict the result
y_pred, confidence_score = lvq.predict(X_test, confidence = 1)
y_pred = encoder.inverse_transform(y_pred)

for i in range (len(y_pred)):
  print("y_pred:", y_pred[i], "y_true:", y_test[i], "confidence:", confidence_score[i])

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))