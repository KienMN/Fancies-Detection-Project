# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Training dataset
filepath1 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-1X.csv')
filepath2 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv')
filepath3 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-3X.csv')
dataset = pd.read_csv(filepath1)
dataset = dataset.append(pd.read_csv(filepath2), ignore_index = True)
dataset = dataset.append(pd.read_csv(filepath3), ignore_index = True)

X_train = dataset.iloc[:, 1: -1].values
y_train = dataset.iloc[:, -1].values.astype(np.int8)

# Importing the Test dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-4X.csv')
test_dataset = pd.read_csv(filepath)
X_test = test_dataset.iloc[:, 1: -1].values
y_test = test_dataset.iloc[:, -1].values.astype(np.int8)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Training the LVQ
from detection.competitive_learning_network.combination import RandomMaps
classifier = RandomMaps(n_estimators = 50, size = 5,
                        learning_rate = 0.75, decay_rate = 1,
                        sigma = 3, sigma_decay_rate = 1,
                        label_weight = 'inverse_distance')
classifier.fit(X_train, y_train, max_first_iters = 20000, first_epoch_size = 4000, max_second_iters = 20000, second_epoch_size = 4000)

# Predict the result
y_pred = classifier.predict(X_test, crit = 'max_voting')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))