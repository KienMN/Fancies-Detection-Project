# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
# filepath = os.path.join(os.path.dirname(__file__), 'data/SD-3X_rocktype.csv')
filepath = os.path.join(os.path.dirname(__file__), 'data/processed_15_1-SD-1X_LQC.csv')

dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
classifier = RandomMaps(n_estimators = 1, size = 5,
                        learning_rate = 0.5, decay_rate = 1,
                        sigma = 2, sigma_decay_rate = 1,
                        label_weight = 'inverse_distance')
classifier.fit(X_train, y_train, max_first_iters = 100, first_epoch_size = 400, max_second_iters = 4000, second_epoch_size = 400,
              features_arr = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], max_maps_each_features = 1)

# Predict the result
y_pred = classifier.predict(X_test, crit='confidence_score')
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