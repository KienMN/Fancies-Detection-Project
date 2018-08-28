# Test single SSOM model for data of 5 wells

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
from sklearn.preprocessing import MinMaxScaler
filenames = ['processed_15_1-SD-2X-DEV_LQC.csv', 'processed_15_1-SD-3X_LQC.csv', 'processed_15_1-SD-5X-PL_LQC.csv', 'processed_15_1-SD-6X_LQC.csv']
n_sample = 0

filepath = os.path.join(os.path.dirname(__file__), 'data/processed_15_1-SD-1X_LQC.csv')
dataset = pd.read_csv(filepath)
n_sample += len(dataset)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

for filename in filenames:
  filepath = os.path.join(os.path.dirname(__file__), 'data/' + filename)
  dataset = pd.read_csv(filepath)
  n_sample += len(dataset)
  X = np.append(X, dataset.iloc[:, 2: -1].values, axis = 0)
  y = np.append(y, dataset.iloc[:, -1].values.astype(np.int8), axis = 0)

# Spliting the dataset into the Training set and the Test set
X_train = X[:1500, :]
X_test = X[1500:, :]
y_train = y[:1500]
y_test = y[1500:]

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Training the SSOM model
from detection.competitive_learning_network.lvq_network import AdaptiveLVQ
ssom = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.75, decay_rate = 1,
                  bias = True, weights_init = 'pca', sigma = 3, sigma_decay_rate = 1,
                  neighborhood = 'bubble', label_weight = 'inverse_distance')
n_training_samples = len(X_train)
ssom.pca_weights_init(X_train)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom.train_competitive(X_train[s], num_iteration = n_training_samples, epoch_size = n_training_samples)
ssom.label_neurons(X_train, y_train)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom.train_batch(X_train[s], y_train[s], num_iteration = n_training_samples, epoch_size = n_training_samples)

# Predicting the result
y_pred = ssom.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

true_results = 0
for i in range (len(cm)):
  true_results += cm[i][i]
print('Accuracy:', true_results / np.sum(cm))