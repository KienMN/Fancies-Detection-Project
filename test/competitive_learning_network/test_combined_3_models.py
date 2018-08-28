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

# Statistic
print('Total samples:', len(y))
n_classes = len(np.unique(y))
print('Number of classes:', n_classes)
for i in np.unique(y):
  print('Class {}: {} samples'.format(i, len(np.where(y == i)[0])))


# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X = sc.fit_transform(X)

# Spliting the dataset into the Training set and the Test set
X_train_1 = X[:500, :]
X_train_2 = X[500: 1000, :]
X_train_3 = X[1000: 1500, :]
X_test = X[1500:, :]
y_train_1 = y[:500]
y_train_2 = y[500: 1000]
y_train_3 = y[1000: 1500]
y_test = y[1500:]

# Label encoder
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# y_train_1 = encoder.fit_transform(y_train_1)
# y_train_2 = encoder.transform(y_train_2)
# y_train_3 = encoder.transform(y_train_3)

print(len(X_train_1), len(X_train_2), len(X_train_3), len(X_test))

# Training the SSOM models
from detection.competitive_learning_network.lvq_network import AdaptiveLVQ
ssom1 = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.75, decay_rate = 1,
                  bias = True, weights_init = 'pca', sigma = 2, sigma_decay_rate = 1,
                  neighborhood = 'bubble', label_weight = 'inverse_distance')
n_training_samples = len(X_train_1)
ssom1.pca_weights_init(X_train_1)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom1.train_competitive(X_train_1[s], num_iteration = n_training_samples, epoch_size = n_training_samples)
ssom1.label_neurons(X_train_1, y_train_1)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom1.train_batch(X_train_1[s], y_train_1[s], num_iteration = n_training_samples, epoch_size = n_training_samples)

ssom2 = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.75, decay_rate = 1,
                  bias = True, weights_init = 'pca', sigma = 2, sigma_decay_rate = 1,
                  neighborhood = 'bubble', label_weight = 'inverse_distance')
n_training_samples = len(X_train_2)
ssom2.pca_weights_init(X_train_2)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom2.train_competitive(X_train_2[s], num_iteration = n_training_samples, epoch_size = n_training_samples)
ssom2.label_neurons(X_train_2, y_train_2)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom2.train_batch(X_train_2[s], y_train_2[s], num_iteration = n_training_samples, epoch_size = n_training_samples)

ssom3 = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.75, decay_rate = 1,
                  bias = True, weights_init = 'pca', sigma = 2, sigma_decay_rate = 1,
                  neighborhood = 'bubble', label_weight = 'inverse_distance')
n_training_samples = len(X_train_3)
ssom3.pca_weights_init(X_train_3)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom3.train_competitive(X_train_3[s], num_iteration = n_training_samples, epoch_size = n_training_samples)
ssom3.label_neurons(X_train_3, y_train_3)
for i in range (5):
  s = np.arange(n_training_samples)
  np.random.shuffle(s)
  ssom3.train_batch(X_train_3[s], y_train_3[s], num_iteration = n_training_samples, epoch_size = n_training_samples)

# Combining 3 models
from detection.competitive_learning_network.combined_som import CombinedSom
combined_som = CombinedSom([ssom1, ssom2, ssom3])

# Predicting the result

# Winner takes all
y_pred_1 = combined_som.winner_takes_all(X_test, crit = 'confidence_score')

# Max sum of confidence score
y_pred_2 = combined_som.combined_with_confidence_score(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred_1)
print(cm1)

true_results = 0
for i in range (len(cm1)):
  true_results += cm1[i][i]
print('Accuracy:', true_results / np.sum(cm1))

cm2 = confusion_matrix(y_test, y_pred_2)
print(cm2)

true_results = 0
for i in range (len(cm2)):
  true_results += cm2[i][i]
print('Accuracy:', true_results / np.sum(cm2))