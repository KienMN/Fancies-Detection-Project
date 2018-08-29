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
n_sample = 0
train_filepath = os.path.join(os.path.dirname(__file__), 'data/HT_Sensor_train_dataset.csv')
test_filepath = os.path.join(os.path.dirname(__file__), 'data/HT_Sensor_test_dataset.csv')

train_dataset = pd.read_csv(train_filepath)
test_dataset = pd.read_csv(test_filepath)

X_train = train_dataset.iloc[:, 2: -1].values
y_train = train_dataset.iloc[:, -1].values.astype(np.int8)
X_test = test_dataset.iloc[:, 2: -1].values
y_test = test_dataset.iloc[:, -1].values.astype(np.int8)

# Statistic
print('Total train samples:', len(y_train))
n_classes = len(np.unique(y_train))
print('Number of classes:', n_classes)
for i in np.unique(y_train):
  print('Class {}: {} samples'.format(i, len(np.where(y_train == i)[0])))

print('Total test samples:', len(y_test))
n_classes = len(np.unique(y_test))
print('Number of classes:', n_classes)
for i in np.unique(y_test):
  print('Class {}: {} samples'.format(i, len(np.where(y_test == i)[0])))

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the argument
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-m', '--n_models', type = int)
args = parser.parse_args()
n_models = args.n_models
print(n_models)

# Training the SSOM model
if args.n_models == 1:
  from detection.competitive_learning_network.lvq_network import AdaptiveLVQ
  ssom = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.5, decay_rate = 1,
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

  # Making the confusion matrix
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_test, y_pred)
  print(cm)

  true_results = 0
  for i in range (len(cm)):
    true_results += cm[i][i]
  print('Accuracy:', true_results / np.sum(cm))

elif args.n_models == 2:
  total_train_samples = len(X_train)
  train_size = total_train_samples // 2
  X_train_1 = X_train[: train_size, :]
  X_train_2 = X_train[train_size:, :]
  y_train_1 = y_train[: train_size]
  y_train_2 = y_train[train_size:]
  
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

  ssom2 = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.5, decay_rate = 1,
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

  # Combining 2 models
  from detection.competitive_learning_network.combined_som import CombinedSom
  combined_som = CombinedSom([ssom1, ssom2])

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

elif args.n_models == 3:
  total_train_samples = len(X_train)
  train_size = total_train_samples // 3
  X_train_1 = X_train[: train_size, :]
  X_train_2 = X_train[train_size: 2 * train_size, :]
  X_train_3 = X_train[2 * train_size:, :]
  y_train_1 = y_train[: train_size]
  y_train_2 = y_train[train_size: 2 * train_size]
  y_train_3 = y_train[2 * train_size:]

  from detection.competitive_learning_network.lvq_network import AdaptiveLVQ
  ssom1 = AdaptiveLVQ(n_rows = 10, n_cols = 10, learning_rate = 0.5, decay_rate = 1,
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