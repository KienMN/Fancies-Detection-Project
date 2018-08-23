# Test unused neurons according to size

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
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X = sc.fit_transform(X)
# X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# print(n_sample)
# print(X)
# print(len(y))

# Training the SSOM
from detection.competitive_learning_network import AdaptiveLVQ
sizes = np.arange(5, 11, 5)
unused_neurons_with_bias = []
unused_neurons_without_bias = []
tested_sizes = []
threshold = 0.05
# print (sizes)

for size in sizes:
  # Model with bias
  print('Testing model with size {} with bias ...'.format(size))
  ssom = AdaptiveLVQ(n_rows = size, n_cols = size,
                    learning_rate = 0.5, decay_rate = 1,
                    sigma = 1, sigma_decay_rate = 1,
                    bias = True, weights_init = 'pca',
                    neighborhood = 'gaussian', label_weight = 'inverse_distance')
  # Init weights
  ssom.pca_weights_init(X)
  init_weights = np.copy(ssom._competitive_layer_weights)
  qe_with_bias = []
  qe_with_bias.append(ssom.quantization_error(X))
  
  # Training process
  for i in range (10):
    np.random.shuffle(X)
    qe_with_bias.append(ssom.train_competitive(X, num_iteration = n_sample, epoch_size = n_sample).quantization_error(X))
  ssom.label_neurons(X, y)
  for i in range (10):
    np.random.shuffle(X)
    qe_with_bias.append(ssom.train_batch(X, y, num_iteration = n_sample, epoch_size = n_sample).quantization_error(X))
  
  # Trained weights
  trained_weights = np.copy(ssom._competitive_layer_weights)
  
  # Determing distance changes
  from detection.competitive_learning_network.utils import euclidean_distance
  distance = np.array([])
  for i in range (size * size):
    distance = np.append(distance, euclidean_distance(init_weights[i], trained_weights[i]))
  
  # Unused neurons
  n_unused_neurons = 0
  max_distance = np.max(distance)
  for i in range (size * size):
    if distance[i] < threshold * max_distance:
      n_unused_neurons += 1
  print('Finished epoch:', ssom._current_epoch)
  print('Unused neurons:', n_unused_neurons)
  unused_neurons_with_bias.append(n_unused_neurons / (size * size))

  # Model without bias
  print('Testing model with size {} without bias ...'.format(size))
  ssom = AdaptiveLVQ(n_rows = size, n_cols = size,
                    learning_rate = 0.5, decay_rate = 1,
                    sigma = 1, sigma_decay_rate = 1,
                    bias = False, weights_init = 'pca',
                    neighborhood = 'gaussian', label_weight = 'inverse_distance')
  # Init weights
  ssom.pca_weights_init(X)
  init_weights = np.copy(ssom._competitive_layer_weights)
  qe_without_bias = []
  qe_without_bias.append(ssom.quantization_error(X))
  
  # Training process
  for i in range (10):
    np.random.shuffle(X)
    qe_without_bias.append(ssom.train_competitive(X, num_iteration = n_sample, epoch_size = n_sample).quantization_error(X))
  ssom.label_neurons(X, y)
  for i in range (10):
    np.random.shuffle(X)
    qe_without_bias.append(ssom.train_batch(X, y, num_iteration = n_sample, epoch_size = n_sample).quantization_error(X))
  
  # Trained weights
  trained_weights = np.copy(ssom._competitive_layer_weights)
  
  # Determing distance changes
  from detection.competitive_learning_network.utils import euclidean_distance
  distance = np.array([])
  for i in range (size * size):
    distance = np.append(distance, euclidean_distance(init_weights[i], trained_weights[i]))
  
  # Unused neurons
  n_unused_neurons = 0
  max_distance = np.max(distance)
  for i in range (size * size):
    if distance[i] < threshold * max_distance:
      n_unused_neurons += 1
  print('Finished epoch:', ssom._current_epoch)
  print('Unused neurons:', n_unused_neurons)
  unused_neurons_without_bias.append(n_unused_neurons / (size * size))

  # Visualization
  iterations = np.arange(0, 20 * n_sample + 1, n_sample)
  tested_sizes.append(size)
  plt.plot(tested_sizes, unused_neurons_with_bias, c = 'r', label = 'With bias')
  plt.plot(tested_sizes, unused_neurons_without_bias, c = 'b', label = 'Without bias')
  plt.legend()
  plt.title('Unused Neurons Proportion (Gaussian neighborhood)')
  # plt.show()
  fig_path = os.path.join(os.path.dirname(__file__), 'images/unused_neurons.png')
  plt.savefig(fig_path)
  
  # Quantization error
  plt.clf()
  plt.plot(iterations, qe_with_bias, c = 'r', label = 'With bias')
  plt.plot(iterations, qe_without_bias, c = 'b', label = 'Without bias')
  plt.legend()
  plt.title('Quantization error of {} * {} model'.format(size, size))
  fig_path = os.path.join(os.path.dirname(__file__), 'images/model_{}*{}.png'.format(size, size))
  plt.savefig(fig_path)
  plt.clf()