from math import sqrt
from numpy import dot, argmax, zeros
import numpy as np
from random import sample
import random

def fast_norm(x):
  """
  Returns norm-2 of a 1-D numpy array.
  """
  return sqrt(dot(x, x.T))

def compet(x):
  """
  Returns a 1-D numpy array with the same size as x,
  where the element having the index of the maximum value in x has value 1, others have value 0.
  """
  idx = argmax(x)
  res = zeros((len(x)))
  res[idx] = 1
  return(res)

def euclidean_distance(a, b):
  """
  Returns Euclidean distance between 2 vectors.
  """
  x = a - b
  return(fast_norm(x))

def default_bias_function(biases, win_idx):
  """
  Default function that reduces bias value of winner neuron and increases bias value of other neurons
  """
  biases = biases * 0.9
  biases[win_idx] = biases[win_idx] / 0.9 - 0.2
  return biases

def default_non_bias_function(biases, win_idx):
  return biases

def default_learning_rate_decay_function(learning_rate, iteration, decay_rate):
  return learning_rate / (1 + decay_rate * iteration)

def default_radius_decay_function(sigma, iteration, decay_rate):
  return sigma / (1 + decay_rate * iteration)

def split_data(data, val_size=0.2, proportional=True):

  if val_size == 0:
    return data, data[:0]

  if proportional:
    y = data[:,-1]
    sort_idx = np.argsort(y)
    val, start_idx= np.unique(y[sort_idx], return_index=True)
    indices = np.split(sort_idx, start_idx[1:])
    val_indices = []

    for idx_list in indices:
      n = len(idx_list)
      val_indices += idx_list[(sample(range(n), round(n*val_size)))].tolist()

    val_indices = sorted(val_indices)
  else:
    n = len(data)
    val_indices = sorted(sample(range(n), round(n*val_size)))

  return np.delete(data, val_indices, axis=0), data[val_indices]

def limit_range(x, feature_range = (0, 1)):
  x[np.where(x > feature_range[1])] = feature_range[1]
  x[np.where(x < feature_range[0])] = feature_range[0]
  return x

def weighted_sampling(weights, sample_size):
  """
  Returns a weighted sample with replacement.
  """
  totals = np.cumsum(weights)
  sample = []
  for i in range(sample_size):
    rnd = random.random() * totals[-1]
    idx = np.searchsorted(totals, rnd, 'right')
    sample.append(idx)
  return sample