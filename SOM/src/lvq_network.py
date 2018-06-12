from math import sqrt
import numpy as np
from numpy import array, argmax, zeros, random, append, dot

def fast_norm(x):
  """
  Returns norm-2 of a 1-D numpy array.
  """
  # x = np.array(x)
  return sqrt(dot(x, x.T))

def compet(x):
  idx = np.argmax(x)
  res = np.zeros((len(x)))
  res[idx] = 1
  return(res)

def euclidean_distance(a, b):
  a = array(a)
  b = array(b)
  x = a - b
  return(fast_norm(x))

class LvqNetwork(object):
  def __init__(self, n_feature, n_subclass, n_class, learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1):
    self._n_feature = n_feature
    self._n_subclass = n_subclass
    self._n_class = n_class
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate
    if learning_rate_decay_function:
      self._learning_rate_decay_function = learning_rate_decay_function
    else:
      self._learning_rate_decay_function = lambda learning_rate, iteration, decay_rate: learning_rate / (1 + decay_rate * iteration)
    
    # initializing competitive layer weights
    self._competitive_layer_weights = random.RandomState().rand(n_subclass, n_feature)
    # normalizing competitive layer weights
    for i in range (n_subclass):
      norm = fast_norm(self._competitive_layer_weights[i])
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm
    
    # initializing linear layer weights
    self._linear_layer_weights = zeros((n_class, n_subclass))
    n_subclass_per_class = n_subclass // n_class
    for i in range (n_class):
      if i != n_class - 1:
        for j in range (i * n_subclass_per_class, (i + 1) * n_subclass_per_class):
          self._linear_layer_weights[i][j] = 1
      else:
        for j in range (i * n_subclass_per_class, n_subclass):
          self._linear_layer_weights[i][j] = 1

  def winner(self, x):
    n = array([])
    for i in range(self._n_subclass):
      n = append(n, euclidean_distance(x, self._competitive_layer_weights[i]))
    n = (-1) * n
    return compet(n)

  def classify(self, win):
    n = dot(self._linear_layer_weights, win.T)
    return argmax(n)

  def update(self, x, y, iteration):
    win = self.winner(x)
    win_idx = argmax(win)
    y_hat = self.classify(win)
    alpha = self._learning_rate_decay_function(self._learning_rate, iteration, self._decay_rate)
    if y_hat == y:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] + alpha * (x - self._competitive_layer_weights[win_idx])
    else:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] - alpha * (x - self._competitive_layer_weights[win_idx])
    # normalizing
    norm = fast_norm(self._competitive_layer_weights[win_idx])
    self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] / norm

  def train_batch(self, X, y, num_iteration, epoch_size):
    iteration = 0
    while iteration < num_iteration:
      idx = iteration % len(X)
      epoch = iteration // epoch_size
      self.update(X[idx], y[idx], epoch)
      iteration += 1

  def predict(self, X):
    y_pred = np.array([])
    n_sample = len(X)
    for i in range (n_sample):
      win = self.winner(X[i])
      y_pred = append(y_pred, self.classify(win))
    return y_pred

  def details(self):
    print(self._competitive_layer_weights)
    print(self._linear_layer_weights)

# x = np.array([0.1, 0.2, 0.3])
# y = 1
# lvq = LvqNetwork(3, 3, 3)
# lvq.update(x, y, 0)
# lvq.details()
# win = lvq.winner(x)
# y = lvq.classify(win)
# print(win)
# print(y)