from math import sqrt
import numpy as np
from numpy import array, argmax, zeros, random, append, dot

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

def default_bias_function(biasses, win_idx):
  """
  Default function that reduces bias value of winner neuron and increases bias value of other neurons
  """
  biasses = biasses * 0.9
  biasses[win_idx] = biasses[win_idx] / 0.9 - 0.2
  return biasses

class LvqNetwork(object):
  def __init__(self, n_feature, n_subclass, n_class,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias_function = None):
    """
    Initializing a Learning Vector Quantization

    Parameters:
    
    n_feature: int
      Number of features of the dataset

    n_subclass: int
      Number of subclasses in the competitive layer

    n_class: int
      Number of classes of the dataset

    learning_rate: float, default = 0.5
      Learning rate of the algorithm

    learning_rate_decay_function: function, default = None
      Function that decreases learning rate after number of iterations
    
    decay_rate: float, default = 1
      Reduction rate of learning rate after number of iterations
    
    bias_function: function, default = None
      Function that updates biases value of neurons in competitive layer
    """

    self._n_feature = n_feature
    self._n_subclass = n_subclass
    self._n_class = n_class
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate

    if learning_rate_decay_function:
      self._learning_rate_decay_function = learning_rate_decay_function
    else:
      self._learning_rate_decay_function = lambda learning_rate, iteration, decay_rate: learning_rate / (1 + decay_rate * iteration)
    
    if bias_function:
      self._bias_function = bias_function
    else:
      self._bias_function = default_bias_function

    # Initializing competitive layer weights
    self._competitive_layer_weights = random.RandomState().rand(n_subclass, n_feature)
    # Normalizing competitive layer weights
    for i in range (n_subclass):
      norm = fast_norm(self._competitive_layer_weights[i])
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

    # Initializing biases value corresponding to competitive layer
    self._biases = zeros((n_subclass))

    # Initializing winner neurons counter
    self._winner_count = zeros((n_subclass))

    # Initializing linear layer weights
    self._linear_layer_weights = zeros((n_class, n_subclass))
    n_subclass_per_class = n_subclass // n_class
    for i in range (n_class):
      if i != n_class - 1:
        for j in range (i * n_subclass_per_class, (i + 1) * n_subclass_per_class):
          self._linear_layer_weights[i][j] = 1
      else:
        for j in range (i * n_subclass_per_class, n_subclass):
          self._linear_layer_weights[i][j] = 1

  def random_weights_init(self, data):
    """Initializes the weights of the competitive layer, picking random samples from data"""
    for i in range (self._n_subclass):
      # Initializing the weights, picking random sample from data
      rand_idx = random.random_integers(0, len(data) - 1)
      self._competitive_layer_weights[i] = data[rand_idx]
      # Normalizing the weights
      norm = fast_norm(self._competitive_layer_weights[i])
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

  def winner(self, x):
    """Determines the winner neuron in competitive layer"""
    n = array([])
    for i in range(self._n_subclass):
      n = append(n, (-1) * euclidean_distance(x, self._competitive_layer_weights[i]) + self._biases[i])
    return compet(n)

  def classify(self, win):
    """Classifies the winner neuron into one class"""
    n = dot(self._linear_layer_weights, win.T)
    return argmax(n)

  def update(self, x, y, epoch):
    """Updates the weights of competitive layer and the biases"""
    win = self.winner(x)
    win_idx = argmax(win)
    self._biases = self._bias_function(self._biases, win_idx)
    self._winner_count[win_idx] += 1
    y_hat = self.classify(win)
    alpha = self._learning_rate_decay_function(self._learning_rate, epoch, self._decay_rate)
    if y_hat == y:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] + alpha * (x - self._competitive_layer_weights[win_idx])
    else:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] - alpha * (x - self._competitive_layer_weights[win_idx])
    # Normalizing the weights
    norm = fast_norm(self._competitive_layer_weights[win_idx])
    self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] / norm

  def train_batch(self, X, y, num_iteration, epoch_size):
    """Trains using all the vectors in data sequentially"""
    iteration = 0
    while iteration < num_iteration:
      idx = iteration % len(X)
      epoch = iteration // epoch_size
      self.update(X[idx], y[idx], epoch)
      iteration += 1

  def predict(self, X):
    """Classifies new data"""
    y_pred = np.array([])
    n_sample = len(X)
    for i in range (n_sample):
      win = self.winner(X[i])
      y_pred = append(y_pred, self.classify(win))
    return y_pred

  def details(self):
    print(self._competitive_layer_weights)
    print(self._linear_layer_weights)
    print(self._biases)
    print(self._winner_count)

class LvqNetworkWithNeighborhood(LvqNetwork):
  def __init__(self, n_feature, n_rows, n_cols, n_class,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              radius = 0):
    super().__init__(n_feature = n_feature, n_subclass = n_rows * n_cols, n_class = n_class,
                    learning_rate = learning_rate, decay_rate = decay_rate)
    self._n_rows_subclass = n_rows
    self._n_cols_subclass = n_cols
    self._radius = radius
  
  def neighborhood(self, win_idx, radius):
    """Computes correlation between each neurons and winner neuron"""
    correlation = zeros(self._n_subclass)
    win_i = win_idx // self._n_cols_subclass
    win_j = win_idx % self._n_cols_subclass
    for idx in range (self._n_subclass):
      i = idx // self._n_cols_subclass
      j = idx % self._n_cols_subclass
      if (win_i - i) ** 2 + (win_j - j) ** 2 <= radius ** 2:
        correlation[idx] = 1
    return correlation

  def is_class(self, y):
    """Determines whether neurons in competitive layer belong to class y or not"""
    res = self._linear_layer_weights[y]
    for i in range (self._n_subclass):
      if res[i] == 0:
        res[i] = -1
    return res

  def update(self, x, y, epoch):
    """Updates the weights of competitive layer"""
    win = self.winner(x)
    win_idx = argmax(win)
    alpha = self._learning_rate_decay_function(self._learning_rate, epoch, self._decay_rate)
    radius = self._learning_rate_decay_function(self._radius, epoch, self._decay_rate)
    correlation = self.neighborhood(win_idx, radius)
    is_class = self.is_class(y)
    for i in range(self._n_subclass):
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] + correlation[i] * is_class[i] * alpha * (x - self._competitive_layer_weights[i])
      norm = fast_norm(self._competitive_layer_weights[i])
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm