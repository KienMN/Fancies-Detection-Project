from math import exp
import numpy as np
from numpy import array, argmax, zeros, random, append, dot, copy, amax, amin, ones, argwhere, argmin, argsort, unique, sum
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from .utils import fast_norm, compet, euclidean_distance, default_bias_function
from .utils import default_learning_rate_decay_function, default_radius_decay_function, default_non_bias_function
# from .utils import limit_range

feature_range = (-1, 1)
visual_feature_range = (0, 1)

# Ignoring deprecation warning
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

class TheoreticalLvq(object):
  """Theoretical Learning Vector Quantization.

  Parameters
  ----------

  n_subclass : int
    Number of subclasses in the competitive layer.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias : bool, default: 1
    Conscience which prevents a neuron from winning too many times.

  bias_function : function, default: None
    Function that updates biases value of neurons in the competitive layer.

  weights_normalization : options ['length'], default: None
    Normalizing weights of neuron.

  weights_init : option ['sample', 'random'], default: 'sample'
    Weights initialization strategy, if weights_init is not given or wrongly given, weights will be initializaed randomly.

  verbose : int, default: 1
    Setting verbose to any positive number for verbosity
  """

  def __init__(self, n_subclass,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias = False, bias_function = None, weights_normalization = None, weights_init = 'sample',
              verbose = 1):

    # Basic parameters
    self._n_subclass = n_subclass
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate

    # Epoch
    self._current_epoch = 0
    
    # Quantization error
    self._epochs_set = []
    self._qe = []

    # Information
    self._info = {}

    if learning_rate_decay_function:
      self._learning_rate_decay_function = learning_rate_decay_function
    else:
      self._learning_rate_decay_function = default_learning_rate_decay_function
    
    if bias:
      if bias_function:
        self._bias_function = bias_function
      else:
        self._bias_function = default_bias_function
    else:
      self._bias_function = default_non_bias_function

    # Weights normalization
    if weights_normalization == "length":
      self._weights_normalization = weights_normalization
    else:
      self._weights_normalization = "default"

    # Weights initialization strategy
    # Note: pca is not used for this class, it is useful for constructor of subclass
    if weights_init == 'pca' or weights_init == 'sample':
      self._weights_init = weights_init
    else:
      self._weights_init = 'random'
    
    if verbose > 0:
      print('Model:')
      print('Number of subclasses: {}; Learning rate: {}; Decay rate: {};'.format(n_subclass, learning_rate, decay_rate))
      if learning_rate_decay_function:
        print('Using user-defined learning rate decay function.')
      else:
        print('Using default learning rate decay function.')
      if bias:
        if bias_function:
          print('Using user-defined bias function.')
        else:
          print('Using default bias function.')
      else:
        print('No conscience applied.')
      if weights_init == 'sample' or weights_init == 'random' or weights_init == 'pca':
        print('Using {} weights initialization.'.format(weights_init))
      else:
        print('Warning: Wrong weights initialization. Using random weights initialization instead.')
        
    # Weights
    self._competitive_layer_weights = None
    self._linear_layer_weights = None

    # Initializing biases value corresponding to competitive layer
    self._biases = zeros((n_subclass))

    # Initializing winner neurons counter
    self._winner_count = zeros((n_subclass))

  def fit(self, X, y, num_iteration, epoch_size, quantization_error = False, verbose = 1):
    """Fit the model according to the given training data.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    y : 1D numpy array, shape (n_samples,)
      Target vector relative to X.
    
    num_iteration : int
      Number of iterations.

    epoch_size : int
      Size of chunk of data, after each chunk of data, parameter such as learning rate and sigma will be recalculated.

    quantization_error: boolean, default: False
      Determining to save quantization error or not
    
    Returns
    -------
    self : object
      Returns self.
    """
    if len(X.shape) <= 1:
      raise Exception("Data X is expected to be 2D array. Reshape or try another dataset.")
    self._n_feature = X.shape[1]

    y = y.astype(np.int8)
    self._n_class = len(unique(y))
    
    if self._n_subclass < self._n_class:
      raise Exception("The number of subclasses must be more than or equal to the number of classes.")
    
    # Initializing competitive layer weights
    if self._competitive_layer_weights is None:
      self._competitive_layer_weights = random.RandomState().rand(self._n_subclass, self._n_feature)
    
      if self._weights_init == 'sample':
        self.sample_weights_init(X)
      elif self._weights_init == 'pca':
        self.pca_weights_init(X)

      # Normalizing competitive layer weights
      for i in range (self._n_subclass):
        if self._weights_normalization == "length":
          norm = fast_norm(self._competitive_layer_weights[i])
          self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

    # Initializing linear layer weights
    if self._linear_layer_weights is None:
      self._linear_layer_weights = zeros((self._n_class, self._n_subclass))
      n_subclass_per_class = self._n_subclass // self._n_class
      for i in range (self._n_class):
        if i != self._n_class - 1:
          for j in range (i * n_subclass_per_class, (i + 1) * n_subclass_per_class):
            self._linear_layer_weights[i][j] = 1
        else:
          for j in range (i * n_subclass_per_class, self._n_subclass):
            self._linear_layer_weights[i][j] = 1
    
    print('Training...')
    self.train_batch(X, y, num_iteration, epoch_size, quantization_error)
    print('Trained.')
    return self

  def pca_weights_init(self, *args, **kwargs):
    return self

  def sample_weights_init(self, data):
    """
    Initializes the weights of the competitive layer, picking random samples from data.
    
    Parameters
    ----------
    data : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    self : object
      Returns self.
    """
    if self._competitive_layer_weights is None:
      self._competitive_layer_weights = zeros((self._n_subclass, data.shape[1]))

    for i in range (self._n_subclass):
      # Initializing the weights, picking random sample from data
      rand_idx = random.random_integers(0, len(data) - 1)
      self._competitive_layer_weights[i] = data[rand_idx].copy()
    return self

  def train_batch(self, X, y, num_iteration, epoch_size, quantization_error = False):
    """Looping through input vectors to update weights of neurons.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    y : 1D numpy array, shape (n_samples,)
      Target vector relative to X.
    
    num_iteration : int
      Number of iterations.

    epoch_size : int
      Size of chunk of data, after each chunk of data, parameter such as learning rate and sigma will be recalculated.

    quantization_error: boolean, default: False
      Determining to save quantization error or not

    Returns
    -------
    self : object
      Returns self.
    """
    iteration = 0
    s = np.arange(len(X))
    np.random.shuffle(s)
    while iteration < num_iteration:
      idx = iteration % len(X)
      # Updating neurons' weights, picking sample randomly
      self.update(x = X[s[idx]], y = y[s[idx]], epoch = self._current_epoch)
      iteration += 1
      if iteration % epoch_size == 0:
        # Increasing current epoch
        self._current_epoch += 1
        # Updating quantization error
        self._epochs_set.append(len(self._epochs_set) + 1)
        self._qe.append(self.quantization_error(X))
        np.random.shuffle(s)
    return self

  def update(self, x, y, epoch):
    """
    Updates the weights of competitive layer and the biases.

    Parameters
    ----------
    x : 1D numpy array shape (n_features,)
      Input vector where n_features is the number of features.

    y : int
      Class to which input vector is relative.

    epoch : int
      Sequence number of epoch iterations, after each iterations, learning rate and sigma will be recalculated.

    Returns
    -------
    self : object
      Returns self.
    """
    win = self.winner(x)
    win_idx = argmax(win)
    self._biases = self._bias_function(self._biases, win_idx)
    self._winner_count[win_idx] += 1
    y_hat = self.classify(win)
    alpha = self._learning_rate_decay_function(self._learning_rate, epoch, self._decay_rate)
    beta = alpha / 3
    if y_hat == y:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] + alpha * (x - self._competitive_layer_weights[win_idx])
    else:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] - beta * (x - self._competitive_layer_weights[win_idx])
    
    # Limiting value range
    self._competitive_layer_weights[win_idx, self._competitive_layer_weights[win_idx] > 1] = 1
    self._competitive_layer_weights[win_idx, self._competitive_layer_weights[win_idx] < -1] = -1

    # Normalizing the weights
    if self._weights_normalization == "length":
      norm = fast_norm(self._competitive_layer_weights[win_idx])
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] / norm

    return self

  def winner(self, x):
    """
    Determines the winner neuron in competitive layer.

    Parameters
    ----------
    x : 1D numpy array, shape (n_features,)
      Input vector where n_features is the number of features.

    Returns
    -------
    n : 1D numpy array, shape (n_subclass,)
      Array where element with index of the winner neuron has value 1, others have value 0.
    """
    n = array([])
    for i in range(self._n_subclass):
      n = append(n, (-1) * euclidean_distance(x, self._competitive_layer_weights[i]) + self._biases[i])
    return compet(n)

  def classify(self, win):
    """
    Classifies the winner neuron into one class.
    
    Parameters
    ----------
    win : 1D numpy array, shape (n_subclass,)
      Array which determines the winner neuron.

    Returns
    -------
    class : int
      Class to which winner neuron belongs.
    """
    n = dot(self._linear_layer_weights, win.T)
    return int(argmax(n))

  def predict(self, X, confidence = False):
    """Predicting the class according to input vectors.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    confidence : bool, default: False
      Computes and returns confidence score if confidence is true.

    Returns
    -------
    y_pred : 1D numpy array, shape (n_samples,)
      Prediction target vector relative to X.

    confidence_score : 1D numpy array, shape (n_samples,)
      If confidence is true, returns confidence scores of prediction made.
    """

    y_pred = array([]).astype(np.int8)
    confidence_score = array([])
    k = self._n_subclass // 25
    n_sample = len(X)
    for i in range (n_sample):
      x = X[i]
      win = self.winner(x)
      y_i = int(self.classify(win))
      y_pred = append(y_pred, y_i)

      # Computing confidence score
      if confidence:
        distances = array([])
        classes = array([]).astype(np.int8)
        
        for j in range (self._n_subclass):
          distance = euclidean_distance(x, self._competitive_layer_weights[j]) - self._biases[j]
          class_name = argmax(self._linear_layer_weights[:, j])
          distances = append(distances, distance)
          classes = append(classes, int(class_name))
        
        neighbors = argsort(distances)
        a = 0
        b = 0
        
        for j in range (k):
          if classes[neighbors[j]] == y_i:
            a = a + exp(-(distances[neighbors[j]] ** 2))
          b = b + exp(-(distances[neighbors[j]] ** 2))
        confidence_score = append(confidence_score, a / b)

    if confidence:
      return y_pred, confidence_score
    else:
      return y_pred

  def distance_from_winner_neurons(self, X, prediction = False):
    """Computing the distance between samples and their winner neuron.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    distances : 1D numpy array, shape (n_samples,)
      Distance between the samples and their winner neuron.
    """
    n_sample = len(X)
    y_pred = array([]).astype(np.int8)
    distances = np.array([])
    for i in range (n_sample):
      x = X[i]
      win = self.winner(x)
      win_idx = argmax(win)
      y_i = int(self.classify(win))
      y_pred = append(y_pred, y_i)
      distances = np.append(distances, euclidean_distance(x, self._competitive_layer_weights[win_idx]))
    if prediction:
      return y_pred, distances
    return distances

  def quantization_error(self, X):
    """Determining quantization error of the network.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    quantization error : float
      Quantization error of the network.
    """
    
    n = len(X)
    error = 0
    for i in range (n):
      x = X[i]
      win = self.winner(x)
      win_idx = argmax(win)
      error += euclidean_distance(x, self._competitive_layer_weights[win_idx])
    return error / n

  def details(self):
    """Prints detailed information of lvq"""
    print("Competitive layer weights")
    print(self._competitive_layer_weights)
    print("Linear layer weights")
    print(self._linear_layer_weights)
    print("Biases")
    print(self._biases)
    print("Winner neurons count")
    print(self._winner_count)

  def count_border_neurons(self):
    border_neurons = 0
    for i in range (self._n_subclass):
      if any((self._competitive_layer_weights[i] == 1) | (self._competitive_layer_weights[i] == -1)):
        border_neurons += 1
    return border_neurons

class LvqNetworkWithNeighborhood(TheoreticalLvq):
  """Learning Vector Quantization with Neighborhood concept.

  Parameters
  ----------

  n_rows : int
    Number of rows in the competitive layer.

  n_cols : int
    Number of columns in the competitive layer.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias : bool, default: 1
    Conscience which prevents a neuron win too many times.

  bias_function : function, default: None
    Function that updates biases value of neurons in the competitive layer.

  weights_normalization : option ["length"], default: None
    Normalizing weights of neuron.

  weights_init : option ['sample', 'random', 'pca'], default: random
    Weights initialization strategy, if weights_init is not given or wrongly given, weights will be initializaed randomly.

  sigma : float
    Radius of neighborhood around winner neuron in the competitive layer.

  sigma_decay_function : function, default: None
    Function that decreases sigma after number of iterations.

  sigma_decay_rate : float, default: 1
    Reduction rate of sigma after number of iterations.

  neighborhood : option ["bubble", "gaussian", None], default: None
    Function that determines coefficient for neighbors of winner neuron in competitive layer.

  verbose : int, default: 1
    Setting verbose to any positive number for verbosity
  """
  def __init__(self, n_rows, n_cols,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias = False, bias_function = None, weights_normalization = None, weights_init = 'sample',
              sigma = 0, sigma_decay_function = None, sigma_decay_rate = 1,
              neighborhood = None, verbose = 1):
    super().__init__(n_subclass = n_rows * n_cols,
                    learning_rate = learning_rate, learning_rate_decay_function = learning_rate_decay_function,
                    decay_rate = decay_rate,
                    bias = bias, bias_function = bias_function, weights_normalization = weights_normalization,
                    weights_init = weights_init, verbose = verbose)
    self._n_rows_subclass = n_rows
    self._n_cols_subclass = n_cols
    self._radius = sigma
    self._radius_decay_rate = sigma_decay_rate
    if neighborhood == "bubble" or neighborhood == "gaussian":
      self._neighborhood_function = neighborhood
    else:
      self._neighborhood_function = None
    if sigma_decay_function:
      self._radius_decay_function = sigma_decay_function
    else:
      self._radius_decay_function = default_radius_decay_function

    if verbose > 0: 
      print('Size of network: {} x {}; Sigma: {}; Sigma decay rate: {}.'.format(n_rows, n_cols, sigma, sigma_decay_rate))
      if sigma_decay_function:
        print('Using user-defined sigma decay function.')
      else:
        print('Using default sigma decay function.')
      if neighborhood == "bubble" or neighborhood == "gaussian":
        print('Using {} neighborhood.'.format(neighborhood))
      elif neighborhood is None:
        print('No neighborhood is applied.')
      else:
        print('Can not detect type of neighborhood. No neighborhood is applied.')
  
  def update(self, x, epoch, y = None):
    """Updates the weights of competitive layer and biasees value.
    
    Parameters
    ----------
    x : 1D numpy array shape (n_features,)
      Input vector where n_features is the number of features.

    y : int
      Class to which input vector is relative. If y is not given, weights of competitive layer will be updated unsupervised.

    epoch : int
      Sequence number of epoch iterations, after each iterations, learning rate and sigma will be recalculated.

    Returns
    -------
    self : object
      Returns self.
    """
    win = self.winner(x)
    win_idx = argmax(win)
    self._biases = self._bias_function(self._biases, win_idx)
    self._winner_count[win_idx] += 1
    alpha = self._learning_rate_decay_function(self._learning_rate, epoch, self._decay_rate)
    beta = alpha / 3
    radius = self._radius_decay_function(self._radius, epoch, self._radius_decay_rate)
    correlation = self.neighborhood(win_idx, radius)
    is_class = None
    if y is not None:
      is_class = self.is_class(y)
    else:
      is_class = ones(self._n_subclass)
    for i in range(self._n_subclass):
      if is_class[i] == 1:
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] + is_class[i] * alpha * correlation[i] * (x - self._competitive_layer_weights[i])
      elif is_class[i] == -1:
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] + is_class[i] * beta * correlation[i] * (x - self._competitive_layer_weights[i])
      # Limiting the weights
      # if self._weights_init == 'random':
      #   self._competitive_layer_weights[i] = limit_range(self._competitive_layer_weights[i])
      # else:
      #   self._competitive_layer_weights[i] = limit_range(self._competitive_layer_weights[i], feature_range = feature_range)
      
      # Limiting value range of neurons' weights
      self._competitive_layer_weights[i, self._competitive_layer_weights[i] > 1] = 1
      self._competitive_layer_weights[i, self._competitive_layer_weights[i] < -1] = -1

      # Normalizing the weights
      if self._weights_normalization == "length":
        norm = fast_norm(self._competitive_layer_weights[i])
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

    return self

  def neighborhood(self, win_idx, radius):
    """Computes correlation between each neurons and winner neuron.
    
    Parameters
    ----------
    win_idx : int
      Index of the winner neuron in the competitive layer.

    radius : float
      Radius of neighborhood around the winner neuron.

    Returns
    -------
    correlation : 1D numpy array shape (n_subclass,)
      Correlation coefficient between each neurons with the winner neuron where n_subclass is the number of neurons in the competitive layer.
    """
    correlation = zeros(self._n_subclass)
    win_i = win_idx // self._n_cols_subclass
    win_j = win_idx % self._n_cols_subclass

    if self._neighborhood_function == "gaussian":
      for idx in range (self._n_subclass):
        i = idx // self._n_cols_subclass
        j = idx % self._n_cols_subclass
        distance = (win_i - i) ** 2 + (win_j - j) ** 2
        correlation[idx] = exp(- distance / (2 * (radius ** 2)))
    elif self._neighborhood_function == "bubble":
      for idx in range (self._n_subclass):
        i = idx // self._n_cols_subclass
        j = idx % self._n_cols_subclass
        if (win_i - i) ** 2 + (win_j - j) ** 2 <= radius ** 2:
          correlation[idx] = 1
    else:
      correlation[win_idx] = 1

    return correlation

  def is_class(self, y):
    """Determines whether neurons in competitive layer belong to class y or not.
    
    Parameters
    ----------
    y : int
      Class name, i.e 0, 1,...

    Returns
    -------
    res : 1D numpy array shape (n_subclass)
      Sign of coefficient, (+) if neuron belongs to class y, (-) otherwise.
    """
    res = copy(self._linear_layer_weights[y])
    for i in range (self._n_subclass):
      if res[i] == 0:
        res[i] = -1
    return res

  def pca_weights_init(self, data):
    """Initializes the weights of the competitive layers using Principal Component Analysis technique.
    
    Parameters
    ----------
    data : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    self : object
      Returns self.
    """
    self._competitive_layer_weights = zeros((self._n_subclass, data.shape[1]))
    # Pca parameters
    pca_number_of_components = None
    coord = None

    if self._n_cols_subclass == 1 or self._n_rows_subclass == 1 or data.shape[1] == 1:
      pca_number_of_components = 1
      if self._n_cols_subclass == 1 and self._n_rows_subclass == 1:
        coord = array([[1], [0]])
        # print(coord)
        # print(coord[0][0])
      else:  
        coord = zeros((self._n_subclass, 1))
        for i in range (self._n_subclass):
          coord[i][0] = i
    else:
      pca_number_of_components = 2
      coord = zeros((self._n_subclass, 2))
      for i in range (self._n_subclass):
        coord[i][0] = i // self._n_cols_subclass
        coord[i][1] = i % self._n_cols_subclass
    
    mx = amax(coord, axis = 0)
    mn = amin(coord, axis = 0)
    coord = (coord - mn) / (mx - mn)
    coord = (coord - 0.5) * 2
    pca = PCA(n_components = pca_number_of_components)
    pca.fit(data)
    eigvec = pca.components_
    # print(eigvec)
    # print(coord)
    for i in range (self._n_subclass):
      for j in range (eigvec.shape[0]):
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] + coord[i][j] * eigvec[j]
      if fast_norm(self._competitive_layer_weights[i]) == 0:
        self._competitive_layer_weights[i] = 0.01 * eigvec[0]
      # Normalizing the weights
      if self._weights_normalization == "length":
        norm = fast_norm(self._competitive_layer_weights[i])
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm
    return self

class AdaptiveLVQ(LvqNetworkWithNeighborhood):
  """Learning Vector Quantization with flexible competitive layer.

  Parameters
  ----------
  
  n_feature : int
    Number of features of the dataset.

  n_rows : int
    Number of rows in the competitive layer.

  n_cols : int
    Number of columns in the competitive layer.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias : bool, default: 1
    Conscience which prevents a neuron win too many times.

  bias_function : function, default: None
    Function that updates biases value of neurons in the competitive layer.

  weights_normalization : option ['length'], default: None
    Normalizing weights of neuron.

  weights_init : option ['sample', 'random', 'pca'], default: random
    Weights initialization strategy, if weights_init is not given or wrongly given, weights will be initializaed randomly.

  sigma : float
    Radius of neighborhood around winner neuron in the competitive layer.

  sigma_decay_function : function, default: None
    Function that decreases sigma after number of iterations.

  sigma_decay_rate : float, default: 1
    Reduction rate of sigma after number of iterations.

  neighborhood : option ['bubble', 'gaussian'], default: None
    Function that determines coefficient for neighbors of winner neuron in competitive layer.

  label_weight : option ['uniform', 'exponential_distance', 'inverse_distance'], default: None
    Strategy to label class name for neurons in the competitive layer

  verbose : int, default: 1
    Setting verbose to any positive number for verbosity
  """
  def __init__(self, n_rows, n_cols,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias = False, bias_function = None, weights_normalization = None, weights_init = None,
              sigma = 0, sigma_decay_function = None, sigma_decay_rate = 1,
              neighborhood = None, label_weight = None, verbose = 1):
    super().__init__(n_rows = n_rows, n_cols = n_cols,
                    learning_rate = learning_rate, learning_rate_decay_function = learning_rate_decay_function,
                    decay_rate = decay_rate,
                    bias = bias, bias_function = bias_function,
                    weights_normalization = weights_normalization, weights_init = weights_init,
                    sigma = sigma, sigma_decay_function = sigma_decay_function, sigma_decay_rate = sigma_decay_rate,
                    neighborhood = neighborhood, verbose = verbose)
    self._label_weight = label_weight

  def train_competitive(self, X, num_iteration, epoch_size, quantization_error = False):
    """Fitting the weights of the neurons in the competitive layer before labeling class for each neurons.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    y : 1D numpy array, shape (n_samples,)
      Target vector relative to X.
    
    num_iteration : int
      Number of iterations.

    epoch_size : int
      Size of chunk of data, after each chunk of data, parameter such as learning rate and sigma will be recalculated.

    quantization_error: boolean, default: False
      Determining to save quantization error or not

    Returns
    -------
    self : object
      Returns self.
    """
    iteration = 0
    s = np.arange(len(X))
    np.random.shuffle(s)
    while iteration < num_iteration:
      idx = iteration % len(X)
      # epoch = iteration // epoch_size
      # self.update(x = X[idx], epoch = epoch)
      self.update(x = X[s[idx]], epoch = self._current_epoch)
      iteration += 1
      if iteration % epoch_size == 0:
        # Increasing current epoch
        self._current_epoch += 1
        # Updating quantization error
        self._epochs_set.append(len(self._epochs_set) + 1)
        self._qe.append(self.quantization_error(X))
    return self

  def label_neurons(self, X, y):
    """Labeling class and computing confidence for each neurons in the competitve layer according to input data.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    y : 1D numpy array, shape (n_samples,)
      Target vector relative to X.

    Returns
    -------
    self : object
      Returns self.
    """
    self._n_class = len(unique(y))
    self._n_neurons_each_classes = zeros(self._n_class)
    self._neurons_confidence = zeros((self._n_subclass, self._n_class))
    
    # Initializing linear layer weights
    if self._linear_layer_weights is None:
      self._linear_layer_weights = zeros((self._n_class, self._n_subclass))

    if self._label_weight == 'exponential_distance':
      neurons_weight = zeros((self._n_subclass, self._n_class))
      m = len(X)
      k = 20
      for i in range (self._n_subclass):
        n = self._competitive_layer_weights[i]
        
        distances = array([])
        for j in range (m):
          distance = euclidean_distance(n, X[j]) - self._biases[i]
          distances = append(distances, distance)
        neighbors = argsort(distances)
        # Distances are sometime too large
        # print(distances[neighbors[: k]])
        
        for j in range (k):
          neurons_weight[i][y[neighbors[j]]] += exp(-(distances[neighbors[j]] ** 2))
        
        # print(y[neighbors[: k]])
        # print(neurons_weight[i])
        self._neurons_confidence[i] = neurons_weight[i] / sum(neurons_weight[i])
        neuron_class_win = argwhere(self._neurons_confidence[i] == amax(self._neurons_confidence[i])).ravel()
        class_name = neuron_class_win[argmin(self._n_neurons_each_classes[neuron_class_win])]
        # print(i, n, class_name)
        self._n_neurons_each_classes[class_name] += 1
        self._linear_layer_weights[class_name][i] = 1
    
    elif self._label_weight == 'inverse_distance':
      neurons_weight = zeros((self._n_subclass, self._n_class))
      m = len(X)
      k = m // 20
      for i in range (self._n_subclass):
        n = self._competitive_layer_weights[i]
        distances = array([])
        for j in range (m):
          distance = euclidean_distance(n, X[j]) - self._biases[i]
          distances = append(distances, distance)
        neighbors = argsort(distances)
        for j in range (k):
          neurons_weight[i][y[neighbors[j]]] += 1 / distances[neighbors[j]]
        
        self._neurons_confidence[i] = neurons_weight[i] / sum(neurons_weight[i])
        # print(self._neurons_confidence[i])
        neuron_class_win = argwhere(self._neurons_confidence[i] == amax(self._neurons_confidence[i])).ravel()
        # print(neuron_class_win)
        class_name = neuron_class_win[argmin(self._n_neurons_each_classes[neuron_class_win])]
        self._n_neurons_each_classes[class_name] += 1
        self._linear_layer_weights[:, i] = 0
        self._linear_layer_weights[class_name][i] = 1
    
    elif self._label_weight == 'max_voting':
      neurons_weight = zeros((self._n_subclass, self._n_class))
      m = len(X)
      k = m // 20
      for i in range (self._n_subclass):
        n = self._competitive_layer_weights[i]
        distances = array([])
        for j in range (m):
          distance = euclidean_distance(n, X[j]) - self._biases[i]
          distances = append(distances, distance)
        neighbors = argsort(distances)
        for j in range (k):
          neurons_weight[i][y[neighbors[j]]] += 1
        
        self._neurons_confidence[i] = neurons_weight[i] / sum(neurons_weight[i])
        neuron_class_win = argwhere(self._neurons_confidence[i] == amax(self._neurons_confidence[i])).ravel()
        class_name = neuron_class_win[argmin(self._n_neurons_each_classes[neuron_class_win])]
        self._n_neurons_each_classes[class_name] += 1
        self._linear_layer_weights[:, i] = 0
        self._linear_layer_weights[class_name][i] = 1

    elif self._label_weight == 'uniform':
      class_win = zeros((self._n_subclass, self._n_class))
      m = len(X)
      for idx in range (m):
        win = self.winner(X[idx])
        win_idx = argmax(win)
        class_win[win_idx][y[idx]] += 1
      for idx in range (self._n_subclass):
        neuron_class_win = argwhere(class_win[idx] == amax(class_win[idx])).ravel()
        class_name = neuron_class_win[argmin(self._n_neurons_each_classes[neuron_class_win])]
        self._n_neurons_each_classes[class_name] += 1
        self._linear_layer_weights[class_name][idx] = 1
        if sum(class_win[idx]) == 0:
          self._neurons_confidence[idx] = [1 / self._n_class] * self._n_class
        else:
          self._neurons_confidence[idx] = class_win[idx] / sum(class_win[idx])
    
    elif self._label_weight == 'inverse_distance_to_classes':
      n_sample = len(X)
      for idx in range (self._n_subclass):
        distances = array([])
        classes = array([]).astype(np.int8)
        
        for j in range (n_sample):
          x = X[j]
          distance = euclidean_distance(x, self._competitive_layer_weights[idx]) - self._biases[idx]
          class_name = y[j]
          distances = append(distances, distance)
          classes = append(classes, int(class_name))
        
        neighbors = argsort(distances)
        j = 0
        distance_to_classes = np.zeros(self._n_class)
        
        while np.prod(distance_to_classes) == 0 and j < n_sample:
          if distance_to_classes[classes[neighbors[j]]] == 0:
            distance_to_classes[classes[neighbors[j]]] = distances[neighbors[j]]
          j += 1
        
        total_sum = 0
        for k in range (self._n_class):
          if distance_to_classes[k] != 0:
            total_sum += float(1 / distance_to_classes[k])
        
        for k in range (self._n_class):
          self._neurons_confidence[idx, k] = (1 / distance_to_classes[k]) / total_sum

        l = np.argmax(self._neurons_confidence[idx])
        self._linear_layer_weights[l, idx] = 1

    else:
      n_subclass_per_class = self._n_subclass // self._n_class
      for i in range (self._n_class):
        if i != self._n_class - 1:
          for j in range (i * n_subclass_per_class, (i + 1) * n_subclass_per_class):
            self._linear_layer_weights[i][j] = 1
            self._n_neurons_each_classes[i] += 1
            self._neurons_confidence[j] = [1 / self._n_class] * self._n_class
        else:
          for j in range (i * n_subclass_per_class, self._n_subclass):
            self._linear_layer_weights[i][j] = 1
            self._n_neurons_each_classes[i] += 1
            self._neurons_confidence[j] = [1 / self._n_class] * self._n_class

    return self

  def fit(self, X, y, first_num_iteration, first_epoch_size, second_num_iteration, second_epoch_size, quantization_error = False):
    """Training the network using vectors in data sequentially.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.
    
    y : 1D numpy array, shape (n_samples,)
      Target vector relative to X.

    first_num_iteration : int
      Number of iterations of the first phase of training.

    first_epoch_size : int
      Size of chunk of data for the first phase of training.

    second_num_iteration : int
      Number of iterations of the second phase of training.

    second_epoch_size : int
      Size of chunk of data for the second phase of training.

    quantization_error: boolean, default: False
      Determining to save quantization error or not

    Returns
    -------
    self : object
      Returns self.
    """

    if len(X.shape) <= 1:
      raise Exception("Data is expected to be 2D array")
    self._n_feature = X.shape[1]
    y = y.astype(np.int8)
    self._n_class = len(unique(y))
    
    if self._n_subclass < self._n_class:
      raise Exception("The number of subclasses must be more than or equal to the number of classes")
    
    # Initializing competitive layer weights
    if self._competitive_layer_weights is None:
      self._competitive_layer_weights = random.RandomState().rand(self._n_subclass, self._n_feature)
      
      if self._weights_init == 'sample':
        self.sample_weights_init(X)
      elif self._weights_init == 'pca':
        self.pca_weights_init(X)
      
      # Normalizing competitive layer weights
      if self._weights_normalization == "length":
        for i in range (self._n_subclass):
          norm = fast_norm(self._competitive_layer_weights[i])
          self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

    # Initializing linear layer weights
    if self._linear_layer_weights is None:
      self._linear_layer_weights = zeros((self._n_class, self._n_subclass))
    
    # Phase 1: Training using SOM concept
    self.train_competitive(X, first_num_iteration, first_epoch_size, quantization_error)
    self.label_neurons(X, y)
    # Phase 2: Training using LVQ concept
    self.train_batch(X, y, second_num_iteration, second_epoch_size, quantization_error)
    self.label_neurons(X, y)

    return self

  def recompute_neurons_confidence(self, X, y):
    n_sample = len(X)
    for idx in range (self._n_subclass):
      distances = array([])
      classes = array([]).astype(np.int8)
      
      for j in range (n_sample):
        x = X[j]
        distance = euclidean_distance(x, self._competitive_layer_weights[idx]) - self._biases[idx]
        class_name = y[j]
        distances = append(distances, distance)
        classes = append(classes, int(class_name))
      
      neighbors = argsort(distances)
      j = 0
      distance_to_classes = np.zeros(self._n_class)
      
      while np.prod(distance_to_classes) == 0 and j < n_sample:
        if distance_to_classes[classes[neighbors[j]]] == 0:
          distance_to_classes[classes[neighbors[j]]] = distances[neighbors[j]]
        j += 1
      
      total_sum = 0
      for k in range (self._n_class):
        if distance_to_classes[k] != 0:
          total_sum += float(1 / distance_to_classes[k])
      
      for k in range (self._n_class):
        self._neurons_confidence[idx, k] = (1 / distance_to_classes[k]) / total_sum

      l = np.argmax(self._neurons_confidence[idx])
      self._linear_layer_weights[:, idx] = 0
      self._linear_layer_weights[l, idx] = 1
    
    return self

  # def predict(self, X, confidence = False):
  #   """Predicting the class according to input vectors.

  #   Parameters
  #   ----------

  #   X : 2D numpy array, shape (n_samples, n_features)
  #     Data vectors, where n_samples is the number of samples and n_features is the number of features.

  #   confidence : bool, default: False
  #     Computes and returns confidence score if confidence is true.

  #   Returns
  #   -------
  #   y_pred : 1D numpy array, shape (n_samples,)
  #     Prediction target vector relative to X.

  #   confidence_score : 1D numpy array, shape (n_samples,)
  #     If confidence is true, returns confidence scores of prediction made.
  #   """
  #   y_pred = array([]).astype(np.int8)
  #   confidence_score = array([])
  #   n_sample = len(X)
  #   for i in range (n_sample):
  #     x = X[i]
  #     win = self.winner(x)
  #     win_idx = argmax(win)
  #     y_i = int(self.classify(win))
  #     y_pred = append(y_pred, y_i)

  #     # Computing confidence score
  #     if confidence:
  #       confidence_score = append(confidence_score, self._neurons_confidence[win_idx, y_i])
    
  #   if confidence:
  #     return y_pred, confidence_score
  #   else:
  #     return y_pred

  def predict(self, X, confidence = False, distance = False, crit = 'distance'):
    """Predicting the class according to input vectors.

    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Data vectors, where n_samples is the number of samples and n_features is the number of features.

    confidence : bool, default: False
      Computes and returns confidence score if confidence is true.

    crit : option ['distance', 'winner_neuron'], default: 'distance'
      Criterion to compute confidence score.

    Returns
    -------
    y_pred : 1D numpy array, shape (n_samples,)
      Prediction target vector relative to X.

    confidence_score : 1D numpy array, shape (n_samples,)
      If confidence is true, returns confidence scores of prediction made.
    """

    # If confidence score is included
    if confidence and crit == 'distance':
      return super().predict(X, confidence = True)
    elif confidence and crit == 'winner_neuron':
      n_sample = len(X)
      y_pred = array([]).astype(np.int8)
      # confidence_score = np.zeros((n_sample, self._n_class))
      confidence_score = array([])
      
      for i in range (n_sample):
        x = X[i]
        win = self.winner(x)
        win_idx = argmax(win)
        y_i = int(self.classify(win))
        y_pred = append(y_pred, y_i)
    
        # Computing confidence score
        # confidence_score[i] = self._neurons_confidence[win_idx].copy()

        confidence_score = np.append(confidence_score, self._neurons_confidence[win_idx, y_i])
      return y_pred, confidence_score

    # If confidence score is not included
    return super().predict(X)