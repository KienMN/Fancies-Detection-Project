from math import exp
import numpy as np
from numpy import array, argmax, zeros, random, append, dot, copy, amax, amin, ones, argwhere, argmin, argsort
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from .utils import fast_norm, compet, euclidean_distance, default_bias_function, default_learning_rate_decay_function, default_radius_decay_function

class LvqNetwork(object):
  """Learning Vector Quantization.

  Parameters
  ----------
  
  n_feature : int
    Number of features of the dataset.

  n_subclass : int
    Number of subclasses in the competitive layer.

  n_class : int
    Number of classes of the dataset.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias_function : function, default: None
    Function that updates biases value of neurons in competitive layer.

  weights_normalization : option ["length"], default: None
    Normalizing weights of neuron.
  """

  def __init__(self, n_feature, n_subclass, n_class,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias_function = None, weights_normalization = None):
    if n_subclass < n_class:
      raise Exception("The number of subclasses must be more than or equal to the number of classes")

    self._n_feature = n_feature
    self._n_subclass = n_subclass
    self._n_class = n_class
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate

    if learning_rate_decay_function:
      self._learning_rate_decay_function = learning_rate_decay_function
    else:
      self._learning_rate_decay_function = default_learning_rate_decay_function
    
    if bias_function:
      self._bias_function = bias_function
    else:
      self._bias_function = default_bias_function

    if weights_normalization == "length":
      self._weights_normalization = weights_normalization
    else:
      self._weights_normalization = "default"

    # Initializing competitive layer weights
    self._competitive_layer_weights = random.RandomState().rand(n_subclass, n_feature)
    # Normalizing competitive layer weights
    for i in range (n_subclass):
      if self._weights_normalization == "length":
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
    
    # Label encoder
    self._label_encoder = LabelEncoder()

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
    if len(data.shape) != 2:
      raise Exception("Data is expected to be 2D array")
    elif data.shape[1] != self._n_feature:
      raise Exception("Data must have the same number of features as the number of features of the model")
    for i in range (self._n_subclass):
      # Initializing the weights, picking random sample from data
      rand_idx = random.random_integers(0, len(data) - 1)
      self._competitive_layer_weights[i] = data[rand_idx]
      # Normalizing the weights
      if self._weights_normalization == "length":
        norm = fast_norm(self._competitive_layer_weights[i])
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm
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
    if y_hat == y:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] + alpha * (x - self._competitive_layer_weights[win_idx])
    else:
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] - alpha * (x - self._competitive_layer_weights[win_idx])
    # Normalizing the weights
    if self._weights_normalization == "length":
      norm = fast_norm(self._competitive_layer_weights[win_idx])
      self._competitive_layer_weights[win_idx] = self._competitive_layer_weights[win_idx] / norm

  def fit(self, X, y, num_iteration, epoch_size):
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
    
    Returns
    -------
    self : object
      Returns self.
    """
    if len(X.shape) <= 1:
      raise Exception("Data is expected to be 2D array")
    elif X.shape[1] != self._n_feature:
      raise Exception("Data must have the same number of features as the number of features of the model")
    y = y.astype(np.int8)
    y = self._label_encoder.fit_transform(y)
    self.train_batch(X, y, num_iteration, epoch_size)

    return self

  def train_batch(self, X, y, num_iteration, epoch_size):
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

    Returns
    -------
    self : object
      Returns self.
    """
    iteration = 0
    while iteration < num_iteration:
      idx = iteration % len(X)
      epoch = iteration // epoch_size
      self.update(x = X[idx], y = y[idx], epoch = epoch)
      iteration += 1
    return self

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
      If confidence is true, returns confidence scores of prediction has been made.
    """
    y_pred = array([]).astype(np.int8)
    confidence_score = array([])
    k = self._n_subclass // 20
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

    y_pred = self._label_encoder.inverse_transform(y_pred)
    
    if confidence:
      return y_pred, confidence_score
    else:
      return y_pred

  def details(self):
    """Prints parameters of lvq"""
    print("Competitive layer weights")
    print(self._competitive_layer_weights)
    print("Linear layer weights")
    print(self._linear_layer_weights)
    print("Biases")
    print(self._biases)
    print("Winner neurons count")
    print(self._winner_count)

class LvqNetworkWithNeighborhood(LvqNetwork):
  """Learning Vector Quantization with Neighborhood concept.

  Parameters
  ----------

  n_feature : int
    Number of features of the dataset.

  n_rows : int
    Number of rows in the competitive layer.

  n_cols : int
    Number of columns in the competitive layer.

  n_class : int
    Number of classes of the dataset.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias_function : function, default: None
    Function that updates biases value of neurons in the competitive layer.

  weights_normalization : option ["length"], default: None
    Normalizing weights of neuron.

  sigma : float
    Radius of neighborhood around winner neuron in the competitive layer.

  sigma_decay_function : function, default: None
    Function that decreases sigma after number of iterations.

  sigma_decay_rate : float, default: 1
    Reduction rate of sigma after number of iterations.

  neighborhood : option ["bubble", "gaussian"], default: None
    Function that determines coefficient for neighbors of winner neuron in competitive layer.
  """
  def __init__(self, n_feature, n_rows, n_cols, n_class,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias_function = None, weights_normalization = None,
              sigma = 0, sigma_decay_function = None, sigma_decay_rate = 1,
              neighborhood = None):
    super().__init__(n_feature = n_feature, n_subclass = n_rows * n_cols, n_class = n_class,
                    learning_rate = learning_rate, learning_rate_decay_function = learning_rate_decay_function,
                    decay_rate = decay_rate,
                    bias_function = bias_function, weights_normalization = weights_normalization)
    self._n_rows_subclass = n_rows
    self._n_cols_subclass = n_cols
    self._radius = sigma
    self._radius_decay_rate = sigma_decay_rate
    self._neighborhood_function = neighborhood
    if sigma_decay_function:
      self._radius_decay_function = sigma_decay_function
    else:
      self._radius_decay_function = default_radius_decay_function
  
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
    if len(data.shape) != 2:
      raise Exception("Data is expected to be 2D array")
    elif data.shape[1] != self._n_feature:
      raise Exception("Data must have the same number of features as the number of features of the model")

    self._competitive_layer_weights = zeros((self._n_subclass, self._n_feature))
    pca_number_of_components = None
    coord = None
    if self._n_cols_subclass == 1 or self._n_rows_subclass == 1 or data.shape[1] == 1:
      pca_number_of_components = 1
      if self._n_cols_subclass == self._n_rows_subclass:
        coord = array([[1], [0]])
        print(coord)
        print(coord[0][0])
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
    print(eigvec)
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
    radius = self._radius_decay_function(self._radius, epoch, self._radius_decay_rate)
    correlation = self.neighborhood(win_idx, radius)
    is_class = None
    if y is not None:
      is_class = self.is_class(y)
    else:
      is_class = ones(self._n_subclass)
    for i in range(self._n_subclass):
      self._competitive_layer_weights[i] = self._competitive_layer_weights[i] + is_class[i] * alpha * correlation[i] * (x - self._competitive_layer_weights[i])
      # Normalizing the weights
      if self._weights_normalization == "length":
        norm = fast_norm(self._competitive_layer_weights[i])
        self._competitive_layer_weights[i] = self._competitive_layer_weights[i] / norm

    return self

  def visualize(self, figure_path = None):
    """Visualizing the competitive layer.
    
    Parameters
    ----------
    figure_path: str
      The path of file to save figure, if there is no path provided, figure will be shown
    """
    # Rescaling weights to (0, 1) range
    from sklearn.preprocessing import MinMaxScaler
    sc_weights = MinMaxScaler(feature_range=(0, 1))
    weights = np.copy(self._competitive_layer_weights)
    weights = sc_weights.fit_transform(weights)

    # Parameters
    n_subclass = self._n_subclass
    n_class = self._n_class
    n_rows = self._n_rows_subclass
    n_cols = self._n_cols_subclass
    n_feature = self._n_feature

    # Meshgrid of the layer
    meshgrid = np.zeros((n_rows, n_cols))
    for idx in range (n_subclass):
      i = n_rows - 1 - (idx // n_cols)
      j = idx % n_cols
      for c in range (n_class):
        if self._linear_layer_weights[c][idx] == 1:
          meshgrid[i][j] = c
          break
    meshgrid = meshgrid.astype(np.int8)
    meshgrid = self._label_encoder.inverse_transform(meshgrid)

    # Drawing meshgrid of the layer
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = (8, 8))
    global_ax = fig.add_axes([0, 0, 1, 1])
    global_ax.pcolormesh(meshgrid, edgecolors = 'black', linewidth = 0.1, alpha = 0.3)
    global_ax.set_yticklabels([])
    global_ax.set_xticklabels([])

    # Drawing for each subclass of the layer
    for idx in range (n_subclass):
      i = n_rows - 1 - (idx // n_cols)
      j = idx % n_cols
      cell_width = 1 / n_rows
      cell_height = 1 / n_cols

      # Name of the class, to which subclass belongs
      grid_ax = fig.add_axes([j / n_cols, i / n_rows, cell_width, cell_height], polar=False, frameon = False)
      grid_ax.axis('off')
      grid_ax.text(0.05, 0.05, int(meshgrid[i][j]))
      
      # Pie chart corresponding to weight
      polar_ax = fig.add_axes([j / n_cols + cell_height * 0.1, i / n_rows + cell_width * 0.1, cell_width * 0.8, cell_height * 0.8], polar=True, frameon = False)
      polar_ax.axis('off')
      theta = np.array([])
      width = np.array([])  
      for k in range (n_feature):
        theta = np.append(theta, k * 2 * np.pi / n_feature)
        width = np.append(width, 2 * np.pi / n_feature)
      radii = weights[idx]
      color = ['b', 'g', 'r', 'black', 'm', 'y', 'k', 'w']
      bars = polar_ax.bar(theta, radii, width=width, bottom=0.0)
      for k in range (n_feature):
        bars[k].set_facecolor(color[k])
        bars[k].set_alpha(1)
    if figure_path:
      plt.savefig(figure_path)
    else:
      plt.show()

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

  n_class : int
    Number of classes of the dataset.

  learning_rate : float, default: 0.5
    Learning rate of the algorithm.

  learning_rate_decay_function : function, default: None
    Function that decreases learning rate after number of iterations.
  
  decay_rate : float, default: 1
    Reduction rate of learning rate after number of iterations.
  
  bias_function : function, default: None
    Function that updates biases value of neurons in the competitive layer.

  weights_normalization : option ["length"], default: None
    Normalizing weights of neuron.

  sigma : float
    Radius of neighborhood around winner neuron in the competitive layer.

  sigma_decay_function : function, default: None
    Function that decreases sigma after number of iterations.

  sigma_decay_rate : float, default: 1
    Reduction rate of sigma after number of iterations.

  neighborhood : option ["bubble", "gaussian"], default: None
    Function that determines coefficient for neighbors of winner neuron in competitive layer.
  """
  def __init__(self, n_feature, n_rows, n_cols, n_class,
              learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
              bias_function = None, weights_normalization = None,
              sigma = 0, sigma_decay_function = None, sigma_decay_rate = 1,
              neighborhood = None):
    super().__init__(n_feature, n_rows, n_cols, n_class,
                    learning_rate = 0.5, learning_rate_decay_function = None, decay_rate = 1,
                    bias_function = None, weights_normalization = None,
                    sigma = 0, sigma_decay_function = None, sigma_decay_rate = 1,
                    neighborhood = None)
    self._linear_layer_weights = zeros((n_class, n_rows * n_cols))

  def train_competitive(self, X, num_iteration, epoch_size):
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

    Returns
    -------
    self : object
      Returns self.
    """
    iteration = 0
    while iteration < num_iteration:
      idx = iteration % len(X)
      epoch = iteration // epoch_size
      self.update(x = X[idx], epoch = epoch)
      iteration += 1
    return self

  def label_neurons(self, X, y):
    """Labeling class for each neurons in the competitve layer according to the most used class.

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
    self._number_of_neurons_each_classes = zeros(self._n_class)
    class_win = zeros((self._n_subclass, self._n_class))
    m = len(X)
    for idx in range (m):
      win = self.winner(X[idx])
      win_idx = argmax(win)
      class_win[win_idx][y[idx]] += 1

    for idx in range (self._n_subclass):
      neuron_class_win = argwhere(class_win[idx] == amax(class_win[idx])).ravel()
      class_name = neuron_class_win[argmin(self._number_of_neurons_each_classes[neuron_class_win])]
      self._number_of_neurons_each_classes[class_name] += 1
      self._linear_layer_weights[class_name][idx] = 1

    return self

  def fit(self, X, y, first_num_iteration, first_epoch_size, second_num_iteration, second_epoch_size):
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

    Returns
    -------
    self : object
      Returns self.
    """
    if len(X.shape) <= 1:
      raise Exception("Data is expected to be 2D array")
    elif X.shape[1] != self._n_feature:
      raise Exception("Data must have the same number of features as the number of features of the model")
    y = y.astype(np.int8)
    y = self._label_encoder.fit_transform(y)
    
    # Phase 1: Training using SOM concept
    self.train_competitive(X, first_num_iteration, first_epoch_size)
    self.label_neurons(X, y)
    # Phase 2: Training using LVQ concept
    self.train_batch(X, y, second_num_iteration, second_epoch_size)

    return self