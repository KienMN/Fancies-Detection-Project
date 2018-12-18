from .lvq_network import AdaptiveLVQ
import random
import numpy as np
from .utils import weighted_sampling

class RandomMaps(object):
  def __init__(self, n_estimators, size,
              learning_rate, decay_rate, sigma, sigma_decay_rate, label_weight):
    self._n_estimators = n_estimators
    self._size = size
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate
    self._sigma = sigma 
    self._sigma_decay_rate = sigma_decay_rate
    self._label_weight = label_weight
    self._models = []
    self._features_each_models = []

  def fit(self, X, y, max_first_iters, first_epoch_size, max_second_iters, second_epoch_size, features_arr = None, max_maps_each_features = None):
    
    if features_arr is None:
      # features_arr = [[]]
      # for i in range (X.shape[1]):
      #   features_arr[0].append(i)
      corr_coef = np.corrcoef(np.append(X, y.reshape((-1, 1)), axis = 1).T)[-1, :-1]
      number_of_features = X.shape[1]
      weights = corr_coef.copy()
      for i in range (self._n_estimators):
        features = np.unique(weighted_sampling(weights, number_of_features))
        neighborhood = None
        if (random.randint(0, 1)):
          neighborhood = 'gaussian'
        else:
          neighborhood = 'bubble'
        X_train = X[:, features]
        
        lvq = AdaptiveLVQ(n_rows = self._size, n_cols = self._size, bias = False, learning_rate = self._learning_rate,
                          decay_rate = self._decay_rate, sigma = self._sigma, sigma_decay_rate = self._sigma_decay_rate,
                          neighborhood = neighborhood, label_weight = self._label_weight, weights_init = "sample", verbose = 0)
        lvq.fit(X_train, y, first_num_iteration = max_first_iters, first_epoch_size = first_epoch_size,
                second_num_iteration = max_second_iters, second_epoch_size = second_epoch_size)
        self._features_each_models.append(features)
        self._models.append(lvq)

    else:
      if max_maps_each_features is None:
        max_maps_each_features = self._n_estimators // len(features_arr)
      
      # Trick to handle later
      self._n_estimators = max_maps_each_features * len(features_arr)
      
      for i in range (max_maps_each_features):
        for features in features_arr:
          neighborhood = None
          if (random.randint(0, 1)):
            neighborhood = 'gaussian'
          else:
            neighborhood = 'bubble'
          X_train = X[:, features]
          
          lvq = AdaptiveLVQ(n_rows = self._size, n_cols = self._size, bias = False, learning_rate = self._learning_rate,
                            decay_rate = self._decay_rate, sigma = self._sigma, sigma_decay_rate = self._sigma_decay_rate,
                            neighborhood = neighborhood, label_weight = self._label_weight, weights_init = "sample", verbose = 0)
          lvq.fit(X_train, y, first_num_iteration = max_first_iters, first_epoch_size = first_epoch_size,
                  second_num_iteration = max_second_iters, second_epoch_size = second_epoch_size)
          self._features_each_models.append(features)
          self._models.append(lvq)
          # print('Border neurons:', lvq.count_border_neurons())
    
  def predict(self, X, crit = 'max_voting'):
    '''
    crit = ['confidence_score', 'distance', 'max_voting']
    '''
    n_samples = len(X)
    n_classes = self._models[0]._n_class
    n_models = len(self._models)
    y_pred = None
    y_pred_final = np.array([]).astype(np.int8)
    if crit == 'max_voting' or crit == 'confidence_score':
      confidence_score = None
      for i in range(n_models):
        model = self._models[i]
        X_test = X[:, self._features_each_models[i]]
        if y_pred is None:
          y_pred, confidence_score = model.predict(X_test, confidence = True, crit = 'winner_neuron')
          y_pred = y_pred.reshape((-1, 1))
          confidence_score = confidence_score.reshape((-1, 1))
        else:
          y_pred_tmp, confidence_score_tmp = model.predict(X_test, confidence = True, crit = 'winner_neuron')
          y_pred_tmp = y_pred_tmp.reshape((-1, 1))
          confidence_score_tmp = confidence_score_tmp.reshape((-1, 1))
          y_pred = np.append(y_pred, y_pred_tmp, axis = 1)
          confidence_score = np.append(confidence_score, confidence_score_tmp, axis = 1)
      if crit == 'max_voting':
        for i in range (n_samples):
          y_pred_final = np.append(y_pred_final, int(np.bincount(y_pred[i]).argmax()))
      elif crit == 'confidence_score':
        for i in range (n_samples):
          y_i = y_pred[i, np.argmax(confidence_score[i])]
          y_pred_final = np.append(y_pred_final, y_i)
    elif crit == 'distance':
      distances = None
      for i in range (n_models):
        model = self._models[i]
        X_test = X[:, self._features_each_models[i]]
        if y_pred is None:
          y_pred, distances = model.distance_from_winner_neurons(X_test, prediction = True)
          y_pred = y_pred.reshape((-1, 1))
          distances = distances.reshape((-1, 1))
        else:
          y_pred_tmp, distances_tmp = model.distance_from_winner_neurons(X_test, prediction = True)
          y_pred_tmp = y_pred_tmp.reshape((-1, 1))
          distances_tmp = distances_tmp.reshape((-1, 1))
          y_pred = np.append(y_pred, y_pred_tmp, axis = 1)
          distances = np.append(distances, distances_tmp, axis = 1)
      
      weights = np.zeros((n_samples, n_classes))
      for i in range (n_samples):
        for j in range (n_models):
          weights[i, y_pred[i, j]] += (1 / distances[i, j])
        y_i = np.argmax(weights[i])
        y_pred_final = np.append(y_pred_final, y_i)
    return y_pred_final