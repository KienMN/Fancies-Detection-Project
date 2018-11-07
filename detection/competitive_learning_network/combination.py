from .lvq_network import AdaptiveLVQ
import random
import numpy as np

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

  def fit(self, X, y, max_first_iters, first_epoch_size, max_second_iters, second_epoch_size):
    for i in range (self._n_estimators):
      neighborhood = None
      if (random.randint(0, 1)):
        neighborhood = 'gaussian'
      else:
        neighborhood = 'bubble'

      lvq = AdaptiveLVQ(n_rows = self._size, n_cols = self._size, learning_rate = self._learning_rate,
                        decay_rate = self._decay_rate, sigma = self._sigma, sigma_decay_rate = self._sigma_decay_rate,
                        neighborhood = neighborhood, label_weight = self._label_weight, weights_init = "sample")
      lvq.fit(X, y, first_num_iteration = max_first_iters, first_epoch_size = first_epoch_size,
              second_num_iteration = max_second_iters, second_epoch_size = second_epoch_size)
      
      self._models.append(lvq)
    
  def predict(self, X, crit = 'max_voting'):
    '''
    crit = ['confidence_score', 'distance', 'max_voting']
    '''
    n_sample = len(X)
    y_pred = None
    confidence_score = None
    y_pred_final = np.array([]).astype(np.int8)
    for model in self._models:
      if y_pred is None:
        y_pred, confidence_score = model.predict(X, confidence = True, crit = 'winner_neuron')
        y_pred = y_pred.reshape((-1, 1))
        confidence_score = confidence_score.reshape((-1, 1))
      else:
        y_pred_tmp, confidence_score_tmp = model.predict(X, confidence = True, crit = 'winner_neuron')
        y_pred_tmp = y_pred_tmp.reshape((-1, 1))
        confidence_score_tmp = confidence_score_tmp.reshape((-1, 1))
        y_pred = np.append(y_pred, y_pred_tmp, axis = 1)
        confidence_score = np.append(confidence_score, confidence_score_tmp, axis = 1)
    if crit == 'max_voting':
      for i in range (n_sample):
        y_pred_final = np.append(y_pred_final, int(np.bincount(y_pred[i]).argmax()))
    elif crit == 'confidence_score':
      for i in range (n_sample):
        y_i = y_pred[i, np.argmax(confidence_score[i])]
        y_pred_final = np.append(y_pred_final, y_i)
    return y_pred_final