import numpy as np

class CombinedSom(object):
  
  def __init__(self, models):
    self._models = models

  def winner_takes_all(self, X, crit):
    """
    Parameters
    ----------
    crit : ['distance', 'confidence_score']
    """
    n_sample = len(X)
    y_pred = np.array([]).astype(np.int8)
    if crit == 'confidence_score':
      predictions = []
      scores = []
      for model in self._models:
        pred, score = model.predict(X, confidence = 1, crit = 'winner_neuron')
        predictions.append(pred)
        scores.append(score)
      predictions = np.array(predictions)
      scores = np.array(scores)
      winners = np.argmax(scores, axis = 0)
      for i in range (n_sample):
        y_pred = np.append(y_pred, predictions[winners[i]][i])
    
    elif crit == 'distance':
      predictions = []
      distances = []
      for model in self._models:
        pred = np.array([]).astype(np.int8)
        distance = np.array([])
        for i in range (n_sample):
          x = X[i]
          win = model.winner(x)
          y_i = int(model.classify(win))
          pred = np.append(pred, y_i)
          distance = np.append(distance, model.distance_from_winner(x))
        predictions.append(pred)
        distances.append(distance)

      predictions = np.array(predictions)
      distances = np.array(distances)
      winners = np.argmin(distances, axis = 0)
      for i in range (n_sample):
        y_pred = np.append(y_pred, predictions[winners[i]][i])
    
    return y_pred

  def major_voting(self, crit = 'distance'):
    """
    Parameters
    ----------
    crit : ['distance']
    """
    pass

  def combined_with_weights(self, X):
    n_sample = len(X)
    n_class = len(self._models[0]._linear_layer_weights)
    y_pred = np.array([]).astype(np.int8)
    scores = np.zeros((n_sample, n_class))
    for i in range (n_sample):
      x = X[i]
      for model in self._models:
        win = model.winner(x)
        win_idx = np.argmax(win)
        distance = model.distance_from_winner(x)
        for j in range (n_class):
          scores[i][j] += 1 / distance * model._neurons_confidence[win_idx][j]
      y_pred = np.append(y_pred, int(np.argmax(scores[i, :])))
    return y_pred

  def combined_with_confidence_score(self, X):
    n_sample = len(X)
    n_class = len(self._models[0]._linear_layer_weights)
    y_pred = np.array([]).astype(np.int8)
    scores = np.zeros((n_sample, n_class))
    for i in range (n_sample):
      x = X[i]
      for model in self._models:
        win = model.winner(x)
        win_idx = np.argmax(win)
        # distance = model.distance_from_winner(x)
        for j in range (n_class):
          scores[i][j] += model._neurons_confidence[win_idx][j]
      y_pred = np.append(y_pred, int(np.argmax(scores[i, :])))
    return y_pred

  def major_voting_with_confidence_score(self):
    pass

  def predict(self, X, option):
    """
    Parameters
    ----------
    option : options ['']
    """
    pass