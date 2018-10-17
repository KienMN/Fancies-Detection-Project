# Testing stacking

# Adding path
import os
import sys

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/dataset1.csv')
dataset = pd.read_csv(filepath)
dataset = dataset.drop(dataset[dataset.target < 0].index)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into the training and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training seperated models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

print('number of training samples:', len(X_train))
k_folds = 10
y_pred_level_1 = np.array([[0, 0]])

for i in range (k_folds):
  n_rows = len(X_train) // k_folds
  if i != k_folds - 1:
    idx = np.arange(i * n_rows, (i + 1) * n_rows)
  else:
    idx = np.arange(i * n_rows, len(X_train))

  training_idx = [i for i in range (len(X_train)) if i not in idx]
  
  linear_reg = LinearRegression()
  svr = SVR(kernel='rbf')
  
  linear_reg.fit(X_train[training_idx, :], y_train[training_idx])
  svr.fit(X_train[training_idx, :], y_train[training_idx])

  y_pred_1 = linear_reg.predict(X_train[idx]).reshape((-1, 1))
  y_pred_2 = svr.predict(X_train[idx]).reshape((-1, 1))
  y_pred = np.append(y_pred_1, y_pred_2, axis = 1)
  y_pred_level_1 = np.append(y_pred_level_1, y_pred, axis = 0)

y_pred_level_1 = y_pred_level_1[1:, :]

meta_learner = LinearRegression()
meta_learner.fit(y_pred_level_1, y_train)

linear_reg = LinearRegression()
svr = SVR(kernel='rbf')

linear_reg.fit(X_train, y_train)
svr.fit(X_train, y_train)

y_pred_1 = linear_reg.predict(X_test)
y_pred_2 = svr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print('Linear regression:', mean_squared_error(y_test, y_pred_1), r2_score(y_test, y_pred_1))
print('SVC:', mean_squared_error(y_test, y_pred_2), r2_score(y_test, y_pred_2))

y_level_1 = np.append(y_pred_1.reshape((-1, 1)), y_pred_2.reshape((-1, 1)), axis = 1)
y_pred = meta_learner.predict(y_level_1)
print('Stacking:', mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))