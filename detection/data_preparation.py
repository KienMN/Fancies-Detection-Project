# Data preparation

# Importing the libraries
import numpy as np
import pandas as pd
import os

# Importing the dataset
# filename = 'SVNE-2P_SVNE-2P-new.csv'
# file_directory = os.path.dirname(os.getcwd()) + '/data/'
# filepath = file_directory + filename
# fieldnames = ['wellName', 'datasetName', 'DEPTH', 'DepoFacies', 'DT', 'Facies', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL']
# dataset = pd.read_csv(filepath, skiprows=range(1, 2))
# dataset.round({'DepoFacies': 0})
# X = dataset.iloc[:, [2, 4, 6, 7, 8, 9, 10]].values
# y = dataset.iloc[:, [3]].values

# Transforming the dataset
# new_dataset = np.append(X, y, axis = 1)
# new_fieldnames = ['DEPTH', 'DT', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL', 'DepoFacies']
# new_dataset = pd.DataFrame(new_dataset, columns=new_fieldnames)
# new_dataset.to_csv(file_directory + 'processed_' + filename, index=False)

def duplicate_data(X, total_length, step, y = None, random_order = False):
  X_dup = []
  y_dup = []
  X_i = []
  y_i = []
  length_of_data = len(X)
  if y is None:
    for idx in range (total_length):
      i = idx % length_of_data
      X_i.append(X[i])
      if len(X_i) == step:
        X_dup.append(np.array(X_i))
        X_i = []
    if (len(X_i) != 0):
      X_dup.append(np.array(X_i))
  else:
    if length_of_data != len(y):
      raise Exception("Data must be same length")
    for idx in range (total_length):
      i = idx % length_of_data
      X_i.append(X[i])
      y_i.append(y[i])
      if len(X_i) == step:
        X_dup.append(np.array(X_i))
        y_dup.append(np.array(y_i))
        X_i = []
        y_i = []
    if (len(X_i) != 0):
      X_dup.append(np.array(X_i))
      y_dup.append(np.array(y_i))  
  if y is None:
    return X_dup
  else:
    return X_dup, y_dup

# if __name__ == '__main__':
#   X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
#   y = np.array([1, 2, 3, 4])
#   X_dup, y_dup = duplicate_data(X, y = y, total_length = 12, step = 3)
#   for X in X_dup:
#     print(X)
#   for y in y_dup:
#     print(y)