# Importing the libraries

import pandas as pd
import numpy as np
import os

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/RUBY-4X_RUBY-4X_1.csv')
dataset = pd.read_csv(filepath, skiprows=range(1, 2))
dataset = dataset.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9]]

print (len(dataset))

for i in range (dataset.shape[1]):
  dataset = dataset.drop(dataset[dataset.iloc[:, i] < 0].index)

dataset = dataset.drop(dataset[dataset.iloc[:, 1] > 4].index)

print(len(dataset))

fieldnames = ['Depth', 'GR', 'LLD', 'NPHI', 'PHIE', 'RHOB', 'VWCL', 'DEPOFACIES']

X = dataset.iloc[:, [0, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, [1]].values

for i in (np.unique(y)):
  print('Class', str(i), len(np.where(y == i)[0]), 'samples')

new_dataset = pd.DataFrame(np.append(X, y, axis = 1), columns = fieldnames)
new_dataset.iloc[:, -1] = new_dataset.iloc[:, -1].astype(int)

print(len(new_dataset))
new_dataset.to_csv(os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-4X.csv'), index = False)