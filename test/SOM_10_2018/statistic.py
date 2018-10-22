# Importing the libraries
import pandas as pd
import numpy as np
import os
# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-1X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)
print("Dataset RUBY-1X:")
print("Number of samples:", len(y))
for i in np.unique(y):
  print("Facy {}: {} samples.".format(i, len(np.where(y == i)[0])))

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)
print("Dataset RUBY-2X:")
print("Number of samples:", len(y))
for i in np.unique(y):
  print("Facy {}: {} samples.".format(i, len(np.where(y == i)[0])))

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-3X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)
print("Dataset RUBY-3X:")
print("Number of samples:", len(y))
for i in np.unique(y):
  print("Facy {}: {} samples.".format(i, len(np.where(y == i)[0])))

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-4X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)
print("Dataset RUBY-4X:")
print("Number of samples:", len(y))
for i in np.unique(y):
  print("Facy {}: {} samples.".format(i, len(np.where(y == i)[0])))