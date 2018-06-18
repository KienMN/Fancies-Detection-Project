# LVQ with Neighborhood Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the LVQ
from lvq_network import LvqNetwork, LvqNetworkWithNeighborhood
lvq = LvqNetwork(n_feature = 7, n_subclass = 90, n_class = 3, learning_rate = 0.5, decay_rate = 1)
lvq.random_weights_init(X_train)
lvq.train_batch(X_train, y_train, num_iteration = 100, epoch_size = len(X_train))
lvq.details()