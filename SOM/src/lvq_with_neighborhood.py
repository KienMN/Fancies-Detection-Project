# LVQ with Neighborhood Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the LVQ
from lvq_network import LvqNetworkWithNeighborhood
lvq = LvqNetworkWithNeighborhood(n_feature = 7, n_rows = 9, n_cols = 9, n_class = 3,
                                learning_rate = 0.5, decay_rate = 1,
                                sigma = 2, sigma_decay_rate = 1,
                                neighborhood="bubble")
lvq.sample_weights_init(X_train)
# lvq.pca_weights_init(X_train)
lvq.train_batch(X_train, y_train, num_iteration = 10000, epoch_size = len(X_train))

# Predict the result
y_pred = lvq.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm))

# lvq.details()