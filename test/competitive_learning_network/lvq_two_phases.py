# LVQ with Neighborhood Model

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/SD-3X_rocktype.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 0: -1].values
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
from detection.competitive_learning_network import AdaptiveLVQ
lvq = AdaptiveLVQ(n_feature = 6, n_rows = 8, n_cols = 8, n_class = 4,
                                learning_rate = 0.5, decay_rate = 1,
                                sigma = 1, sigma_decay_rate = 1,
                                # weights_normalization = "length",
                                neighborhood="gaussian")
# lvq.sample_weights_init(X_train)
lvq.pca_weights_init(X_train)
lvq.fit(X_train, y_train, first_num_iteration = 4000, first_epoch_size = 100, second_num_iteration = 4000, second_epoch_size = 100)

# Predict the result
y_pred = lvq.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / np.sum(cm))

# Visualization
# lvq.details()
print(lvq._number_of_neurons_each_classes)
# lvq.visualize()