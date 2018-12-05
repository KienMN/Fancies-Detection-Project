# LVQ two phases

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Inputing arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-m', '--model_name')
parser.add_argument('-s', '--size', type = int)
parser.add_argument('-fi', '--first_iterations', type = int)
parser.add_argument('-fe', '--first_epoch_size', type = int)
parser.add_argument('-si', '--second_iterations', type = int)
parser.add_argument('-se', '--second_epoch_size', type = int)
parser.add_argument('-lr', '--learning_rate', type = float)
parser.add_argument('-sig', '--sigma', type = float)
parser.add_argument('-ne', '--neighborhood', type = str)
args = parser.parse_args()

# Hyper parameters
size = args.size
first_iterations = args.first_iterations
first_epoch_size = args.first_epoch_size
second_iterations = args.second_iterations
second_epoch_size = args.second_epoch_size
learning_rate = args.learning_rate
sigma = args.sigma
neighborhood = args.neighborhood

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/train_dataset.csv')
dataset = pd.read_csv(filepath)
X_train = dataset.iloc[:, : -1].values
y_train = dataset.iloc[:, -1].values.astype(np.int8)

filepath = os.path.join(os.path.dirname(__file__), 'data/test_dataset.csv')
dataset = pd.read_csv(filepath)
X_test = dataset.iloc[:, : -1].values
y_test = dataset.iloc[:, -1].values.astype(np.int8)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Training the LVQ
from detection.competitive_learning_network import AdaptiveLVQ
lvq = AdaptiveLVQ(n_rows = size, n_cols = size,
                  learning_rate = learning_rate, decay_rate = 1,
                  sigma = sigma, sigma_decay_rate = 1,
                  bias = False, weights_init = 'pca',
                  neighborhood = neighborhood, label_weight = 'exponential_distance')

lvq.fit(X_train, y_train,
        first_num_iteration = first_iterations, first_epoch_size = first_epoch_size,
        second_num_iteration = second_iterations, second_epoch_size = second_epoch_size)

# Predict the result
y_pred = lvq.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))