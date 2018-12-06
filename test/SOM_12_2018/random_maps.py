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
parser.add_argument('-n', '--num_estimators')
parser.add_argument('-s', '--size', type = int)
parser.add_argument('-fi', '--first_iterations', type = int)
parser.add_argument('-fe', '--first_epoch_size', type = int)
parser.add_argument('-si', '--second_iterations', type = int)
parser.add_argument('-se', '--second_epoch_size', type = int)
parser.add_argument('-lr', '--learning_rate', type = float)
parser.add_argument('-sig', '--sigma', type = float)
parser.add_argument('-c', '--used_cols')
parser.add_argument('-fa', '--features_array')
parser.add_argument('-mm', '--max_maps_each_features', type = int)
args = parser.parse_args()

# Importing the dataset
used_cols = [int(i) for i in args.used_cols.split(',')]
features_array = None
if args.features_array is not None:
  features_array = [[int(x) for x in arr.split(',')] 
                    for arr in args.features_array.split('-')]
  

filepath = os.path.join(os.path.dirname(__file__), 'data/train_dataset.csv')
dataset = pd.read_csv(filepath)
X_train = dataset.iloc[:, used_cols].values
y_train = dataset.iloc[:, -1].values.astype(np.int8)

filepath = os.path.join(os.path.dirname(__file__), 'data/test_dataset.csv')
dataset = pd.read_csv(filepath)
X_test = dataset.iloc[:, used_cols].values
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
n_estimators = int(args.num_estimators)
size = args.size
first_iterations = args.first_iterations
first_epoch_size = args.first_epoch_size
second_iterations = args.second_iterations
second_epoch_size = args.second_epoch_size
learning_rate = args.learning_rate
sigma = args.sigma
max_maps_each_features = args.max_maps_each_features

from detection.competitive_learning_network.combination import RandomMaps

classifier = RandomMaps(n_estimators = n_estimators, size = size,
                        learning_rate = learning_rate , decay_rate = 1,
                        sigma = sigma, sigma_decay_rate = 1,
                        label_weight = 'exponential_distance')

classifier.fit(X_train, y_train,
              max_first_iters = first_iterations, first_epoch_size = first_epoch_size,
              max_second_iters = second_iterations, second_epoch_size = second_epoch_size,
              features_arr = features_array, max_maps_each_features = max_maps_each_features)

# Predict the result
y_pred = classifier.predict(X_test, crit = 'max_voting')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print("Max voting")
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Predict the result
y_pred = classifier.predict(X_test, crit = 'confidence_score')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print("Max confidence score")
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Predict the result
y_pred = classifier.predict(X_test, crit = 'distance')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print("Distance")
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))