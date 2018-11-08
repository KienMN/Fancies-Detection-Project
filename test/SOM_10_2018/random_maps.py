# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputing arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-tr', '--train_dataset')
parser.add_argument('-te', '--test_dataset')
parser.add_argument('-m', '--model_name')
parser.add_argument('-n', '--num_estimators')
parser.add_argument('-s', '--size', type = int)
parser.add_argument('-fi', '--first_iterations', type = int)
parser.add_argument('-si', '--second_iterations', type = int)
parser.add_argument('-lr', '--learning_rate', type = float)
parser.add_argument('-sig', '--sigma', type = float)
parser.add_argument('-c', '--used_cols')
args = parser.parse_args()

# Importing the Training dataset
train_dataset_name = [a for a in args.train_dataset.split(',')]
used_cols = [int(i) for i in args.used_cols.split(',')]

filepath1 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-' + train_dataset_name[0] + '.csv')
filepath2 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-' + train_dataset_name[1] + '.csv')
filepath3 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-' + train_dataset_name[2] + '.csv')
dataset = pd.read_csv(filepath1)
dataset = dataset.append(pd.read_csv(filepath2), ignore_index = True)
dataset = dataset.append(pd.read_csv(filepath3), ignore_index = True)

X_train = dataset.iloc[:, used_cols].values
y_train = dataset.iloc[:, -1].values.astype(np.int8)

# Importing the Test dataset
test_dataset_name = args.test_dataset

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-' + test_dataset_name +'.csv')
test_dataset = pd.read_csv(filepath)
X_test = test_dataset.iloc[:, used_cols].values
y_test = test_dataset.iloc[:, -1].values.astype(np.int8)

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
from detection.competitive_learning_network.combination import RandomMaps
classifier = RandomMaps(n_estimators = args.num_estimators, size = args.size,
                        learning_rate = args.learning_rate , decay_rate = 1,
                        sigma = args.sigma, sigma_decay_rate = 1,
                        label_weight = 'inverse_distance')
classifier.fit(X_train, y_train, max_first_iters = args.first_iterations, first_epoch_size = 4000, max_second_iters = args.second_iterations, second_epoch_size = 4000)

# Predict the result
y_pred = classifier.predict(X_test, crit = 'max_voting')
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

# Predict the result
y_pred = classifier.predict(X_test, crit = 'confidence_score')
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