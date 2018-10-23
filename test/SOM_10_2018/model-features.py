# LVQ with Neighborhood Model

# Inputing arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-tr', '--train_dataset')
parser.add_argument('-te', '--test_dataset')
parser.add_argument('-m', '--model_name')
parser.add_argument('-s', '--size', type = int)
parser.add_argument('-fi', '--first_iterations', type = int)
parser.add_argument('-si', '--second_iterations', type = int)
parser.add_argument('-lr', '--learning_rate', type = float)
parser.add_argument('-sig', '--sigma', type = float)
parser.add_argument('-n', '--neighborhood')
parser.add_argument('-c', '--used_cols')
args = parser.parse_args()

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

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
size = args.size
first_iterations = args.first_iterations
second_iterations = args.second_iterations
learning_rate = args.learning_rate
sigma = args.sigma
neighborhood = args.neighborhood

from detection.competitive_learning_network import AdaptiveLVQ
lvq = AdaptiveLVQ(n_rows = size, n_cols = size,
                  learning_rate = learning_rate, decay_rate = 1,
                  sigma = sigma, sigma_decay_rate = 1,
                  # weights_normalization = "length",
                  bias = False, weights_init = 'pca',
                  neighborhood = neighborhood, label_weight = 'inverse_distance_to_classes')

# Training phase 1
lvq.fit(X_train, y_train, first_num_iteration = first_iterations, first_epoch_size = 2000, second_num_iteration = 0, second_epoch_size = 2000)

# Predict the result
y_pred, confidence_score = lvq.predict(X_test, confidence = 1, crit = 'winner_neuron')
y_pred = encoder.inverse_transform(y_pred)
# print('confidence', confidence_score)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Dumping the models
model_name = 'SOM-' + args.model_name
from sklearn.externals import joblib
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-' + model_name + '.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-' + model_name + '.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-' + model_name +'.sav')
joblib.dump(lvq, model_filepath)
joblib.dump(encoder, label_filepath)
joblib.dump(sc, scaler_filepath)

# Trainging phase 2
lvq.fit(X_train, y_train, first_num_iteration = 0, first_epoch_size = 2000, second_num_iteration = second_iterations, second_epoch_size = 2000)

# Predict the result
y_pred, confidence_score = lvq.predict(X_test, confidence = 1, crit = 'winner_neuron')
y_pred = encoder.inverse_transform(y_pred)
# print('confidence', confidence_score)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Dumping the models
model_name = 'SSOM-' + args.model_name
from sklearn.externals import joblib
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-' + model_name + '.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-' + model_name + '.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-' + model_name +'.sav')
joblib.dump(lvq, model_filepath)
joblib.dump(encoder, label_filepath)
joblib.dump(sc, scaler_filepath)