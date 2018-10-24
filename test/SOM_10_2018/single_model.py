# LVQ with Neighborhood Model

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-4X.csv')
dataset = pd.read_csv(filepath)
X_train = dataset.iloc[:, 4: -1].values
y_train = dataset.iloc[:, -1].values.astype(np.int8)

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv')
dataset = pd.read_csv(filepath)
X_test = dataset.iloc[:, 4: -1].values
y_test = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, shuffle = True)

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
lvq = AdaptiveLVQ(n_rows = 15, n_cols = 15,
                  learning_rate = 0.5, decay_rate = 1,
                  sigma = 2, sigma_decay_rate = 1,
                  # weights_normalization = "length",
                  bias = False, weights_init = 'pca',
                  neighborhood='bubble', label_weight = 'inverse_distance_to_classes')
lvq.fit(X_train, y_train, first_num_iteration = 00, first_epoch_size = 400, second_num_iteration = 00, second_epoch_size = 400)

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(random_state = 0)
# classifier.fit(X_train, y_train)

# Predict the result
y_pred, confidence_score = lvq.predict(X_test, confidence = 1, crit = 'winner_neuron')
# y_pred = classifier.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)
print('confidence', confidence_score)
print(confidence_score.shape)
print(y_pred.shape)
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))