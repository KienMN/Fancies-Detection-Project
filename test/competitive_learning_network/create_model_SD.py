# LVQ with Neighborhood Model

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/processed_15_1-SD-6X_LQC.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
ssom = AdaptiveLVQ(n_rows = 6, n_cols = 6,
                  learning_rate = 0.5, decay_rate = 1,
                  sigma = 1, sigma_decay_rate = 1,
                  bias = True, weights_init = 'pca',
                  neighborhood = 'gaussian', label_weight = 'inverse_distance')
ssom.fit(X_train, y_train, first_num_iteration = 1500, first_epoch_size = 200, second_num_iteration = 1500, second_epoch_size = 200)

# Predict the result
y_pred, confidence_score = ssom.predict(X_test, confidence = 1, crit='winner_neuron')
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

# Dumping the trained model
from sklearn.externals import joblib
model_id = 'model-SD-6X'
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dump_model/' + model_id + ".sav")
joblib.dump(ssom, model_filepath)