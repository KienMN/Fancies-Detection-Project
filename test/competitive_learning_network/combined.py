# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/processed_15_1-SD-6X_LQC.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
X = sc.fit_transform(X)
# X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Loading models
from sklearn.externals import joblib
model_names = ['model-SD-2X', 'model-SD-3X', 'model-SD-5X']
models = []
try:
  for model_id in model_names:
    model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dump_model/' + model_id + ".sav")
    ssom = joblib.load(model_filepath)
    models.append(ssom)
except FileNotFoundError:
  print('File not found')

# Creating combined model
from detection.competitive_learning_network.combined_som import CombinedSom
combined_som = CombinedSom(models)

# Making prediction
y_pred = combined_som.winner_takes_all(X, crit = 'distance')

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))