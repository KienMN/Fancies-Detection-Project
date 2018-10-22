# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-3X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Loading trained model
from sklearn.externals import joblib
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-RUBY-1X.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-RUBY-1X.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-RUBY-1X.sav')

lvq = joblib.load(model_filepath)
encoder = joblib.load(label_filepath)
sc = joblib.load(scaler_filepath)

# Predicting
X_test = sc.transform(X)
y_pred_1, confidence_score_1 = lvq.predict(X_test, confidence = 1, crit = 'class_distance')
y_pred_1 = encoder.inverse_transform(y_pred_1)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_1)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Loading trained model
from sklearn.externals import joblib
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-RUBY-2X.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-RUBY-2X.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-RUBY-2X.sav')

lvq = joblib.load(model_filepath)
encoder = joblib.load(label_filepath)
sc = joblib.load(scaler_filepath)

# Predicting
X_test = sc.transform(X)
y_pred_2, confidence_score_2 = lvq.predict(X_test, confidence = 1, crit = 'class_distance')
y_pred_2 = encoder.inverse_transform(y_pred_2)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_2)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Loading trained model
from sklearn.externals import joblib
model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-RUBY-4X.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-RUBY-4X.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-RUBY-4X.sav')

lvq = joblib.load(model_filepath)
encoder = joblib.load(label_filepath)
sc = joblib.load(scaler_filepath)

# Predicting
X_test = sc.transform(X)
y_pred_3, confidence_score_3 = lvq.predict(X_test, confidence = 1, crit = 'class_distance')
y_pred_3 = encoder.inverse_transform(y_pred_3)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_3)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))

# Combining
confidence_score_1 = confidence_score_1.reshape((-1, 1))
confidence_score_2 = confidence_score_2.reshape((-1, 1))
confidence_score_3 = confidence_score_3.reshape((-1, 1))
y_pred_1 = y_pred_1.reshape((-1, 1))
y_pred_2 = y_pred_2.reshape((-1, 1))
y_pred_3 = y_pred_3.reshape((-1, 1))

confidence_score = np.append(confidence_score_1, confidence_score_2, axis = 1)
confidence_score = np.append(confidence_score, confidence_score_3, axis = 1)
y_pred = np.append(y_pred_1, y_pred_2, axis = 1)
y_pred = np.append(y_pred, y_pred_3, axis = 1)

print(confidence_score.shape)
print(y_pred.shape)

y_pred_combined = np.array([]).astype(np.int8)

n_samples = len(y)
for i in range (n_samples):
  j = np.argmax(confidence_score[i])
  y_pred_combined = np.append(y_pred_combined, y_pred[i, j])

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_combined)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print(true_result / np.sum(cm))