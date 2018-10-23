# Inputing arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-m', '--models_name')
parser.add_argument('-gc', '--used_group_cols')
args = parser.parse_args()

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset_name = args.dataset
group_cols = [a for a in args.used_group_cols.split('-')]
print(group_cols)
# used_cols = [int(i) for i in args.used_cols.split(',')]

filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-' + dataset_name + '.csv')
dataset = pd.read_csv(filepath)
# X = dataset.iloc[:, used_cols].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Loading trained model
models = [m for m in args.models_name.split(',')]
from sklearn.externals import joblib

model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-SSOM-' + models[0] + '.sav')
label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-SSOM-' + models[0] + '.sav')
scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-SSOM-' + models[0] + '.sav')

lvq = joblib.load(model_filepath)
encoder = joblib.load(label_filepath)
sc = joblib.load(scaler_filepath)

# Predicting
used_cols = [int(j) for j in group_cols[0].split(',')]
X_test = dataset.iloc[:, used_cols].values
X_test = sc.transform(X_test)
y_pred, confidence_score = lvq.predict(X_test, confidence = 1, crit = 'winner_neuron')
y_pred = encoder.inverse_transform(y_pred)
y_pred = y_pred.reshape((-1, 1))
confidence_score = confidence_score.reshape((-1, 1))

for i in range (1, len(models)):
  model_name = models[i]
  model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/model-SSOM' + model_name + '.sav')
  label_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/label-SSOM' + model_name + '.sav')
  scaler_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/scaler-SSOM' + model_name + '.sav')

  lvq = joblib.load(model_filepath)
  encoder = joblib.load(label_filepath)
  sc = joblib.load(scaler_filepath)

  # Predicting
  used_cols = [int(j) for j in group_cols[i].split(',')]
  X_test = dataset.iloc[:, used_cols].values
  X_test = sc.transform(X_test)
  y_pred_tmp, confidence_score_tmp = lvq.predict(X_test, confidence = 1, crit = 'winner_neuron')
  y_pred_tmp = encoder.inverse_transform(y_pred_tmp)
  y_pred = np.append(y_pred, y_pred_tmp.reshape((-1, 1)))
  confidence_score = np.append(confidence_score, confidence_score_tmp.reshape((-1, 1)))

# Combining
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