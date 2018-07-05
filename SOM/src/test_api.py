import requests
from flask import jsonify
import json
import numpy as np
import pandas as pd
import os

headers = {"content-type": "application/json"}

payload = {}
payload['model_id'] = "model-2"
payload['params'] = {}
payload['train'] = {}

payload['params']['n_rows'] = 9
payload['params']['n_cols'] = 9
payload['params']['learning_rate'] = 0.5
payload['params']['decay_rate'] = 1
payload['params']['neighborhood'] = 'bubble'
payload['params']['sigma'] = 1
payload['params']['sigma_decay_rate'] = 1
payload['params']['weights_initialization'] = "pca"

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = X_train.T
X_test = X_test.T

payload['train']["data"] = X_train.tolist()
payload['train']["target"] = y_train.tolist()
payload['params']["num_iteration"] = 10000
payload['params']["epoch_size"] = len(X)

payload = json.dumps(payload)

re = requests.post("http://127.0.0.1:1234/api/v1.0/lvq/train", data = payload, headers = headers)
print(re.text)

payload = {}
payload["model_id"] = "model-2"
payload["data"] = X_test.tolist()
payload = json.dumps(payload)
re = requests.post("http://127.0.0.1:1234/api/v1.0/lvq/predict", data = payload, headers = headers)

res = re.json()
res = json.loads(res)
print(res.get('message'))
y_pred = res.get('y_pred')

y_pred = np.array(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm))