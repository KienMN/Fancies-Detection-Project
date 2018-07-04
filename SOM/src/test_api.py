import requests
from flask import jsonify
import json
import numpy as np
import pandas as pd
import os

headers = {"content-type": "application/json"}

payload = {}
payload['model_id'] = "0"
payload['n_rows'] = 9
payload['n_cols'] = 9
payload['learning_rate'] = 0.5
payload['decay_rate'] = 1
payload['neighborhood'] = 'bubble'
payload['sigma'] = 2
payload['sigma_decay_rate'] = 1
payload['weights_initialization'] = "random"

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

payload["X"] = X_train.tolist()
payload["y"] = y_train.tolist()
payload["num_iteration"] = 10000
payload["epoch_size"] = len(X)

payload = json.dumps(payload)

# re = requests.post("http://127.0.0.1:1234/api/v1.0/lvq/train", data = payload, headers = headers)
# print(re.text)

payload = {}
payload["model_id"] = "0"
payload["X"] = X_test.tolist()
payload = json.dumps(payload)
re = requests.post("http://127.0.0.1:1234/api/v1.0/lvq/predict", data = payload, headers = headers)
res = re.json()
res = json.loads(res)
y_pred = res.get('y_pred')
y_pred = np.array(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm))