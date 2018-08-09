import requests
from flask import jsonify
import json
import numpy as np
import pandas as pd
import os

server_address = "http://127.0.0.1"
headers = {"content-type": "application/json"}

payload = {}
payload['model_id'] = "model-1"
payload['params'] = {}
payload['dataset'] = {}

payload['params']['size'] = 5
payload['params']['first_num_iteration'] = 2000
payload['params']['second_num_iteration'] = 2000
payload['params']['first_epoch_size'] = 200
payload['params']['second_epoch_size'] = 200
# payload['params']['learning_rate'] = 0.5
# payload['params']['decay_rate'] = 1
# payload['params']['neighborhood'] = 'bubble'
# payload['params']['sigma'] = 1
# payload['params']['sigma_decay_rate'] = 1
# payload['params']['weights_initialization'] = "pca"

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/SD-3X_rocktype.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = X_train.T
X_test = X_test.T

payload['dataset']['features'] = X_train.tolist()
payload['dataset']['target'] = y_train.tolist()
payload = json.dumps(payload)

uri = server_address + ":1234/api/v1/ssom/train"
re = requests.post(uri, data = payload, headers = headers)
print(re.text)

payload = {}
payload['model_id'] = "model-1"
payload['dataset'] = {}
payload['dataset']['features'] = X_test.tolist()
payload = json.dumps(payload)

uri = server_address + ":1234/api/v1/ssom/predict"
re = requests.post(uri, data = payload, headers = headers)
print(re.text)
res = re.json()
res = json.loads(res)

y_pred = res.get('target')

y_pred = np.array(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / np.sum(cm))