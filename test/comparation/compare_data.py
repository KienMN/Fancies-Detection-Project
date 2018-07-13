# Comparing the data

# Importing the dataset
from detection.comparation.data_comparator import DataComparator
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import csv

# Importing the arguments
parser = ArgumentParser()
parser.add_argument('--type', '-t', type = int)
parser.add_argument('--window', '-w', type = int)
parser.add_argument('--accept_window', '-aw', type = int)
parser.add_argument('--epsilon', '-e', type = float)
parser.add_argument('--features', '-f', type = str)
parser.add_argument('--train_filename', '-d', type = str)
parser.add_argument('--test_filename', '-s', type = str)
args = parser.parse_args()

train_filename = args.train_filename
test_filename = args.test_filename
train_data_file_path = os.path.dirname(os.getcwd()) + '/data/' + train_filename
test_data_file_path = os.path.dirname(os.getcwd()) + '/data/' + test_filename
type = args.type
features = [int(feature) for feature in args.features.split(',')]
window = args.window
accept_window = args.accept_window
epsilon = args.epsilon

# Importing the dataset
train_dataset = pd.read_csv(train_data_file_path)
X_train = train_dataset.iloc[:, 1: 9].values
X_train = np.append(np.ones((len(X_train), 1)), X_train, axis = 1)
y_train = train_dataset.iloc[:, 9].values
test_dataset = pd.read_csv(test_data_file_path)
X_test = test_dataset.iloc[:, 1: 9].values
X_test = np.append(np.ones((len(X_test), 1)), X_test, axis = 1)
y_test = test_dataset.iloc[:, 9].values

# Training
comparator = DataComparator()
comparator.fit(X_train, y_train)

# Predicting the test set results
y_pred = comparator.predict(X_test, type = type, features = features, epsilon = epsilon,
					window = window, accept_window = accept_window)

# Validating the results
precision, recall = comparator.validate(y_test, y_pred, type)
print(precision, recall)

# Printing to the log file
log_fieldnames = ['Time', 'Test file', 'Train file', 'Type', 'Features', 'Window', 'Accept Window', 'Epsilon', 'Precision', 'Recall']
log_file_path = os.path.dirname(os.getcwd()) + '/log_mean/log1/log_type_' + str(type) + '.csv'
print(log_file_path)
with open(log_file_path, 'a') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames = log_fieldnames)
	log = {'Time': datetime.now(), 'Test file': test_filename, 'Train file': train_filename, 'Type': type, 'Features': features, 'Window': window, 'Accept Window': accept_window, 'Epsilon': epsilon, 'Precision': precision, 'Recall': recall}
	writer.writerow(log)
