# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the Training dataset
filepath1 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-1X.csv')
filepath2 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv')
filepath3 = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-3X.csv')
dataset = pd.read_csv(filepath1)
dataset = dataset.append(pd.read_csv(filepath2), ignore_index = True)
dataset = dataset.append(pd.read_csv(filepath3), ignore_index = True)

# Correlation of training dataset
corr_matrix = dataset.corr()
print(corr_matrix)

# X_train = dataset.iloc[:, : -1].values
# y_train = dataset.iloc[:, -1].values.astype(np.int8)

# Importing the Test dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-4X.csv')
test_dataset = pd.read_csv(filepath)

# X_test = test_dataset.iloc[:, : -1].values
# y_test = test_dataset.iloc[:, -1].values.astype(np.int8)

# Correlation of training dataset
corr_matrix = test_dataset.corr()
print(corr_matrix)