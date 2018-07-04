# Importing the libraries
import numpy as np
import pandas as pd
import os
import pickle

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/Data_PETREL_INPUT.csv'
dataset = pd.read_csv(filepath)
dataset = dataset.loc[dataset['wellName'] == 'A10']
# print(dataset)
X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the LVQ model
from lvq_network import LvqNetworkWithNeighborhood
lvq = LvqNetworkWithNeighborhood(n_feature = 2, n_rows = 20, n_cols = 20, n_class = 4,
                                learning_rate = 0.8, decay_rate = 1,
                                sigma = 2, sigma_decay_rate = 1,
                                neighborhood = "bubble")
lvq.sample_weights_init(X_train)
# lvq.pca_weights_init(X_train)
lvq.train_batch(X_train, y_train, num_iteration = 500, epoch_size = len(X_train))

# Saving the model
from sklearn.externals import joblib
filename = "finalized_model.sav"
joblib.dump(lvq, filename)
# pickle.dump(lvq, open(filename, 'wb'))