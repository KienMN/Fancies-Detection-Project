# LVQ with Neighborhood Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the LVQ
from lvq_network import LvqNetworkWithNeighborhood
lvq = LvqNetworkWithNeighborhood(n_feature = 6, n_rows = 9, n_cols = 9, n_class = 3,
                                learning_rate = 0.5, decay_rate = 1,
                                sigma = 2, sigma_decay_rate = 1,
                                neighborhood="bubble")
lvq.sample_weights_init(X_train)
# lvq.pca_weights_init(X_train)
lvq.train_batch(X_train, y_train, num_iteration = 10000, epoch_size = len(X_train))

# Predict the result
y_pred = lvq.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print((cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm))

lvq.details()

# Visualization
n_subclass = lvq._n_subclass
n_class = lvq._n_class
n_rows = lvq._n_rows_subclass
n_cols = lvq._n_cols_subclass
n_feature = lvq._n_feature

meshgrid = np.zeros((n_rows, n_cols))

for idx in range (n_subclass):
  i = n_rows - 1 - (idx // n_cols)
  j = idx % n_cols
  for c in range (n_class):
    if lvq._linear_layer_weights[c][idx] == 1:
      meshgrid[i][j] = c
      break

from matplotlib import pyplot as plt
fig = plt.figure(figsize = (8, 8))

ax = fig.add_axes([0, 0, 1, 1])
ax.pcolormesh(meshgrid, edgecolors = 'black', linewidth = 0.1)
ax.set_yticklabels([])
ax.set_xticklabels([])

for idx in range (n_subclass):
  i = n_rows - 1 - (idx // n_cols)
  j = idx % n_cols
  cell_width = 1 / n_rows
  cell_height = 1 / n_cols
  ax = fig.add_axes([i / n_rows + cell_width * 0.1, j / n_cols + cell_height * 0.1, cell_width * 0.8, cell_height * 0.8], polar=True)
  ax.grid(False)
  theta = np.array([])
  width = np.array([])
  for i in range (n_feature):
    theta = np.append(theta, i * 2 * np.pi / n_feature)
    width = np.append(width, 2 * np.pi / n_feature)
  radii = lvq._competitive_layer_weights[idx]
  color = ['black', 'red', 'green', 'blue', 'cyan', 'yellow']
  
  bars = ax.bar(theta, radii, width=width, bottom=0.0)

  for i in range (n_feature):
    bars[i].set_facecolor(color[i])
    bars[i].set_alpha(1)
  ax.set_xticklabels([])
  ax.set_yticklabels([])

plt.savefig('/Users/kienmaingoc/Desktop/lvq_test.png')
# plt.show()