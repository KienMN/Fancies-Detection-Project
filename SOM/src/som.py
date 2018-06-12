# SOM Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 15, y = 15, input_len = 7, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(data = X)
som.train_random(data = X, num_iteration = 1)

# Visualizing the result
from pylab import bone, plot, show, pcolor, colorbar
bone()
pcolor(som.distance_map().T)
colorbar()
for i, x in enumerate(X):
  w = som.winner(x)
  markers = ['o', 's', 'D']
  colors = ['r', 'g', 'b']
  plot(
    w[0] + 0.5,
    w[1] + 0.5,
    markers[y[i]],
    markeredgecolor = colors[y[i]],
    markerfacecolor = 'None',
    markersize = 10, markeredgewidth = 2
  )
print('Number of type 0:', len(np.where(y == 0)[0]))
print('Number of type 1:', len(np.where(y == 1)[0]))
print('Number of type 2:', len(np.where(y == 2)[0]))
print(som._weights.shape)
show()
# plt.savefig('/Users/kienmaingoc/Desktop/som_test.png')