import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_name = 'RUBY-1X'
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_' + dataset_name + '.csv')
dataset = pd.read_csv(filepath)

id_x = 4
id_y = 5

X = dataset.iloc[:, [id_x, id_y]].values
y = dataset.iloc[:, -1].values.astype(np.int8)

fieldnames = ['Depth', 'DT', 'GR', 'LLD', 'NPHI', 'RHOB']
colors = ['c', 'r', 'g', 'b', 'y']

plt.figure(figsize=(12, 8))

for i in np.unique(y):
  plt.scatter(X[np.where(y == i)[0], 0], X[np.where(y == i)[0], 1], c = colors[i], s = 4, label = 'class {}'.format(i))

plt.xlabel(fieldnames[id_x])
plt.ylabel(fieldnames[id_y])
plt.legend()
plt.title('Dataset {}: {} and {}'.format(dataset_name, fieldnames[id_x], fieldnames[id_y]))
figpath = os.path.join(os.path.dirname(__file__), 'images/{}.jpg'.format(dataset_name))
plt.savefig(figpath)