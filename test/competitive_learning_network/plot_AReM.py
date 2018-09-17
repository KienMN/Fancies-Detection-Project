# Plot AReM dataset

# Adding path to libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath = os.path.join(os.path.dirname(__file__), 'data/AReM_train_dataset.csv')
dataset = pd.read_csv(filepath).values

x = []
y = []
label = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
legend = {-1}
count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

for i in range (len(dataset)):
  if i == len(dataset) - 1:
    label.append(dataset[i, -1])
    x.append(dataset[i, 0])
    y.append(dataset[i, 4])
    l = int(np.unique(label)[0])
    count[l] += 1
    if l == 4 or l == 5:
      plt.plot(x[0: 50], y[0: 50], label = l if l not in legend else "", c = colors[l])
  if dataset[i, 0] == 0:
    if len(x) != 0 and len(y) != 0:
      l = int(np.unique(label)[0])
      count[l] += 1
      if l == 4 or l == 5:
        plt.plot(x[0: 50], y[0: 50], label = l if l not in legend else "", c = colors[l])
      legend.add(l)
      label = []
      x = []
      y = []
    label.append(dataset[i, -1])
    x.append(dataset[i, 0])
    y.append(dataset[i, 4])
  else:
    label.append(dataset[i, -1])
    x.append(dataset[i, 0])
    y.append(dataset[i, 4])
print(count)
plt.legend()
plt.show()