# Importing the libraries

import pandas as pd
import numpy as np
import os

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/RUBY-2X_RUBY-2X.csv')
dataset = pd.read_csv(filepath, skiprows=range(1, 2))

# print(dataset)
dataset = dataset.drop(dataset[dataset.BH_FLAG != 0].index)
dataset = dataset.drop(dataset[dataset.FACIES == 5].index)
for i in range (8):
  dataset = dataset.drop(dataset[dataset.iloc[:, i] < 0].index)

fieldnames = ['Depth', 'DT', 'GR', 'LLD', 'NPHI', 'RHOB', 'FACIES']

X = dataset.iloc[:, [0, 2, 4, 5, 6, 7]].values
y = dataset.iloc[:, [3]].values
new_dataset = pd.DataFrame(np.append(X, y, axis = 1), columns = fieldnames)
new_dataset.iloc[:, -1] = new_dataset.iloc[:, -1].astype(int)

print(len(new_dataset))
new_dataset.to_csv(os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv'), index = False)