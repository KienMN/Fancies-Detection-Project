# Importing the libraries

import pandas as pd
import numpy as np
import os

# Importing the dataset

filepath = os.path.join(os.path.dirname(__file__), 'data/RBA-12P-ST4.csv')
filepath1 = os.path.join(os.path.dirname(__file__), 'data/RBA-12P-ST4_2.csv')
dataset = pd.read_csv(filepath, skiprows=range(1, 2), usecols=range(2, 15))
dataset = dataset.append(pd.read_csv(filepath1, skiprows=range(1, 2), usecols=range(2, 15)), ignore_index = True)
# print(dataset)
dataset = dataset.drop(dataset[dataset.FLAG != 0].index)
for i in range (13):
  dataset = dataset.drop(dataset[dataset.iloc[:, i] < 0].index)

dataset.loc[dataset[dataset.PERM_PRE < 0.001].index, 'PERM_PRE'] = 0.001

fieldnames = ['MD', 'DT', 'GR', 'LLD', 'LLD_D_LLS', 'LLS', 'NPHI', 'PHIE', 'PHIZ', 'RHOB', 'VCLAV', 'PERM_PRE']

X = dataset.iloc[:, [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12]].values
y = dataset.iloc[:, [8]]
new_dataset = pd.DataFrame(np.append(X, y, axis = 1), columns = fieldnames)
# print(new_dataset)
new_dataset.to_csv(os.path.join(os.path.dirname(__file__), 'data/filtered_RBA-12P-ST4.csv'), index = False)