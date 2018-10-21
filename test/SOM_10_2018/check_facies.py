import pandas as pd
import numpy as np
import os

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RUBY-2X.csv')
dataset = pd.read_csv(filepath)
y = dataset.iloc[:, -1].values
for i in np.unique(y):
  # print(len(np.where(y == i)[0]))
  print("Facy {}: {} samples.".format(i, len(np.where(y == i)[0])))