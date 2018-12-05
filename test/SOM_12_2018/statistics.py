import numpy as np
import pandas as pd

filenames = ['filtered_RBA-3P.csv', 'filtered_RBA-6P.csv', 'filtered_RUBY-1X.csv', 'filtered_RUBY-4X.csv', 'filtered_TN-3X.csv']

for filename in filenames:
  print(filename)
  dataset = pd.read_csv('data/' + filename)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, [-1]].values
  print(len(y))
  print(np.amax(X, axis = 0))
  print(np.amin(X, axis = 0))

  for i in np.unique(y):
    print('class', str(int(i)) + ':', np.where(y == i)[0].shape[0], 'samples')