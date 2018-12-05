import numpy as np
import pandas as pd

train_files = ['filtered_RBA-3P.csv', 'filtered_RUBY-1X.csv', 'filtered_RUBY-4X.csv']
test_files = ['filtered_RBA-6P.csv', 'filtered_TN-3X.csv']

train_dataset = None
test_dataset = None
rows = 0
cols = 0

for filename in train_files:
  if train_dataset is None:
    train_dataset = pd.read_csv('data/' + filename).values
  else:
    dataset = pd.read_csv('data/' + filename).values
    train_dataset = np.append(train_dataset, dataset, axis = 0)

for filename in test_files:
  if test_dataset is None:
    test_dataset = pd.read_csv('data/' + filename).values
  else:
    dataset = pd.read_csv('data/' + filename).values
    test_dataset = np.append(test_dataset, dataset, axis = 0)

fieldnames = ['Depth', 'GR', 'LLD', 'NPHI', 'PHIE', 'RHOB', 'VWCL', 'DEPOFACIES']

train_dataset = pd.DataFrame(train_dataset, columns=fieldnames)
train_dataset.iloc[:, -1] = train_dataset.iloc[:, -1].astype(int)
train_dataset.to_csv('data/train_dataset.csv', index=False)

test_dataset = pd.DataFrame(test_dataset, columns=fieldnames)
test_dataset.iloc[:, -1] = test_dataset.iloc[:, -1].astype(int)
test_dataset.to_csv('data/test_dataset.csv', index=False)