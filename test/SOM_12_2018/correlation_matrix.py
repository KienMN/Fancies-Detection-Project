import pandas as pd
import numpy as np

dataset = pd.read_csv('data/train_dataset.csv')

corr_matrix = dataset.corr()

print(corr_matrix)

corr2 = np.corrcoef(dataset.values.T)

print(corr2)

print(np.sum(corr2[:, -1]))

# corr_matrix.style.background_gradient()