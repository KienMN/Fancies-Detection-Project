# Data preparation

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importing the dataset
filename = 'SVNE-2P_SVNE-2P-new.csv'
file_directory = os.path.dirname(os.getcwd()) + '/data/'
filepath = file_directory + filename
fieldnames = ['wellName', 'datasetName', 'DEPTH', 'DepoFacies', 'DT', 'Facies', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL']
dataset = pd.read_csv(filepath, skiprows=range(1, 2))
dataset.round({'DepoFacies': 0})
X = dataset.iloc[:, [2, 4, 6, 7, 8, 9, 10]].values
y = dataset.iloc[:, [3]].values

# Transforming the dataset
new_dataset = np.append(X, y, axis = 1)
new_fieldnames = ['DEPTH', 'DT', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL', 'DepoFacies']
new_dataset = pd.DataFrame(new_dataset, columns=new_fieldnames)
new_dataset.to_csv(file_directory + 'processed_' + filename, index=False)