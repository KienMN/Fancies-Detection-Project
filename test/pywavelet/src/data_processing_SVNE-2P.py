# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
filepath = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : 7].values
y = dataset.iloc[:, 7].values