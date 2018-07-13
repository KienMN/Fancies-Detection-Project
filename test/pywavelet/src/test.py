import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Setting the path of datafile
filepath = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/15_1-SD-1X_LQC.csv'

# Field names
# fieldnames = ['WELL', 'MD', 'TVDSS', 'GR', 'NPHI', 'RHOZ', 'DT', 'VCL', 'PHIE', 'Deltaic_Facies']

# Importing the dataset
dataset = pd.read_csv(filepath, skiprows=range(1, 5000))
dataset = dataset.round({'Deltaic_Facies': 0})

X = dataset.iloc[:, 1: 9].values
X = np.append(np.zeros((len(X), 1)), X, axis = 1)
y = dataset.iloc[:, 9].values

# Getting piece of original dataset
feature = 5
ts = X[:, feature]

# Decomposing original data
# ca, cd = pywt.dwt(ts, 'haar')
ca, cd = pywt.dwt(ts, 'db20')

# Performing thresholding in the Wavelet domain
cat = pywt.threshold(ca, np.std(ca), 'hard')
cdt = pywt.threshold(cd, np.std(cd), 'hard')

# Reconstructing the data
# ts_rec = pywt.idwt(cat, cdt, 'haar')
ts_rec = pywt.idwt(cat, cdt, 'db20')

# Examining the length of original and reconstructed data
print(len(ts), len(ts_rec))

# Visualizing the data
fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(ts[1400: 1550], '--r', linewidth = 0.7, label = "Original data")
ax.plot(ts_rec[1400: 1550], 'b', linewidth = 0.7, label = "Denoised data")
ax.set_xlabel("Number of samples")
# ax.set_ylabel(fieldnames[feature])
ax.legend(loc = 1)
plt.grid('on')
plt.show()
# plt.savefig("/Users/kienmaingoc/Desktop/demo")