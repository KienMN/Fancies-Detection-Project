# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pywt

# Importing the dataset
fieldnames = ['wellName', 'datasetName', 'DEPTH', 'DepoFacies', 'DT', 'Facies', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL']
feature_names = ['DEPTH', 'DT', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL']
filepath = os.path.dirname(os.getcwd()) + '/data/SVNE-2P_SVNE-2P-new.csv'
dataset = pd.read_csv(filepath, skiprows=range(1, 2))
dataset = dataset.round({'DepoFacies': 0})
X = dataset.iloc[:, [2, 4, 6, 7, 8, 9, 10]].values
y = dataset.iloc[:, [3]].values

# Preprocessing
ncols = X.shape[1]

for feature in range(1, ncols):
	# Getting piece of original dataset
	ts = np.copy(X[:, feature])

	# Wavelet denoising
	# Decomposing original data
	ca, cd = pywt.dwt(ts, 'db20')

	# Performing thresholding in the Wavelet domain
	cat = pywt.threshold(ca, np.std(ca), 'hard')
	cdt = pywt.threshold(cd, np.std(cd), 'hard')

	# Reconstructing the data
	ts_rec = pywt.idwt(cat, cdt, 'db20')
	ts_rec[np.where(ts_rec < 0)[0]] = 0

	# Eliminating the last element if reconstructed data has more elements than original data
	if len(ts) < len(ts_rec):
		ts_rec = np.delete(ts_rec, len(ts))

	# Normalizing the data
	max_val = ts_rec.max()
	min_val = ts_rec.min()
	X[:, feature] = (ts_rec - min_val) / (max_val - min_val)

# Creating new dataset from result of preprocessing
feature_names.append('Depo Fancies')
new_dataset = pd.DataFrame(np.append(X, y, axis = 1), columns = feature_names)
new_dataset.to_csv(os.path.dirname(os.path.dirname(os.getcwd())) + '/new_data/processed_SVNE-2P_SVNE-2P-new.csv', index = False)

# Visualizing the data
lines = ['b', 'g', 'r', 'y', 'c', '--b', '--r']
fig, ax = plt.subplots(figsize = (12, 8))
for feature in range (1, ncols):
	ax.plot(X[0: 150, feature], lines[feature], linewidth = 0.7, label = feature_names[feature])
ax.set_xlabel("Number of samples")
ax.legend(loc = 1)
plt.grid('on')
plt.show()