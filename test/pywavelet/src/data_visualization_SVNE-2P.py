# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--depo_fancy', '-d', type = int)
parser.add_argument('--feature', '-f', type = int)
args = parser.parse_args()

# Importing the dataset
filepath = os.path.dirname(os.getcwd()) + '/data/processed_SVNE-2P_SVNE-2P-new.csv'
print(filepath)
feature_names = ['DEPTH', 'DT', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL']
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, : 7].values
y = dataset.iloc[:, 7].values

# Visualizing
depo_fancy = args.depo_fancy
feature = args.feature
tmp = np.where(y == depo_fancy)[0]
m = len(tmp)

fig, ax = plt.subplots(figsize=(12, 8))
lines = ['b-', 'g--', 'r-.', 'c:', 'm-', 'y--', 'k-.', 'b:', 'g-', 'r--', 'c-.']

feature_value = []
count = 0
feature_value.append(X[tmp[0], feature])
for i in range (1, m):
	if tmp[i] - tmp[i - 1] != 1:
		# if len(feature_value) > 200:
			# pass
			# ax.plot(feature_value[:200], lines[count % len(lines)])
		# else:
		ax.plot(feature_value, lines[count % len(lines)])
		count += 1
		feature_value = []
	else:
		feature_value.append(X[tmp[i], feature])
	if i == m - 1:
		# if len(feature_value) > 200:
		# 	pass
			# ax.plot(feature_value[:200], lines[count % len(lines)])
		# else:
		ax.plot(feature_value, lines[count % len(lines)])

print("Number of continuous samples", count)

ax.set_xlabel("Number of samples")
ax.set_ylabel(feature_names[feature])
ax.set_title(feature_names[feature] + " of fancy " + str(depo_fancy))
plt.show()
# plt.savefig("/Users/kienmaingoc/Desktop/Fancy_" + str(depo_fancy) + "_" + feature_names[feature])