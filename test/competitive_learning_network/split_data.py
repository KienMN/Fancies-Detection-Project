import os
import numpy as np
import pandas as pd

# Importing data
filename = "15_1-SD-6X_LQC.csv"
filepath = os.path.join(os.path.dirname(__file__), 'data/' + filename)
fieldname = ["wellName", "datasetName", "DEPT", "DT", "DEPO_FACIES", "GR", "NPHI", "PHIE", "RHOB", "VCL"]

# wellName,datasetName,DEPT,DT,DEPO_FACIES,GR,NPHI,PHIE,RHOB,VCL
# wellName,datasetName,DEPT,DT,DEPO_FACIES,GR,NPHI,PHIE,RHOB,VCL
# wellName,datasetName,DEPT,DT,DEPO_FACIES,GR,NPHI,PHIE,RHOB,VCL
# wellName,datasetName,DEPT,DT,DEPO_FACIES,GR,NPHI,PHIE,RHOB,VCL

dataset = pd.read_csv(filepath, skiprows = range(1, 2))
dataset.round({"DEPO_FACIES": 0})

X = dataset.iloc[np.where(dataset['DEPO_FACIES'] != -9999)[0], [0, 2, 3, 5, 6, 7, 8, 9]].values
y = dataset.iloc[np.where(dataset['DEPO_FACIES'] != -9999)[0], [4]].values
# print(X)
# print(y)

# Exporting new data
new_filepath = os.path.join(os.path.dirname(__file__), 'data/processed_' + filename)
new_dataset = np.append(X, y, axis = 1)
new_fieldnames = ["wellName", "DEPT", "DT", "GR", "NPHI", "PHIE", "RHOB", "VCL", "DEPO_FACIES"]
new_dataset = pd.DataFrame(new_dataset, columns=new_fieldnames)
new_dataset.to_csv(new_filepath, index=False)