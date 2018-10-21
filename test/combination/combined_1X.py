# Importing libraries
import pandas as pd
import numpy as np
import os

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RBA-12P-ST4.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0, shuffle = False)

# Importing training algorithm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Training model 1
# Importing the dataset for model 1
filepath1 = os.path.join(os.path.dirname(__file__), 'data/filtered_RB-1X.csv')
dataset1 = pd.read_csv(filepath1)
X_1 = dataset1.iloc[:, 1: -1].values
y_1 = dataset1.iloc[:, -1].values


# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc_X_1 = MinMaxScaler(feature_range=(0, 1))
X_1 = sc_X_1.fit_transform(X_1)
X_train_1 = sc_X_1.transform(X_train)
X_test_1 = sc_X_1.transform(X_test)

# Fitting polynomial regression
poly_reg_1 = PolynomialFeatures(degree = 3)
X_1 = poly_reg_1.fit_transform(X_1)
X_train_1 = poly_reg_1.transform(X_train_1)
X_test_1 = poly_reg_1.transform(X_test_1)

# Fitting regression to the training test
# from sklearn.tree import DecisionTreeRegressor
regressor1 = LinearRegression()
regressor1.fit(X_1, y_1)

# Predicting the result for next phase
y_train_1 = regressor1.predict(X_train_1).reshape((-1, 1))
y_test_1 = regressor1.predict(X_test_1).reshape((-1, 1))

# Training model 2
# Importing the dataset for model 2
filepath2 = os.path.join(os.path.dirname(__file__), 'data/filtered_RB-3X.csv')
dataset2 = pd.read_csv(filepath2)
X_2 = dataset2.iloc[:, 1: -1].values
y_2 = dataset2.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc_X_2 = MinMaxScaler(feature_range=(0, 1))
X_2 = sc_X_2.fit_transform(X_2)
X_train_2 = sc_X_2.transform(X_train)
X_test_2 = sc_X_2.transform(X_test)

# Fitting polynomial regression
poly_reg_2 = PolynomialFeatures(degree = 3)
X_2 = poly_reg_2.fit_transform(X_2)
X_train_2 = poly_reg_2.transform(X_train_2)
X_test_2 = poly_reg_2.transform(X_test_2)

# Fitting regression to the training test
# from sklearn.tree import DecisionTreeRegressor
regressor2 = LinearRegression()
regressor2.fit(X_2, y_2)

# Predicting the result for next phase
y_train_2 = regressor2.predict(X_train_2).reshape((-1, 1))
y_test_2 = regressor2.predict(X_test_2).reshape((-1, 1))

# Training model 3
# Importing the dataset for model 3
filepath3 = os.path.join(os.path.dirname(__file__), 'data/filtered_RB-4X.csv')
dataset3 = pd.read_csv(filepath3)
X_3 = dataset3.iloc[:, 1: -1].values
y_3 = dataset3.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc_X_3 = MinMaxScaler(feature_range=(0, 1))
X_3 = sc_X_3.fit_transform(X_3)
X_train_3 = sc_X_3.transform(X_train)
X_test_3 = sc_X_3.transform(X_test)

# Fitting polynomial regression
poly_reg_3 = PolynomialFeatures(degree = 3)
X_3 = poly_reg_3.fit_transform(X_3)
X_train_3 = poly_reg_3.transform(X_train_3)
X_test_3 = poly_reg_3.transform(X_test_3)

# Fitting regression to the training test
# from sklearn.tree import DecisionTreeRegressor
regressor3 = LinearRegression()
regressor3.fit(X_3, y_3)

# Predicting the result for next phase
y_train_3 = regressor3.predict(X_train_3).reshape((-1, 1))
y_test_3 = regressor3.predict(X_test_3).reshape((-1, 1))

# Training model 4
# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc_X_4 = MinMaxScaler(feature_range=(0, 1))
X_train_4 = sc_X_4.fit_transform(X_train)
X_test_4 = sc_X_4.transform(X_test)

# Fitting polynomial regression
poly_reg_4 = PolynomialFeatures(degree = 3)
X_train_4 = poly_reg_4.fit_transform(X_train_4)
X_test_4 = poly_reg_4.transform(X_test_4)

# from sklearn.tree import DecisionTreeRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X_train_4, y_train)

# Prediciting the result for next phase
y_train_4 = regressor.predict(X_train_4).reshape((-1, 1))
y_test_4 = regressor.predict(X_test_4).reshape((-1, 1))

# Preparing dataset for combined model
y_pred_level_1 = np.append(y_train_1, y_train_2, axis = 1)
y_pred_level_1 = np.append(y_pred_level_1, y_train_3, axis = 1)
y_pred_level_1 = np.append(y_pred_level_1, y_train_4, axis = 1)

y_test_level_1 = np.append(y_test_1, y_test_2, axis = 1)
y_test_level_1 = np.append(y_test_level_1, y_test_3, axis = 1)
y_test_level_1 = np.append(y_test_level_1, y_test_4, axis = 1)

# Training combined model
# from sklearn.tree import DecisionTreeRegressor
meta_learner = LinearRegression()
meta_learner.fit(y_pred_level_1, y_train)

# Predicting the result
y_pred = meta_learner.predict(y_test_level_1)

# Evaluating the result
from sklearn.metrics import r2_score, mean_squared_error
print("Single model:")
print(r2_score(y_test, y_test_4))
print(mean_squared_error(y_test, y_test_4))
print("Combined model:")
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))