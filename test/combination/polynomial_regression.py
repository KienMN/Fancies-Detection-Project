# Testing linear regression model

# Adding path
import os

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
filepath = os.path.join(os.path.dirname(__file__), 'data/filtered_RB-1X.csv')
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(feature_range=(0, 1))
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)

# Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

# Predicting the result
y_pred = regressor.predict(X_test_poly)

# Evaluating the result
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))