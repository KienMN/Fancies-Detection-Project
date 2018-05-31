import numpy as np
import pandas as pd

def distance_matrices(matrix1, matrix2):
	"""Computing the distance between 2 matrices"""
	matrix1 = np.array(matrix1, dtype = np.float32)
	matrix2 = np.array(matrix2, dtype = np.float32)
	return np.sum(np.abs(matrix1 - matrix2))

def balance_range(matrix1, matrix2):
	"""Making the mean of each rows of each matrices equal"""
	matrix1 = np.array(matrix1, dtype = np.float32)
	matrix2 = np.array(matrix2, dtype = np.float32)
	avg_matrix1 = np.mean(matrix1, axis = 0)
	avg_matrix2 = np.mean(matrix2, axis = 0)
	_, n = matrix1.shape
	for j in range(n):
		tmp = avg_matrix1[j] - avg_matrix2[j]
		if tmp > 0:
			matrix2[:, j] = matrix2[:, j] + tmp
		elif tmp < 0:
			matrix1[:, j] = matrix1[:, j] - tmp
	return (matrix1, matrix2)

class DataComparator():
	"""This class is used for creating a comparator"""
	X_train = np.array([])
	splits = {}

	# def __init__(self, X_test, X_train):
	# 	self.X_test = np.array(X_test, dtype = np.float32)
	# 	self.X_train = np.array(X_train, dtype = np.float32)

	def fit(self, X_train, y_train):
		self.X_train = np.array(X_train, dtype = np.float32)
		for i, j in enumerate(np.unique(y_train)):
			tmp = np.where(y_train == j)[0]
			# print(tmp)
			start = tmp[0]
			m = len(tmp)
			for k in range(1, m):
				if (tmp[k] - tmp[k - 1] != 1):
					end = tmp[k - 1] + 1
					if self.splits.get(j) == None:
						self.splits[j] = [(start, end)]
					else:
						self.splits.get(j).append((start, end))
					start = tmp[k]
				if k == m - 1:
					if self.splits.get(j) == None:
						self.splits[j] = [(start, tmp[k] + 1)]
					else:
						self.splits.get(j).append((start, tmp[k] + 1))
		
		# Optimizing type = 5
		type_5_groups = self.splits.get(5)
		n = len(type_5_groups)
		for i in range (n - 1, -1, -1):
			group = type_5_groups[i]
			if group[1] - group[0] > 100 or group[1] - group[0] < 15:
				type_5_groups.remove(group)

	def predict(self, X_test, type, features, epsilon, window, accept_window):
		X_test = np.array(X_test, dtype = np.float32)
		# Number of samples
		m = len(X_test)
		y_pred = np.zeros((m, 1))
		groups = self.splits.get(type)
		# print(groups)
		if groups == None:
			print("We don't have this type of data in the training set")
		else:
			start = 0
			count = 0
			while start <= m - window:
				X_test_window = X_test[start: start + window, features]
				for train_start, train_end in groups:
					count += 1
					# print(train_start, train_end)
					for i in range(train_start, train_end - window + 1):
						# tmp1, tmp2 = balance_range(X_test_window, self.X_train[i: i + window, features])
						# print(tmp1, tmp2)

						tmp1 = X_test_window
						tmp2 = self.X_train[i: i + window, features]
						# print(tmp1, tmp2)
						if distance_matrices(tmp1, tmp2) <= epsilon:
							y_pred[start: start + accept_window] = type
							start += accept_window - 1
							break
					if y_pred[start] == type:
						break
				start += 1
			
			# Handle the leftover of test set (number of left samples < window)
			if start < m:
				window = m - start
				accept_window = window
				X_test_window = X_test[start: start + window, features]
				for train_start, train_end in groups:
					count += 1
					for i in range(train_start, train_end - window + 1):
						# tmp1, tmp2 = balance_range(X_test_window, self.X_train[i: i + window, features])

						tmp1 = X_test_window
						tmp2 = self.X_train[i: i + window, features]
						if distance_matrices(tmp1, tmp2) <= epsilon:
							y_pred[start: start + accept_window] = type
							start += accept_window - 1
							break
					if y_pred[start] == type:
						break

			# print("count", count)
		return(y_pred)

	def validate(self, y_true, y_pred, type):
		m = len(y_true)
		tp = 0
		fp = 0
		fn = 0
		p = 0
		r = 0
		for i in range (m):
			if y_true[i] == type and y_pred[i] == type:
				tp += 1
			elif y_true[i] != type and y_pred[i] == type:
				# print(i)
				fp += 1
			elif y_true[i] == type and y_pred[i] != type:
				# print(i)
				fn += 1
		try:
			p = tp / (tp + fp)
			r = tp / (tp + fn)
			print(tp, fp, fn)
		except:
			pass
		return (p, r)

	# Filtering already known type out of the dataset
	def filter_data_by_type(self, X, y, types):
		tmp = np.where(np.isin(y, types, invert = True))[0]
		start = tmp[0]
		filter_splits = []
		m = len(tmp)
		# print(m)
		for k in range(1, m):
			if (tmp[k] - tmp[k - 1] != 1):
				end = tmp[k - 1] + 1
				if len(filter_splits) == 0:
					filter_splits = [(start, end)]
				else:
					filter_splits.append((start, end))
				start = tmp[k]
			if k == m - 1:
				if len(filter_splits) == 0:
					filter_splits = [(start, tmp[k] + 1)]
				else:
					filter_splits.append((start, tmp[k] + 1))
		return (filter_splits)
