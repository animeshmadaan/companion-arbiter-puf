import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	X_train_mapped = []
	for i in range(len(X_train)):
		X_train_mapped.append(my_map(X_train[i]))
	X_train_mapped = np.array(X_train_mapped)
	
	# # Load testing data
	# test_data = np.genfromtxt('test.dat', delimiter=' ', dtype=None)

	# X_test = []
	# for i in range(len(test_data)):
	# 	X_test.append(my_map(test_data[i][:-1]))
	# X_test = np.array(X_test)

	# y_test = test_data[:, -1]

	# Feature scaling
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train_mapped)
	# X_test_scaled = scaler.transform(X_test)

	C_value = 100

	logistic_regression = LogisticRegression(solver='lbfgs', C=C_value, tol=1e-5)
	
	logistic_regression.fit(X_train_scaled, y_train)
	# y_pred_trn = logistic_regression.predict(X_train_scaled)
	# y_pred = logistic_regression.predict(X_test_scaled)

	# # Evaluating the model
	# accuracy_train = accuracy_score(y_train, y_pred_trn)
	# accuracy_test = accuracy_score(y_test, y_pred)
	# print("Train Accuracy:", 100*accuracy_train)
	# print("Test Accuracy:", 100*accuracy_test)

	w = logistic_regression.coef_[0]
	b = logistic_regression.intercept_

	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	X = np.array(X) 
	  
	if X.ndim == 1:
		d = 1 - 2 * X
		t = np.ones(32)
		t = np.cumprod(d[::-1])[::-1]
		matrix = np.triu(np.outer(t, t),1)
		feat = matrix[np.triu_indices(matrix.shape[0],1)]
		feat = np.concatenate((feat, t))
		del t, d, matrix
    
	else:
        
		for i in range(len(X)):
			d = 1 - 2 * X[i]
			t = np.ones(32)
			t = np.cumprod(d[::-1])[::-1]
			matrix = np.triu(np.outer(t, t),1)
			matrix = matrix[np.triu_indices(matrix.shape[0],1)]
			matrix = np.concatenate((matrix, t))
			if i == 0:
				feat = matrix
			else:
				feat = np.vstack((feat, matrix))
			del t, d, matrix 
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	
	return feat


# # Load training data
# train_data = np.genfromtxt('train.dat', delimiter=' ', dtype=None)
# w, b = my_fit(train_data[:, :-1], train_data[:, -1])