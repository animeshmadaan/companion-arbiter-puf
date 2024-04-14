import numpy as np
from sklearn import linear_model, metrics
from sklearn.linear_model import SGDClassifier
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	X_train_mapped = np.array([my_map(X_train[i]) for i in range(len(X_train))])
	y_train = np.array(y_train)
	y_train = 2*y_train - 1

	if isinstance(X_train_mapped, np.ndarray) and isinstance(y_train, np.ndarray):
		if X_train_mapped.ndim == 1:
			X_train_mapped = X_train_mapped.reshape(-1, 1)
		if y_train.ndim == 1:
			y_train = y_train.reshape(-1, 1)
	
	if X_train_mapped.ndim == 1:
		X_train_mapped = X_train_mapped.reshape(-1, 1)

	if y_train.ndim == 1:
		y_train = y_train.reshape(-1, 1)
		
	svm_classifier = SVC(kernel='linear', C=1.0)
	svm_classifier.fit(X_train_mapped, y_train.ravel())
	
	w = svm_classifier.coef_[0]
	b = svm_classifier.intercept_

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
