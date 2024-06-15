import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 315636753
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  X = np.c_[np.ones((X.shape[0], 1)), X]    # Add a column of ones to X for the bias term
  X_transpose= np.transpose(X)
  inverse_matrix = np.linalg.inv(np.dot(X_transpose, X))

  return np.dot(np.dot(inverse_matrix, X_transpose),y)

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  prediction = model.predict(X)
  correct_prediction = np.sum(prediction == s)
  accuracy = (correct_prediction / len(s)) * 100
  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-0.060275098863884574, 0.0164227259556691, -0.02661181214278923, 0.00864438831110523, -0.03969022274428453, -0.018151809502529804, 0.0893324721880855, -0.017020070103715295, 0.04055325941849598, -0.03313583624821699, 0.03538793016988223, 0.007479107125564478, 0.08666236718897556, 0.1582478949262071, 0.7650690151627796, 0.036791879166358746, 0.020924188500499846, 0.009370445239890855, 0.025221747265943523, -0.006431296481186966, 0.05092962098616309, 0.015228656319622634, 0.0025763781832655423, -0.037961188241101985, -0.028388276317477035, 0.043027987720183916, -0.018778655271298576, -0.029390924829043312]


def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return 2.0709601051017953e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.28920864009438985, -0.230811684431569, 0.3526679198263, -0.22996652670020062, -0.34075409918497945, 0.04550602623768541, -0.11659373011469486, -0.026418571226048338, 0.0928379707616084, 0.08738524803151185, -0.5304848954446387, 0.05314669730432251, -0.058999817184825144, 0.9783565995707971, 2.801053190969795, -0.40248136067304047, 0.11078785585568059, -0.061795867641137, -0.18161754201881683, -0.10948467745424026, 0.050801865435508804, 0.07000470478609956, 0.00975930117742559, -0.21020406271341346, -0.3990593042006408, 0.11622081471843333, 0.07535819279871829, 0.25549380988045844]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.63711754]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]