import numpy as np
# Dont add any more imports here!

# Make Sure you fill your ID here. Without this ID you will not get a grade!
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 315636753
def ID2():
    '''
        Personal ID of the second student.
    '''
    # Insert your ID here
    return 000000000

def sigmoid(z):
  return 1 / (1 + np.exp(-z))
    
def cross_entropy(t, y):
  return -t * np.log(y) - (1 - t) * np.log(1 - y)


def get_accuracy(y, t):
  acc = 0
  N = 0
  for i in range(len(y)):
    N += 1
    if (y[i] >= 0.5 and t[i] == 1) or (y[i] < 0.5 and t[i] == 0):
      acc += 1
  return acc / N

def pred(w, b, X): 
  """
  Returns the prediction `y` of the target based on the weights `w` and scalar bias `b`.

  Preconditions: np.shape(w) == (90,)
                 type(b) == float
                 np.shape(X) = (N, 90) for some N
  Postconditions: np.shape(y)==(N,)

  >>> pred(np.zeros(90), 1, np.ones([2, 90]))
  array([0.73105858, 0.73105858]) # It's okay if your output differs in the last decimals
  """
  # Your code goes here
  z = np.dot(X, w.T) + b
  y = sigmoid(z)
  return y

def cost(y, t):
  """
  Returns the cost(risk function) `L` of the prediction 'y' and the ground truth 't'.

  - parameter y: prediction
  - parameter t: ground truth
  - return L: cost/risk
  Preconditions: np.shape(y) == (N,) for some N
                 np.shape(t) == (N,)
  
  Postconditions: type(L) == float
  >>> cost(0.5*np.ones(90), np.ones(90))
  0.69314718 # It's okay if your output differs in the last decimals
  """
  # Your code goes here
  N = len(y)
  assert y.shape == t.shape == (N,)
  L = np.sum(cross_entropy(t,y)) / N
  return L

def derivative_cost(X, y, t):
  """
  Returns a tuple containing the gradients dLdw and dLdb.

  Precondition: np.shape(X) == (N, 90) for some N
                np.shape(y) == (N,)
                np.shape(t) == (N,)

  Postcondition: np.shape(dLdw) = (90,)
           type(dLdb) = float
           return dLdw,dldb
  """
  # Your code goes here
  N = len(y)

  assert X.shape == (N, 90)
  assert y.shape == t.shape == (N,)

  error = y - t 

  dLdw = np.dot(X.T, error) / N
  dLdb = np.sum(error) / N
  return (dLdw,dLdb)

def finite_difference_derivative(f, x, h=0.00001):
    return (f(x + h) - f(x)) / h