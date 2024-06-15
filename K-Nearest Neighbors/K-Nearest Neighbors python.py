import json
import numpy as np  # check out how to install numpy
from utils import load, plot_sample

# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '315636753'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

sampleNum = 0
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])


# for i in range(len(Xtest)):
#   plot_sample(Xtest[i,:],np.array([1]))
# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.


# < your code here >

# Functions
def mse_distance(x, y):
    return np.mean(np.square(x - y))

def decision(cost_function, k, input):
    # compute the distances between input and each element in Xtrain (list comprehension)
    distances = [cost_function(input, item) for item in Xtrain]
    # find the indices of the k smallest distances
    smallest_indices = np.argsort(distances)[:k]
    # determine the output label by taking the most frequent label among the k nearest neighbors
    output = np.argmax(np.bincount(Ytrain[smallest_indices].flatten()))
    return output


mse_w = []
print("optimizing")

for k in range(1, 10):
    print("currently doing k=" + str(k))  # Personal tracking
    wrong_counter_mse = 0
    for i in range(len(Xvalid)):
        item = Xvalid[i]
        if i % 30 == 0:
            print("currently doing iteration {}/{}".format(i, len(Xvalid))) # run checking
        mse_decision = decision(mse_distance, k, item)
        if mse_decision != Yvalid[i]:
            wrong_counter_mse += 1
    mse_w.append(wrong_counter_mse)

print("mse best case had k = " + str(np.argmin(np.array(mse_w)) + 1))

k_opt = np.argmin(np.array(mse_w)) + 1


# work
Ytest = []
for input in Xtest:
    Ytest.append(decision(mse_distance, k_opt, input))


# save classification results
print('save')
filename = ID + '.txt'
np.savetxt(filename, Ytest, delimiter=',', fmt='%i')
print('done')
