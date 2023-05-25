import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def load(file):
    mat = scipy.io.loadmat(file)
    return mat['Xtrain'], mat['Ytrain'], mat['Xvalid'], mat['Yvalid'], mat['Xtest']


def plot_sample(image_vector, digit):
    image_square = image_vector.reshape(28, 28).T  # transpose the image
    plt.imshow(image_square)
    plt.title("Digit: {}".format(*digit))
    plt.show()
