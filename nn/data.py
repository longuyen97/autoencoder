from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def generate_categorical_data():
    (X, Y), (_, _) = mnist.load_data()
    Y = Y.reshape(Y.shape[0], 1)
    Y = to_categorical(Y, num_classes=10).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y


def generate_binary_data():
    (X, Y), (_, _) = mnist.load_data()
    Y_ones = Y[Y == 1]
    Y_twos = Y[Y == 2]
    X_ones = X[Y == 1]
    X_twos = X[Y == 2]
    X = np.concatenate([X_ones, X_twos])
    Y = np.concatenate([Y_ones, Y_twos])
    Y[Y == 1] = 1
    Y[Y != 1] = 0
    Y = Y.reshape(Y.shape[0], 1).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y
