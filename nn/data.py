import mnist
import numpy as np


def generate_data():
    return mnist.train_images(), mnist.train_labels(), mnist.test_images(), mnist.test_labels()


def one_hot_encode(y, classes=10, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not classes:
        classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def generate_categorical_data():
    X, Y, x, y = generate_data()
    Y = Y.reshape(Y.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    Y = one_hot_encode(Y).T
    y = one_hot_encode(y).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    x = x.reshape(x.shape[0], -1).T
    x = x / 255.
    return X, Y, x, y


def generate_binary_data():
    X, Y, x, y = generate_data()
    Y_ones = Y[Y == 1]
    Y_zeros = Y[Y == 0]
    X_ones = X[Y == 1]
    X_zeros = X[Y == 0]
    X = np.concatenate([X_ones, X_zeros])
    Y = np.concatenate([Y_ones, Y_zeros])
    Y[Y == 1] = 1
    Y[Y != 1] = 0
    Y = Y.reshape(Y.shape[0], 1).T
    y = y.reshape(y.shape[0], 1).T
    X = X.reshape(X.shape[0], -1).T
    x = x.reshape(X.shape[0], -1).T
    X = X / 255.
    x = x / 255.
    return X, Y, x, y
