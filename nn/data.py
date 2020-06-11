import mnist
import numpy as np
import math


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


def generate_regression_data():
    x = np.arange(-10, 10, 0.01)
    y = np.sin(2 * math.pi * x)
    m = int(x.shape[0] * 0.8)
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:m], indices[m:]
    X, x = x[training_idx], x[test_idx]
    Y, y = y[training_idx], y[test_idx]
    x = x.reshape((1, x.shape[0]))
    y = y.reshape((1, y.shape[0]))
    X = X.reshape((1, X.shape[0]))
    Y = X.reshape((1, Y.shape[0]))
    return X, Y, x, y


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
    y_ones = y[y == 1]
    y_zeros = y[y == 0]
    x_ones = x[y == 1]
    x_zeros = x[y == 0]
    X = np.concatenate([X_ones, X_zeros])
    Y = np.concatenate([Y_ones, Y_zeros])
    x = np.concatenate([x_ones, x_zeros])
    y = np.concatenate([y_ones, y_zeros])
    Y[Y == 1] = 1
    Y[Y != 1] = 0
    y[y == 1] = 1
    y[y != 1] = 0
    Y = Y.reshape(Y.shape[0], 1).T
    y = y.reshape(y.shape[0], 1).T
    X = X.reshape(X.shape[0], -1).T
    x = x.reshape(x.shape[0], -1).T
    X = X / 255.
    x = x / 255.
    return X, Y, x, y
