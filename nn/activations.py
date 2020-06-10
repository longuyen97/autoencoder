import numpy as np
import abc


class ActivationFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def activate(self, x):
        pass

    def derivate(self, x):
        pass


def cat_cross_entropy(y_true, y_pred):
    loss = - np.sum((y_true * np.log(y_pred)), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / y_true.shape[1]
    return cost


def binary_cross_entropy(y_true, y_pred):
    pass


def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=0, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
