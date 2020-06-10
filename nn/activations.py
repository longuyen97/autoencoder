import numpy as np
import abc


class ActivationFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def activate(self, x, y=None):
        pass

    @abc.abstractmethod
    def derivate(self, x, y=None):
        pass


class LossFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def derivate(self, y_true, y_pred):
        pass


class CatCrossEntropy(LossFunction):
    def compute(self, y_true, y_pred):
        loss = - np.sum((y_true * np.log(y_pred)), axis=0, keepdims=True)
        cost = np.sum(loss, axis=1) / y_true.shape[1]
        return cost

    def derivate(self, y_true, y_pred):
        pass


class BinaryCrossEntropy(LossFunction):
    def compute(self, y_true, y_pred):
        pass

    def derivate(self, y_true, y_pred):
        pass


class Relu(ActivationFunction):
    def activate(self, x, y=None):
        return np.maximum(0, x)

    def derivate(self, x, y=None):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Softmax(ActivationFunction):

    def activate(self, x, y=None):
        e = np.exp(x)
        return e / np.sum(e, axis=0, keepdims=True)

    def derivate(self, x, y=None):
        pass


class Sigmoid(ActivationFunction):
    def activate(self, x, y=None):
        return 1 / (1 + np.exp(-x))

    def derivate(self, x, y=None):
        pass


class Linear(ActivationFunction):
    def derivate(self, x, y=None):
        pass

    def activate(self, x, y=None):
        return x


def cat_cross_entropy(y_true, y_pred):
    loss = - np.sum((y_true * np.log(y_pred)), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / y_true.shape[1]
    return cost


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
