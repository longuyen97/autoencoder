import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def derivate(self, y_true, y_pred):
        pass


class CategoricalCrossEntropy(Loss):
    def compute(self, y_true, y_pred):
        loss = - np.sum((y_true * np.log(y_pred)), axis=0, keepdims=True)
        cost = np.sum(loss, axis=1) / y_true.shape[1]
        return cost

    def derivate(self, y_true, y_pred):
        """
        This derivation already includes the derivation of softmax
        """
        ret = (y_true - y_pred) / y_true.shape[1]
        return ret


class BinaryCrossEntropy(Loss):
    def compute(self, y_true, y_pred):
        return -(1.0 / y_true.shape[1]) * (
                np.dot(np.log(y_pred), y_true.T) + np.dot(np.log(1 - y_pred), (1 - y_true).T))

    def derivate(self, y_true, y_pred):
        ret = (y_true - y_pred) / y_true.shape[1]
        return ret


class MeanSquaredError(Loss):
    def compute(self, y_true, y_pred, axis=1):
        diff = y_pred - y_true
        squared = diff ** 2
        mean_squared = np.sum(squared) / y_true.shape[1]
        return mean_squared

    def derivate(self, y_true, y_pred):
        ret = -(2 * (y_true - y_pred)) / y_true.shape[1]
        return ret


class MeanAbsoluteError(Loss):
    def __init__(self, axis=1):
        self.axis = axis

    def compute(self, y_true, y_pred):
        return np.mean(np.absolute(y_pred - y_true), axis=self.axis)

    def derivate(self, y_true, y_pred):
        return (y_true - y_pred) / (y_true.shape[self.axis] * np.absolute(y_true - y_pred))
