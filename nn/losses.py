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
        return (y_true - y_pred) / y_true.shape[1]


class BinaryCrossEntropy(Loss):
    def compute(self, y_true, y_pred):
        return -(1.0 / y_true.shape[1]) * (
                np.dot(np.log(y_pred), y_true.T) + np.dot(np.log(1 - y_pred), (1 - y_true).T))

    def derivate(self, y_true, y_pred):
        return (y_true - y_pred) / y_true.shape[1]


class MeanSquaredError(Loss):
    def compute(self, y_true, y_pred):
        pass

    def derivate(self, y_true, y_pred):
        pass


class MeanAbsoluteError(Loss):
    def compute(self, y_true, y_pred):
        pass

    def derivate(self, y_true, y_pred):
        pass
