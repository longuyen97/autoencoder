import numpy as np
import abc
from autograd import grad
import autograd.numpy as dnp


class Loss(abc.ABC):
    """
    Abstract class as template for every loss function
    """
    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class CrossEntropy(Loss):
    """
    Categorical cross entropy with multiple classes
    """
    def compute(self, y_true, y_pred):
        loss = - np.sum((y_true * np.log(y_pred)), axis=0, keepdims=True)
        cost = np.sum(loss, axis=1) / y_true.shape[1]
        return cost

    def derivative(self, y_true, y_pred):
        """
        This derivation already includes the derivation of softmax
        """
        ret = (y_true - y_pred) / y_true.shape[1]
        return ret


class MeanSquaredError(Loss):
    """
    Mean squared error for regression problem
    """
    def compute(self, y_true, y_pred, axis=1):
        diff = y_pred - y_true
        squared = diff ** 2
        whole_sum = np.sum(squared, axis=0)
        return np.mean(whole_sum)

    def derivative(self, y_true, y_pred):
        def do(_y_true, _y_pred):
            diff = _y_pred - _y_true
            squared = diff ** 2
            whole_sum = dnp.sum(squared, axis=0)
            return dnp.mean(whole_sum)
        return -grad(do, 1)(y_true, y_pred)


class MeanAbsoluteError(Loss):
    """
    Mean absolute error for regression problem
    """
    def compute(self, y_true, y_pred):
        diff = np.subtract(y_pred, y_true)
        absolute = np.absolute(diff)
        whole_sum = np.sum(absolute, axis=0)
        ret = np.mean(whole_sum)
        return ret

    def derivative(self, y_true, y_pred):
        def do(_y_true, _y_pred):
            diff = dnp.subtract(_y_pred, _y_true)
            absolute = dnp.absolute(diff)
            whole_sum = dnp.sum(absolute, axis=0)
            return dnp.mean(whole_sum)
        return -grad(do, 1)(y_true, y_pred)
