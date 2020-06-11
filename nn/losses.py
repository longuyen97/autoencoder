import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class CategoricalCrossEntropy(Loss):
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


class BinaryCrossEntropy(Loss):
    def compute(self, y_true, y_pred):
        return -(1.0 / y_true.shape[1]) * (
                np.dot(np.log(y_pred), y_true.T) + np.dot(np.log(1 - y_pred), (1 - y_true).T))

    def derivative(self, y_true, y_pred):
        ret = (y_true - y_pred) / y_true.shape[1]
        return ret


class MeanSquaredError(Loss):
    def compute(self, y_true, y_pred, axis=1):
        diff = y_pred - y_true
        squared = diff ** 2
        mean_squared = np.sum(squared) / y_true.shape[1]
        return mean_squared

    def derivative(self, y_true, y_pred):
        ret = -(2 * (y_true - y_pred)) / y_true.shape[1]
        return ret


class MeanAbsoluteError(Loss):
    def compute(self, y_true, y_pred):
        absolute = np.absolute(y_pred - y_true)
        return np.mean(absolute, axis=1)

    def derivative(self, y_true, y_pred):
        return (y_true - y_pred) / (y_true.shape[1] * np.absolute(y_true - y_pred))


class AutoEncoderError(Loss):
    def compute(self, y_true, y_pred):
        diff = np.subtract(y_pred, y_true)
        absolute = np.absolute(diff)
        whole_sum = np.sum(absolute, axis=0)
        ret = np.mean(whole_sum)
        return ret

    def derivative(self, y_true, y_pred):
        import autograd.numpy as np
        from autograd import grad

        def do(y_true, y_pred):
            diff = np.subtract(y_pred, y_true)
            absolute = np.absolute(diff)
            whole_sum = np.sum(absolute, axis=0)
            ret = np.mean(whole_sum)
            return ret

        grad_do = grad(do)
        return grad_do(y_true, y_pred)
