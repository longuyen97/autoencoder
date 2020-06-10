import numpy as np
import abc


class Activation(abc.ABC):
    @abc.abstractmethod
    def activate(self, x, y=None):
        pass

    @abc.abstractmethod
    def derivate(self, x, y=None):
        pass


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


class LinearActivation(Activation):
    def activate(self, x, y=None):
        pass

    def derivate(self, x, y=None):
        pass


class Relu(Activation):
    def activate(self, x, y=None):
        return np.maximum(0, x)

    def derivate(self, x, y=None):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class LeakyRelu(Activation):
    def activate(self, x, y=None):
        pass

    def derivate(self, x, y=None):
        pass


class Tanh(Activation):
    def activate(self, x, y=None):
        pass

    def derivate(self, x, y=None):
        pass


class Softmax(Activation):
    def activate(self, x, y=None):
        e = np.exp(x)
        return e / np.sum(e, axis=0, keepdims=True)

    def derivate(self, x, y=None):
        SM = y.reshape((-1, 1))
        jac = np.diagflat(y) - np.dot(SM, SM.T)
        return jac


class Sigmoid(Activation):
    def activate(self, x, y=None):
        ret = 1 / (1 + np.exp(-x))
        return ret

    def derivate(self, x, y=None):
        e = self.activate(x, y)
        return e * (1 - e)


class Linear:
    def activate(self, weights, x, biases):
        return np.dot(weights, x) + biases

    def derivate_weights(self, logit_grad, input_data):
        return np.dot(logit_grad, input_data.T)

    def derivate_biases(self, logit_grad):
        return np.sum(logit_grad, axis=1, keepdims=True)
