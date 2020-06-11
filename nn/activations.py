import numpy as np
import abc


class Activation(abc.ABC):
    @abc.abstractmethod
    def activate(self, x, y=None):
        pass

    @abc.abstractmethod
    def derivate(self, x, y=None):
        pass


class LinearActivation(Activation):
    def activate(self, x, y=None):
        return x

    def derivate(self, x, y=None):
        return 1


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
