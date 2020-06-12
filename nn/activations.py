import numpy as np
import abc


class Activation(abc.ABC):
    """
    Abstract class as template for every activation function
    """

    @abc.abstractmethod
    def activate(self, x):
        """
        Activate input

        :param x: mandatory input x
        :return: activated input
        """
        pass

    @abc.abstractmethod
    def derivate(self, x):
        """
        Return the gradient of the function at x

        :param x: x position
        :return: gradients
        """
        pass


class LinearActivation(Activation):
    """
    A linear activation. Input will not be scaled. This activation can be used for regression.
    """

    def activate(self, x):
        return x

    def derivate(self, x):
        return 1


class Relu(Activation):
    """
    A hidden layer's activation function.
    """

    def activate(self, x):
        return np.maximum(0, x)

    def derivate(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class LeakyRelu(Activation):
    """
    A hidden layer's activation function.
    """

    def activate(self, x):
        return np.maximum(0.01 * x, x)

    def derivate(self, x):
        x[x <= 0] = 0.01
        x[x > 0] = 1
        return x


class Tanh(Activation):
    """
    A hidden layer's activation function.
    """

    def activate(self, x):
        return np.tanh(x)

    def derivate(self, x):
        return 1.0 - np.tanh(x) ** 2


class Softmax(Activation):
    """
    Multimodal output scaling function
    """

    def activate(self, x):
        e = np.exp(x)
        return e / np.sum(e, axis=0, keepdims=True)

    def derivate(self, x):
        S = self.activate(x)
        S_vector = S.reshape(S.shape[0], 1)
        S_matrix = np.tile(S_vector, S.shape[0])
        return np.diag(S) - (S_matrix * np.transpose(S_matrix))


class Sigmoid(Activation):
    """
    Binary output scaling function
    """

    def activate(self, x):
        ret = 1 / (1 + np.exp(-x))
        return ret

    def derivate(self, x):
        e = self.activate(x, y)
        return e * (1 - e)


class Linear:
    """
    Linear function with the form y = wx + b
    """
    def activate(self, weights, x, biases):
        return np.dot(weights, x) + biases

    def derivate_weights(self, logit_grad, input_data):
        return np.dot(logit_grad, input_data.T)

    def derivate_biases(self, logit_grad):
        return np.sum(logit_grad, axis=1, keepdims=True)
