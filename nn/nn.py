import numpy as np
from nn.activations import softmax, relu, cat_cross_entropy, ActivationFunction, Relu, LossFunction, CatCrossEntropy, \
    Softmax
import typing


class NeuralNetwork:
    def __init__(self, layers: typing.List):
        """
        Constructor

        :param layers: Define number of parameters each layer should have
        """
        self.layers = layers
        self.parameters = dict()
        self.biases = dict()
        self.depth = len(layers)
        self.hiddens = len(layers) - 1
        for i in range(1, self.depth):
            self.parameters[f"W{i}"] = np.random.randn(layers[i], layers[i - 1]) * (np.sqrt(2 / layers[i - 1]))
            self.biases[f"b{i}"] = np.zeros((layers[i], 1))

    def forward(self, x, hidden_activation_function=Relu, last_activation_function=Softmax) -> typing.Tuple[
        typing.Dict, typing.Dict]:
        """
        Forward propagation

        :param hidden_activation_function:  Activation functions for the hidden layers
        :param last_activation_function:  Activation functions for the last output layer
        :param x: training data
        :return: return logits and activations of every layers in processing order
        """
        logits = dict()
        activations = dict()

        # Logits and activations of the first hidden layer
        logits["Z1"] = np.dot(self.parameters["W1"], x) + self.biases["b1"]
        activations["A1"] = hidden_activation_function.activate(logits["Z1"])

        # Logits and activations of the further hidden layers
        for i in range(2, self.hiddens):
            logits[f"Z{i}"] = np.dot(self.parameters[f"W{i}"], activations[f"A{i - 1}"]) + self.biases[f"b{i}"]
            activations[f"A{i}"] = hidden_activation_function.activate(logits[f"Z{i}"])

        # Logit of the last layer aka. prediction
        logits[f"Z{self.hiddens}"] = np.dot(self.parameters[f"W{self.hiddens}"], activations[f"A{self.hiddens - 1}"]) + \
                                     self.biases[f"b{self.hiddens}"]
        # Scaling prediction
        activations[f"A{self.hiddens}"] = last_activation_function.activate(logits[f"Z{self.hiddens}"])

        return logits, activations

    def backward(self, logits, activations, x, y, hidden_activation_function=Relu, last_activation_function=Softmax,
                 loss_function=CatCrossEntropy, learning_rate=0.001):
        """
        Backward propagation and parameters fitting

        :param hidden_activation_function:
        :param last_activation_function:
        :param loss_function:
        :param logits: logits of every layer from the forward propagation
        :param activations: activations of every layer from the forward propagation
        :param x: training data
        :param y: labels
        :param learning_rate: at which rate should the learning takes place
        :return: cost of this epoch
        """
        activation_grads = dict()
        logit_grads = dict()

        logit_grads[f"dZ{self.hiddens}"] = (activations[f"A{self.hiddens}"] - y) / y.shape[1]
        for i in range(1, self.hiddens):
            activation_grads[f"dA{self.hiddens - i}"] = np.dot(self.parameters[f"W{self.hiddens - i + 1}"].T, logit_grads[f"dZ{self.hiddens - i + 1}"])
            logit_grads[f"dZ{self.hiddens - i}"] = activation_grads[ f"dA{self.hiddens - i}"] * hidden_activation_function.derivate(logits[f"Z{self.hiddens - i}"], True)

        parameter_grads = dict()
        biases_grads = dict()
        parameter_grads["dW1"] = np.dot(logit_grads["dZ1"], x.T)
        biases_grads["db1"] = np.sum(logit_grads["dZ1"], axis=1, keepdims=True)
        for i in range(2, self.depth):
            parameter_grads[f"dW{i}"] = np.dot(logit_grads[f"dZ{i}"], activations[f"A{i - 1}"].T)
            biases_grads[f"db{i}"] = np.sum(logit_grads[f"dZ{i}"], axis=1, keepdims=True)

        for i in range(1, self.depth):
            self.parameters[f"W{i}"] -= (learning_rate * parameter_grads[f"dW{i}"])
            self.biases[f"b{i}"] -= (learning_rate * biases_grads[f"db{i}"])

        return loss_function.compute(y, activations[f"A{self.hiddens}"])

    def train(self, x, y):
        """
        Forward and backward propagation in one round

        :param x: training data
        :param y: labels
        :return: cost of the epoch
        """
        logits, activations = self.forward(x)
        return self.backward(logits, activations, x, y)
