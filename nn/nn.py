import numpy as np
from nn.activations import softmax, relu, cat_cross_entropy
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

    def forward(self, x) -> typing.Tuple[typing.Dict, typing.Dict]:
        """

        :param x:
        :return:
        """
        logits = dict()
        activations = dict()

        logits["Z1"] = np.dot(self.parameters["W1"], x) + self.biases["b1"]
        activations["A1"] = relu(logits["Z1"])

        for i in range(2, self.hiddens):
            logits[f"Z{i}"] = np.dot(self.parameters[f"W{i}"], activations[f"A{i - 1}"]) + self.biases[f"b{i}"]
            activations[f"A{i}"] = relu(logits[f"Z{i}"])

        logits[f"Z{self.hiddens}"] = np.dot(self.parameters[f"W{self.hiddens}"], activations[f"A{self.hiddens - 1}"]) + \
                                     self.biases[f"b{self.hiddens}"]
        activations[f"A{self.hiddens}"] = softmax(logits[f"Z{self.hiddens}"])

        return logits, activations

    def backward(self, logits, activations, x, y, learning_rate=0.001):
        """

        :param logits:
        :param activations:
        :param x:
        :param y:
        :param learning_rate:
        :return:
        """
        cost = cat_cross_entropy(y, activations[f"A{self.hiddens}"])
        activation_grads = dict()
        logit_grads = dict()

        logit_grads[f"dZ{self.hiddens}"] = (activations[f"A{self.hiddens}"] - y) / y.shape[1]
        for i in range(1, self.hiddens):
            activation_grads[f"dA{self.hiddens - i}"] = np.dot(self.parameters[f"W{self.hiddens - i + 1}"].T,
                                                               logit_grads[f"dZ{self.hiddens - i + 1}"])
            logit_grads[f"dZ{self.hiddens - i}"] = activation_grads[f"dA{self.hiddens - i}"] * relu(
                logits[f"Z{self.hiddens - i}"], True)

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

        return cost

    def train(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        logits, activations = self.forward(x)
        return self.backward(logits, activations, x, y)
