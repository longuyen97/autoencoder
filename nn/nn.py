import typing

import numpy as np

from nn.activations import ActivationFunction, LossFunction
from nn.activations import Linear


class NeuralNetwork:
    def __init__(self, layers: typing.List, activation: ActivationFunction, scale: ActivationFunction, loss: LossFunction):
        """
        Constructor

        :param layers: Define number of parameters each layer should have
        """
        self.activation = activation
        self.scale = scale
        self.loss = loss
        self.layers = layers
        self.parameters = dict()
        self.biases = dict()
        self.depth = len(layers)
        self.hiddens = len(layers) - 1
        for i in range(1, self.depth):
            self.parameters[f"W{i}"] = np.random.randn(layers[i], layers[i - 1]) * (np.sqrt(2 / layers[i - 1]))
            self.biases[f"b{i}"] = np.zeros((layers[i], 1))

    def forward(self, x, activation: ActivationFunction, scale: ActivationFunction) -> typing.Tuple[
        typing.Dict, typing.Dict]:
        """
        Forward propagation

        :param activation:  Activation functions for the hidden layers
        :param scale:  Activation functions for the last output layer
        :param x: training data
        :return: return logits and activations of every layers in processing order
        """
        logits = dict()
        activations = dict()
        linear = Linear()

        # Logits and activations of the first hidden layer
        logits["Z1"] = linear.activate(self.parameters["W1"], x, self.biases["b1"])
        activations["A1"] = activation.activate(logits["Z1"])

        # Logits and activations of the further hidden layers
        for i in range(2, self.hiddens):
            logits[f"Z{i}"] = linear.activate(self.parameters[f"W{i}"],  activations[f"A{i - 1}"], self.biases[f"b{i}"])
            activations[f"A{i}"] = activation.activate(logits[f"Z{i}"])

        # Logit of the last layer aka. prediction
        logits[f"Z{self.hiddens}"] = linear.activate(self.parameters[f"W{self.hiddens}"], activations[f"A{self.hiddens - 1}"], self.biases[f"b{self.hiddens}"])
        # Scaling prediction
        activations[f"A{self.hiddens}"] = scale.activate(logits[f"Z{self.hiddens}"])

        return logits, activations

    def backward(self, logits, activations, x, y, activation: ActivationFunction, loss: LossFunction, learning_rate=0.001):
        """
        Backward propagation and parameters fitting

        :param activation:
        :param scale:
        :param loss:
        :param logits: logits of every layer from the forward propagation
        :param activations: activations of every layer from the forward propagation
        :param x: training data
        :param y: labels
        :param learning_rate: at which rate should the learning takes place
        :return: cost of this epoch
        """
        activation_grads = dict()
        logit_grads = dict()
        linear = Linear()

        # gradients of the last output
        logit_grads[f"dZ{self.hiddens}"] = loss.derivate(activations[f"A{self.hiddens}"], y)

        for i in range(1, self.hiddens):
            # gradient of the hidden layers' activations
            activation_grads[f"dA{self.hiddens - i}"] = np.dot(self.parameters[f"W{self.hiddens - i + 1}"].T, logit_grads[f"dZ{self.hiddens - i + 1}"])

            # gradient of the hidden layer's output
            logit_grads[f"dZ{self.hiddens - i}"] = activation_grads[f"dA{self.hiddens - i}"] * activation.derivate(logits[f"Z{self.hiddens - i}"])

        parameter_grads = dict()
        biases_grads = dict()
        parameter_grads["dW1"] = linear.derivate_weights(logit_grads["dZ1"], x)
        biases_grads["db1"] = linear.derivate_biases(logit_grads["dZ1"])
        for i in range(2, self.depth):
            parameter_grads[f"dW{i}"] = linear.derivate_weights(logit_grads[f"dZ{i}"], activations[f"A{i - 1}"])
            biases_grads[f"db{i}"] = linear.derivate_biases(logit_grads[f"dZ{i}"])

        for i in range(1, self.depth):
            self.parameters[f"W{i}"] -= (learning_rate * parameter_grads[f"dW{i}"])
            self.biases[f"b{i}"] -= (learning_rate * biases_grads[f"db{i}"])

        return loss.compute(y, activations[f"A{self.hiddens}"])

    def train(self, x, y):
        """
        Forward and backward propagation in one round

        :param scale:
        :param activation:
        :param loss:
        :param x: training data
        :param y: labels
        :return: cost of the epoch
        """
        logits, activations = self.forward(x, self.activation, self.scale)
        return self.backward(logits, activations, x, y, self.activation, self.loss)
