import typing
import numpy as np
from nn.activations import Activation, Linear
from nn.losses import Loss
import random
from nn.optimizers import Optimizer


class NeuralNetwork:
    def __init__(self, layers: typing.List, activation: Activation, scale: Activation, loss: Loss, optimizer: Optimizer):
        """
        Constructor

        :param layers: Define number of parameters each layer should have
        """
        self.linear = Linear()
        self.activation = activation
        self.scale = scale
        self.loss = loss
        self.layers = layers
        self.parameters = dict()
        self.biases = dict()
        self.depth = len(layers)
        self.hiddens = len(layers) - 1
        self.optimizer = optimizer
        for i in range(1, self.depth):
            self.parameters[f"W{i}"] = np.random.randn(layers[i], layers[i - 1]) * (np.sqrt(2 / layers[i - 1]))
            self.biases[f"b{i}"] = np.zeros((layers[i], 1))

    def forward(self, x) -> typing.Tuple[typing.Dict, typing.Dict]:
        """
        Forward propagation

        :param x: training data
        :return: return logits and activations of every layers in processing order
        """
        logits = dict()
        activations = dict()

        # Logits and activations of the first hidden layer
        logits["Z1"] = self.linear.activate(self.parameters["W1"], x, self.biases["b1"])
        activations["A1"] = self.activation.activate(logits["Z1"])

        # Logits and activations of the further hidden layers
        for i in range(2, self.hiddens):
            logits[f"Z{i}"] = self.linear.activate(self.parameters[f"W{i}"], activations[f"A{i - 1}"],
                                                   self.biases[f"b{i}"])
            activations[f"A{i}"] = self.activation.activate(logits[f"Z{i}"])

        # Logit of the last layer aka. prediction
        logits[f"Z{self.hiddens}"] = self.linear.activate(self.parameters[f"W{self.hiddens}"],
                                                          activations[f"A{self.hiddens - 1}"],
                                                          self.biases[f"b{self.hiddens}"])
        # Scaling prediction
        activations[f"A{self.hiddens}"] = self.scale.activate(logits[f"Z{self.hiddens}"])

        return logits, activations

    def backward(self, logits, activations, x, y):
        """
        Backward propagation and parameters fitting

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
        logit_grads[f"dZ{self.hiddens}"] = self.loss.derivative(activations[f"A{self.hiddens}"], y)

        for i in range(1, self.hiddens):
            # gradient of the hidden layers' activations
            a_grad = np.dot(self.parameters[f"W{self.hiddens - i + 1}"].T, logit_grads[f"dZ{self.hiddens - i + 1}"])
            activation_grads[f"dA{self.hiddens - i}"] = a_grad

            # gradient of the hidden layer's output
            log_grad = activation_grads[f"dA{self.hiddens - i}"] * self.activation.derivate(
                logits[f"Z{self.hiddens - i}"])
            logit_grads[f"dZ{self.hiddens - i}"] = log_grad

        parameter_grads = dict()
        biases_grads = dict()
        parameter_grads["dW1"] = linear.derivate_weights(logit_grads["dZ1"], x)
        biases_grads["db1"] = linear.derivate_biases(logit_grads["dZ1"])
        for i in range(2, self.depth):
            parameter_grads[f"dW{i}"] = linear.derivate_weights(logit_grads[f"dZ{i}"], activations[f"A{i - 1}"])
            biases_grads[f"db{i}"] = linear.derivate_biases(logit_grads[f"dZ{i}"])

        loss = self.loss.compute(y, activations[f"A{self.hiddens}"])
        for i in range(1, self.depth):
            self.parameters[f"W{i}"] = self.optimizer.compute(self.parameters[f"W{i}"], parameter_grads[f"dW{i}"])
            self.biases[f"b{i}"] = self.optimizer.compute(self.biases[f"b{i}"], biases_grads[f"db{i}"])

        return loss

    def predict(self, x):
        """
        Prediction

        :param x: unknown data
        :return: prediction. Unscaled
        """
        _, activations = self.forward(x)
        return activations[f"A{self.hiddens}"]

    def train(self, x, y):
        """
        Forward and backward propagation in one round

        :param learning_rate:  learning rate for an epoch
        :param x: training data
        :param y: labels
        :return: cost of the epoch
        """
        logits, activations = self.forward(x)
        return self.backward(logits, activations, x, y)

    def mini_batch(self, x, y, batch_size=16):
        """
        Minibatch training
        :param x: training data
        :param y: labels
        :param learning_rate: learning rate at each batch
        :param batch_size: batchs size
        :return: list of every loss at each batch
        """
        losses = []
        for i in range(0, x.shape[1], batch_size):
            _x = x[:, i * batch_size:i * batch_size + batch_size]
            _y = y[:, i * batch_size:i * batch_size + batch_size]
            logits, activations = self.forward(x)
            loss = self.backward(logits, activations, x, y)
            losses.append(loss)
        return losses

    def stochastic_batch(self, x, y):
        index = random.randint(0, x.shape[1])
        _x = x[:, index].reshape(x.shape[0], 1)
        _y = y[:, index].reshape(y.shape[0], 1)
        logits, activations = self.forward(_x)
        return self.backward(logits, activations, _x, _y)
