import typing
import numpy as np
from nn.activations import Activation, Linear, Sigmoid, LeakyRelu
from nn.data import generate_categorical_data
from nn.losses import CrossEntropy


class GanNeuralNetwork:
    def __init__(self, layers: typing.List, activation: Activation, scale: Activation):
        self.linear = Linear()
        self.activation = activation
        self.scale = scale
        self.layers = layers
        self.parameters = dict()
        self.biases = dict()
        self.depth = len(layers)
        self.hiddens = len(layers) - 1
        for i in range(1, self.depth):
            self.parameters[f"W{i}"] = np.random.randn(layers[i], layers[i - 1]) * (np.sqrt(2 / layers[i - 1]))
            self.biases[f"b{i}"] = np.zeros((layers[i], 1))

    def forward(self, x) -> typing.Tuple[typing.Dict, typing.Dict]:
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

    def backward(self, logits, activations, x, loss, learning_rate=0.001):
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
        logit_grads[f"dZ{self.hiddens}"] = loss

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

        for i in range(1, self.depth):
            self.parameters[f"W{i}"] -= (learning_rate * parameter_grads[f"dW{i}"])
            self.biases[f"b{i}"] -= (learning_rate * biases_grads[f"db{i}"])


LATENT_SPACE_DIM = 100
X, Y, x, y = generate_categorical_data()
dis = GanNeuralNetwork([X.shape[0], 512, 256, 64, 32, 1], LeakyRelu(), Sigmoid())
gen = GanNeuralNetwork([LATENT_SPACE_DIM, 256, 512, X.shape[0]], LeakyRelu(), Linear())
cross_entropy = CrossEntropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy.derivative(np.ones_like(real_output), real_output)
    fake_loss = cross_entropy.derivative(np.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy.derivative(np.ones_like(fake_output), fake_output)


def train(real_images):
    noise = np.random.normal([real_images.shape[1], LATENT_SPACE_DIM])

    gen_logits, gen_activations = gen.forward(noise)

    dis_real_logits, dis_real_activations = dis.forward(real_images)

    dis_gen_logits, dis_gen_activations = dis.forward(gen_activations[f"A{gen.hiddens}"])

    gen_loss = generator_loss(gen_activations[f"A{gen.hiddens}"])

    dis_loss = discriminator_loss(real_images, gen_activations[f"A{gen.hiddens}"])

    dis.backward(dis_real_logits, dis_real_activations, real_images, dis_loss)

    dis.backward(dis_gen_logits, dis_gen_activations, real_images, dis_loss)

    gen.backward(gen_logits, gen_activations, real_images, gen_loss)
