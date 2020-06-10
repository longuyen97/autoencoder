from nn.nn import NeuralNetwork
from nn.activations import Relu, Softmax
from nn.losses import CategoricalCrossEntropy


class AutoEncoder:
    def __init__(self, layers):
        self.net = NeuralNetwork(layers, Relu(), Softmax(), CategoricalCrossEntropy())
