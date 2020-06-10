import unittest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from nn.activations import Relu, Softmax, CategoricalCrossEntropy, BinaryCrossEntropy, Sigmoid
from nn.nn import NeuralNetwork


def generate_categorical_data():
    (X, Y), (_, _) = mnist.load_data()
    Y = Y.reshape(Y.shape[0], 1)
    Y = to_categorical(Y, num_classes=10).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y


def generate_binary_data():
    (X, Y), (_, _) = mnist.load_data()
    mask = Y == 1
    X = X[mask]
    Y = Y[mask]
    Y = Y.reshape(Y.shape[0], 1)
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y


class TestNN(unittest.TestCase):
    def test_categorical(self):
        X, Y = generate_categorical_data()
        net = NeuralNetwork([X.shape[0], 256, Y.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
        last_loss = net.train(X, Y)
        for i in range(100):
            loss = net.train(X, Y)
            assert (loss < last_loss)
            last_loss = loss

    def test_binary(self):
        X, Y = generate_binary_data()
        net = NeuralNetwork([X.shape[0], 256, 1], Relu(), Sigmoid(), BinaryCrossEntropy())
        for i in range(100):
            loss = net.train(X, Y)
            print(loss)


if __name__ == '__main__':
    unittest.main()
