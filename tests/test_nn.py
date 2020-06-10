import unittest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from nn.activations import Relu, Softmax, CatCrossEntropy
from nn.nn import NeuralNetwork


def generate_categorical_data():
    (X, Y), (x, y) = mnist.load_data()

    Y = Y.reshape(Y.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    Y = to_categorical(Y, num_classes=10).T
    y = to_categorical(y, num_classes=10).T

    X = X.reshape(X.shape[0], -1).T
    x = x.reshape(x.shape[0], -1).T

    X = X / 255.
    x = x / 255.
    return X, Y, x, y


class TestNN(unittest.TestCase):
    def test_something(self):
        X, Y, x, y = generate_categorical_data()
        net = NeuralNetwork([X.shape[0], 256, y.shape[0]], Relu(), Softmax(), CatCrossEntropy())
        last_loss = net.train(X, Y)
        for i in range(100):
            loss = net.train(X, Y)
            assert (loss < last_loss)
            last_loss = loss


if __name__ == '__main__':
    unittest.main()
