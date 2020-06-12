import unittest

import numpy as np

from nn.activations import Relu, Sigmoid
from nn.data import generate_binary_data
from nn.losses import CrossEntropy
from nn.metrics import Accuracy
from nn.nn import NeuralNetwork
from nn.optimizers import GradientDescent


class TestBinary(unittest.TestCase):
    def test_binary(self):
        optimizer = GradientDescent(learning_rate=0.0001)
        X, Y, x, y = generate_binary_data()
        print(X.shape, Y.shape)
        net = NeuralNetwork([X.shape[0], 256, 1], Relu(), Sigmoid(), CrossEntropy(), optimizer)
        for i in range(100):
            loss = net.train(X, Y)
            prediction = np.round(net.predict(X))
            acc = Accuracy().compute(Y[0], prediction[0])
            print(f"Epoch {i} Loss {loss.item()} Accuracy {acc}")


if __name__ == '__main__':
    unittest.main()
