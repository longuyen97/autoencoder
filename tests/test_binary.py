import unittest

import numpy as np

from nn.activations import Relu, Sigmoid
from nn.data import generate_binary_data
from nn.losses import BinaryCrossEntropy
from nn.metrics import Accuracy
from nn.nn import NeuralNetwork


class TestBinary(unittest.TestCase):
    def test_binary(self):
        X, Y, x, y = generate_binary_data()
        print(X.shape, Y.shape)
        net = NeuralNetwork([X.shape[0], 256, 1], Relu(), Sigmoid(), BinaryCrossEntropy())
        last_acc = 0
        for i in range(100):
            loss = net.train(X, Y)
            prediction = np.round(net.predict(X))
            acc = Accuracy().compute(Y[0], prediction[0])
            print(f"Epoch {i} Loss {loss.item()} Accuracy {acc}")
            self.assertTrue(acc > last_acc)
            last_acc = acc


if __name__ == '__main__':
    unittest.main()
