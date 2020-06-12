import unittest

import numpy as np

from nn.activations import Relu, Softmax
from nn.data import generate_categorical_data
from nn.losses import CrossEntropy
from nn.metrics import Accuracy
from nn.nn import NeuralNetwork
from nn.optimizers import GradientDescent


class TestCategorical(unittest.TestCase):
    def test_categorical(self):
        optimizer = GradientDescent(learning_rate=0.01, depth=3)
        metric = Accuracy()
        X, Y, x, y = generate_categorical_data()
        print(X.shape, Y.shape)
        net = NeuralNetwork([X.shape[0], 256, Y.shape[0]], Relu(), Softmax(), CrossEntropy(), optimizer)
        last_loss = net.train(X, Y)
        for i in range(100):
            loss = net.train(X, Y)
            assert (loss < last_loss)
            last_loss = loss
            prediction = np.argmax(net.predict(X), axis=0)
            ground_truth = np.argmax(Y, axis=0)
            acc = metric.compute(prediction, ground_truth)
            print(f"Epoch {i} Loss {last_loss.item()} Accuracy {acc}")


if __name__ == '__main__':
    unittest.main()
