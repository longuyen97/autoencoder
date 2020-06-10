import unittest
import numpy as np
from nn.activations import Relu, Softmax, Sigmoid
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_categorical_data, generate_binary_data
from nn.losses import CategoricalCrossEntropy, BinaryCrossEntropy


class TestNN(unittest.TestCase):
    def test_categorical(self):
        X, Y, x, y = generate_categorical_data()
        print(X.shape, Y.shape)
        net = NeuralNetwork([X.shape[0], 256, Y.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
        last_loss = net.train(X, Y)
        last_acc = 0
        for i in range(100):
            loss = net.train(X, Y)
            assert (loss < last_loss)
            last_loss = loss
            prediction = np.argmax(net.predict(X), axis=0)
            ground_truth = np.argmax(Y, axis=0)
            acc = Accuracy().compute(prediction, ground_truth)
            print(f"Epoch {i} Loss {last_loss.item()} Accuracy {acc}")
            self.assertTrue(acc > last_acc)
            last_acc = acc

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
