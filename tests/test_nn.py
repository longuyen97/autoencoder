import unittest
import numpy as np
from nn.activations import Relu, Softmax, CategoricalCrossEntropy, BinaryCrossEntropy, Sigmoid
from nn.nn import NeuralNetwork
from sklearn.metrics import accuracy_score
from nn.data import generate_categorical_data, generate_binary_data


class TestNN(unittest.TestCase):
    def test_categorical(self):
        X, Y = generate_categorical_data()
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
            acc = accuracy_score(prediction, ground_truth)
            self.assertTrue(acc > last_acc)
            last_acc = acc
            print(f"Epoch {i} Loss {last_loss.item()} Accuracy {acc}")

    def test_binary(self):
        X, Y = generate_binary_data()
        print(X.shape, Y.shape)
        net = NeuralNetwork([X.shape[0], 256, 1], Relu(), Sigmoid(), BinaryCrossEntropy())
        last_acc = 0
        for i in range(100):
            loss = net.train(X, Y)
            prediction = np.round(net.predict(X))
            acc = accuracy_score(Y[0], prediction[0])
            self.assertTrue(acc > last_acc)
            last_acc = acc
            print(f"Epoch {i} Loss {loss.item()} Accuracy {acc}")


if __name__ == '__main__':
    unittest.main()
