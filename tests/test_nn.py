import unittest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from nn.activations import Relu, Softmax, CategoricalCrossEntropy, BinaryCrossEntropy, Sigmoid
from nn.nn import NeuralNetwork
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_categorical_data():
    (X, Y), (_, _) = mnist.load_data()
    Y = Y.reshape(Y.shape[0], 1)
    Y = to_categorical(Y, num_classes=10).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y


def generate_binary_data():
    (X, Y), (y, y) = mnist.load_data()
    Y_ones = Y[Y == 1]
    Y_twos = Y[Y == 2]
    X_ones = X[Y == 1]
    X_twos = X[Y == 2]
    X = np.concatenate([X_ones, X_twos])
    Y = np.concatenate([Y_ones, Y_twos])
    Y[Y == 1] = 1
    Y[Y != 1] = 0
    Y = Y.reshape(Y.shape[0], 1).T
    X = X.reshape(X.shape[0], -1).T
    X = X / 255.
    return X, Y


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
