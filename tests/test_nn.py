import unittest
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from nn.activations import Relu, Softmax, CatCrossEntropy
from nn.nn import NeuralNetwork

(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()

# Preparing the data
Y_tr_resh = Y_train_orig.reshape(60000, 1)
Y_te_resh = Y_test_orig.reshape(10000, 1)
Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
Y_te_T = to_categorical(Y_te_resh, num_classes=10)
Y_train = Y_tr_T.T
Y_test = Y_te_T.T

# Flattening of inputs
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Preprocessing of Inputs
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.


class TestNN(unittest.TestCase):
    def test_something(self):
        net = NeuralNetwork([X_train.shape[0], 256, Y_train.shape[0]])
        first_loss = net.train(X_train, Y_train, Relu(), Softmax(), CatCrossEntropy())
        second_loss = net.train(X_train, Y_train, Relu(), Softmax(), CatCrossEntropy())
        self.assertTrue(first_loss > second_loss)


if __name__ == '__main__':
    unittest.main()
