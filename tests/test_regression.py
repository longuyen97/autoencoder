import unittest

from nn.activations import Relu, LinearActivation
from nn.data import generate_regression_data
from nn.losses import MeanAbsoluteError, MeanSquaredError
from nn.nn import NeuralNetwork


class TestRegression(unittest.TestCase):
    def test_regression_mae(self):
        X, Y, x, y = generate_regression_data()
        print(X.shape, Y.shape)
        loss_function = MeanAbsoluteError()
        scaling_function = LinearActivation()
        activation_function = Relu()
        net = NeuralNetwork([X.shape[0], 32, 1], activation_function, scaling_function, loss_function)

        for i in range(25):
            loss = net.train(X, Y, learning_rate=0.00001)
            pred = net.predict(x)
            val_loss = loss_function.compute(y, pred)
            print(f"Epoch {i} Loss {loss.item()} Val loss {val_loss.item()}")

    def test_regression_mse(self):
        X, Y, x, y = generate_regression_data()
        print(X.shape, Y.shape)
        loss_function = MeanSquaredError()
        net = NeuralNetwork([X.shape[0], 32, 1], Relu(), LinearActivation(), loss_function)
        for i in range(125):
            loss = net.train(X, Y, learning_rate=0.00001)
            pred = net.predict(x)
            val_loss = loss_function.compute(y, pred)
            print(f"Epoch {i} Loss {loss.item()} Val loss {val_loss.item()}")


if __name__ == '__main__':
    unittest.main()
