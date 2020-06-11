import unittest

from nn.data import generate_data, generate_binary_data, generate_categorical_data, generate_regression_data


class TestGeneratingData(unittest.TestCase):
    def test_generate_data(self):
        X, Y, x, y = generate_data()
        self.assertTrue(X.shape == (60000, 28, 28))
        self.assertTrue(x.shape == (10000, 28, 28))
        self.assertTrue(Y.shape == (60000,))
        self.assertTrue(y.shape == (10000,))

    def test_generate_binary_data(self):
        X, Y, x, y = generate_binary_data()
        self.assertTrue(X.shape == (784, 12665))
        self.assertTrue(Y.shape == (1, 12665))
        self.assertTrue(x.shape == (10000, 784))
        self.assertTrue(y.shape == (1, 10000))

    def test_generate_categorical_data(self):
        X, Y, x, y = generate_categorical_data()
        self.assertTrue(X.shape == (784, 60000))
        self.assertTrue(Y.shape == (10, 60000))
        self.assertTrue(x.shape == (784, 10000))
        self.assertTrue(y.shape == (10, 10000))

    def test_generate_regression_data(self):
        X, Y, x, y = generate_regression_data()
        self.assertTrue(X.shape == (1, 1600))
        self.assertTrue(Y.shape == (1, 1600))
        self.assertTrue(x.shape == (1, 400))
        self.assertTrue(y.shape == (1, 400))


if __name__ == '__main__':
    unittest.main()
