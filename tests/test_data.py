import unittest

from nn.data import generate_data, generate_binary_data, generate_categorical_data


class TestGeneratingData(unittest.TestCase):
    def test_generate_data(self):
        X, Y, x, y = generate_data()
        self.assertTrue(X.shape == (60000, 28, 28))
        self.assertTrue(x.shape == (10000, 28, 28))
        self.assertTrue(Y.shape == (60000,))
        self.assertTrue(y.shape == (10000,))

    def test_generate_binary_data(self):
        X, Y, x, y = generate_binary_data()
        print(X.shape, Y.shape, x.shape, y.shape)

    def test_generate_categorical_data(self):
        X, Y, x, y = generate_categorical_data()
        print(X.shape, Y.shape, x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
