import unittest

from nn.data import generate_data


class MyTestCase(unittest.TestCase):
    def test_generate_data(self):
        (X, Y), (x, y) = generate_data()
        self.assertTrue(X.shape == (60000, 28, 28))
        self.assertTrue(x.shape == (10000, 28, 28))
        self.assertTrue(Y.shape == (60000,))
        self.assertTrue(y.shape == (10000,))


if __name__ == '__main__':
    unittest.main()
