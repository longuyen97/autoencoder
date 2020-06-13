import abc
import numpy as np


class Optimizer(abc.ABC):
    def __init__(self, depth):
        self.depth = depth

    @abc.abstractmethod
    def compute(self, weights, biases, weights_gradients, biases_gradients):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, depth):
        super().__init__(depth)
        self.learning_rate = learning_rate

    def compute(self, weights, biases, weights_gradients, biases_gradients):
        for i in range(2, self.depth):
            weights[f"W{i}"] = weights[f"W{i}"] - self.learning_rate * weights_gradients[f"dW{i}"]
            biases[f"b{i}"] = biases[f"b{i}"] - self.learning_rate * biases_gradients[f"db{i}"]
        return weights, biases


class MomentumGradientDescent(Optimizer):
    def __init__(self, learning_rate, depth, momentum=0.9):
        super().__init__(depth)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights_velocity = dict()
        self.biases_velocity = dict()
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = 0
            self.biases_velocity[f"v{i}"] = 0

    def compute(self, weights, biases, weights_gradients, biases_gradients):
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] *= self.momentum
            self.weights_velocity[f"v{i}"] += self.learning_rate * weights_gradients[f"dW{i}"]
            weights[f"W{i}"] -= self.weights_velocity[f"v{i}"]

            self.biases_velocity[f"v{i}"] *= self.momentum
            self.biases_velocity[f"v{i}"] += self.learning_rate * biases_gradients[f"dW{i}"]
            biases[f"b{i}"] -= self.biases_velocity[f"v{i}"]
        return weights, biases


class NesterovGradientDescent(Optimizer):
    def __init__(self, learning_rate, depth, alpha, gamma):
        super().__init__(depth)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.weights_velocity = dict()
        self.biases_velocity = dict()
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = 0
            self.biases_velocity[f"v{i}"] = 0

    def compute(self, weights, biases, weights_gradients, biases_gradients):
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = self.gamma * self.weights_velocity[f"v{i}"] + self.alpha * weights_gradients[f"dW{i}"]
            weights[f"W{i}"] -= self.weights_velocity[f"v{i}"]

            self.biases_velocity[f"v{i}"] = self.gamma * self.biases_velocity[f"v{i}"] + self.alpha * biases_gradients[f"db{i}"]
            biases[f"b{i}"] -= self.biases_velocity[f"v{i}"]
        return weights, biases


class Adagrad(Optimizer):
    def __init__(self, learning_rate, depth, alpha, eps):
        super().__init__(depth)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weights_gradients_cache = dict()
        self.weights_biases_cache = dict()
        self.initialized = False

    def compute(self, weights, biases, weights_gradients, biases_gradients):
        if not self.initialized:
            for i in range(2, self.depth):
                self.weights_gradients_cache[f"c{i}"] = np.zeros_like(weights_gradients[f"dW{i}"])
                self.weights_biases_cache[f"c{i}"] = np.zeros_like(biases_gradients[f"db{i}"])

        for i in range(2, self.depth):
            self.weights_gradients_cache[f"c{i}"] += weights_gradients[f"dW{i}"]**2
            weights[f"W{i}"] -= self.alpha * weights_gradients[f"dW{i}"] / (np.sqrt(self.weights_gradients_cache[f"c{i}"]) + self.eps)

            self.weights_biases_cache[f"c{i}"] += biases_gradients[f"db{i}"]**2
            biases[f"b{i}"] -= self.alpha * biases_gradients[f"db{i}"] / (np.sqrt(self.weights_biases_cache[f"c{i}"]) + self.eps)

        return weights, biases


class Adadelta(Optimizer):
    def compute(self, weights, biases, weights_gradients, biases_gradients):
        pass


class RMSprop(Optimizer):
    def compute(self, weights, biases, weights_gradients, biases_gradients):
        pass


class Adam(Optimizer):
    def compute(self, weights, biases, weights_gradients, biases_gradients):
        pass