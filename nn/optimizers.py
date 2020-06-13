import abc

import numpy as np


class Optimizer(abc.ABC):
    def __init__(self, depth):
        self.depth = depth

    @abc.abstractmethod
    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, depth):
        super().__init__(depth)
        self.learning_rate = learning_rate

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
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

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
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

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = self.gamma * self.weights_velocity[f"v{i}"] + self.alpha * \
                                             weights_gradients[f"dW{i}"]
            weights[f"W{i}"] -= self.weights_velocity[f"v{i}"]

            self.biases_velocity[f"v{i}"] = self.gamma * self.biases_velocity[f"v{i}"] + self.alpha * biases_gradients[
                f"db{i}"]
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

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
        if not self.initialized:
            for i in range(2, self.depth):
                self.weights_gradients_cache[f"c{i}"] = np.zeros_like(weights_gradients[f"dW{i}"])
                self.weights_biases_cache[f"c{i}"] = np.zeros_like(biases_gradients[f"db{i}"])
            self.initialized = True

        for i in range(2, self.depth):
            self.weights_gradients_cache[f"c{i}"] += weights_gradients[f"dW{i}"] ** 2
            weights[f"W{i}"] -= self.alpha * weights_gradients[f"dW{i}"] / (
                        np.sqrt(self.weights_gradients_cache[f"c{i}"]) + self.eps)

            self.weights_biases_cache[f"c{i}"] += biases_gradients[f"db{i}"] ** 2
            biases[f"b{i}"] -= self.alpha * biases_gradients[f"db{i}"] / (
                        np.sqrt(self.weights_biases_cache[f"c{i}"]) + self.eps)

        return weights, biases


class RMSprop(Optimizer):
    def __init__(self, learning_rate, depth, alpha, gamma, eps):
        super().__init__(depth)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.weights_gradients_cache = dict()
        self.weights_biases_cache = dict()
        self.initialized = False

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
        if not self.initialized:
            for i in range(2, self.depth):
                self.weights_gradients_cache[f"c{i}"] = np.zeros_like(weights_gradients[f"dW{i}"])
                self.weights_biases_cache[f"c{i}"] = np.zeros_like(biases_gradients[f"db{i}"])
            self.initialized = True

        for i in range(2, self.depth):
            self.weights_gradients_cache[f"c{i}"] += self.gamma * self.weights_gradients_cache[f"c{i}"] ** 2 + (
                        1 - self.gamma) * (weights_gradients[f"dW{i}"] ** 2)
            weights[f"W{i}"] -= self.alpha * weights_gradients[f"dW{i}"] / (
                        np.sqrt(self.weights_gradients_cache[f"c{i}"]) + self.eps)

            self.weights_biases_cache[f"c{i}"] += self.gamma * self.weights_biases_cache[f"c{i}"] ** 2 + (
                        1 - self.gamma) * (biases_gradients[f"db{i}"] ** 2)
            biases[f"b{i}"] -= self.alpha * biases_gradients[f"db{i}"] / (
                        np.sqrt(self.weights_biases_cache[f"c{i}"]) + self.eps)

        return weights, biases


class Adam(Optimizer):
    def __init__(self, learning_rate, depth, alpha, eps, beta1=0.9, beta2=0.999):
        super().__init__(depth)
        self.learning_rate = learning_rate
        self.initialized = False
        self.alpha = alpha
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weights_M = dict()
        self.weights_R = dict()
        self.biases_M = dict()
        self.biases_R = dict()
        self.epoch = 1

    def compute(self, weights, biases, weights_gradients, biases_gradients, epoch=0):
        if not self.initialized:
            for i in range(2, self.depth):
                self.weights_M[f"c{i}"] = np.zeros_like(weights_gradients[f"dW{i}"])
                self.weights_R[f"c{i}"] = np.zeros_like(weights_gradients[f"dW{i}"])
                self.biases_M[f"c{i}"] = np.zeros_like(biases_gradients[f"db{i}"])
                self.biases_R[f"c{i}"] = np.zeros_like(biases_gradients[f"db{i}"])
            self.initialized = True

        for i in range(2, self.depth):
            self.weights_M[f"c{i}"] = self.beta1 * self.weights_M[f"c{i}"] + (1 - self.beta1) * weights_gradients[f"dW{i}"]
            self.weights_R[f"c{i}"] = self.beta2 * self.weights_R[f"c{i}"] + (1 - self.beta1) * weights_gradients[f"dW{i}"]**2

            weights_m_hat = self.weights_M[f"c{i}"] / (1. - self.beta1 ** self.epoch)
            weights_r_hat = self.weights_R[f"c{i}"] / (1. - self.beta2 ** self.epoch)
            weights[f"W{i}"] -= self.alpha * weights_m_hat / (np.sqrt(weights_r_hat) + self.eps)

            self.biases_M[f"c{i}"] = self.beta1 * self.biases_M[f"c{i}"] + (1 - self.beta1) * biases_gradients[f"dW{i}"]
            self.biases_R[f"c{i}"] = self.beta2 * self.biases_R[f"c{i}"] + (1 - self.beta1) * biases_gradients[f"dW{i}"]**2

            biases_m_hat = self.biases_M[f"c{i}"] / (1. - self.beta1 ** self.epoch)
            biases_r_hat = self.biases_R[f"c{i}"] / (1. - self.beta2 ** self.epoch)
            biases[f"b{i}"] -= self.alpha * biases_m_hat / (np.sqrt(biases_r_hat) + self.eps)
        self.epoch +=1
        return weights, biases
