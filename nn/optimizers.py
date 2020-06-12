import abc


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
        self.learning_Rate = learning_rate
        self.momentum = momentum
        self.weights_velocity = dict()
        self.biases_velocity = dict()
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = 0
            self.biases_velocity[f"v{i}"] = 0

    def compute(self, weights, biases, weights_gradients, biases_gradients):
        for i in range(2, self.depth):
            self.weights_velocity[f"v{i}"] = self.momentum * self.weights_velocity[f"v{i}"] + self.learning_rate * weights_gradients[f"dW{i}"]
            weights[f"W{i}"] -= self.weights_velocity[f"v{i}"]
            self.biases_velocity[f"v{i}"] = self.momentum * self.biases_velocity[f"v{i}"] + self.learning_rate * biases_gradients[f"dW{i}"]
            biases[f"b{i}"] -= self.biases_velocity[f"v{i}"]
        return weights, biases
