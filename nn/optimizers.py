import abc


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def compute(self, trainables, gradients):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def compute(self, trainables, gradients):
        return trainables - self.learning_rate * gradients
