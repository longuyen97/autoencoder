import numpy as np
import abc


class Metrics(abc.ABC):
    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        pass


class Accuracy(Metrics):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(y_true == y_pred)


class ConfusionMatrix(Metrics):
    def compute(self, y_true, y_pred):
        pass
