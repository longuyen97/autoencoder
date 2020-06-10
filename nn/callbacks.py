import abc


class Callback(abc.ABC):
    @abc.abstractmethod
    def begin(self):
        pass

    @abc.abstractmethod
    def middle(self):
        pass

    @abc.abstractmethod
    def end(self):
        pass
