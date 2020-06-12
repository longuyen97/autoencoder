from drivers.trainer import Trainer
from nn.activations import Relu, Softmax
from nn.losses import CrossEntropy
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_categorical_data

X, Y, x, y = generate_categorical_data()
activation = Relu()
scale = Softmax()
loss_function = CrossEntropy()
metrics = Accuracy()
net = NeuralNetwork([784, 512, 258, 64, 10], activation, scale, loss_function)
trainer = Trainer(activation, scale, loss_function, metrics, net)
history = trainer.train(X, Y, x, y, 20, 0.1)
trainer.plot(history)
