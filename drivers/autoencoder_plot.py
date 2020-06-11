from drivers.trainer import AutoencoderTrainer
from nn.activations import Relu, LinearActivation
from nn.losses import MeanAbsoluteError
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_binary_data

X, Y, x, y = generate_binary_data()
activation = Relu()
scale = LinearActivation()
loss_function = MeanAbsoluteError(axis=1)
metrics = Accuracy()
net = NeuralNetwork([784, 512, 256, 256, 512, 784], activation, scale, loss_function)
trainer = AutoencoderTrainer(activation, scale, loss_function, metrics, net)
history = trainer.train(X, Y, x, y, 20, 0.1)
trainer.plot(history)
