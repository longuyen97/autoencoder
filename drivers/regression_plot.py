from drivers.trainer import RegressionTrainer
from nn.activations import Relu, LinearActivation
from nn.losses import MeanAbsoluteError
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_binary_data
from nn.optimizers import GradientDescent

optimizer = GradientDescent(learning_rate=0.01, depth=3)
X, Y, x, y = generate_binary_data()
activation = Relu()
scale = LinearActivation()
loss_function = MeanAbsoluteError()
metrics = Accuracy()
net = NeuralNetwork([784, 32, 1], activation, scale, loss_function, optimizer)
trainer = RegressionTrainer(activation, scale, loss_function, metrics, net)
history = trainer.train(X, Y, x, y, 20)
trainer.plot(history)
