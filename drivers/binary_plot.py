from drivers.trainer import BinaryTrainer
from nn.activations import Relu, Sigmoid
from nn.losses import CrossEntropy
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_binary_data
from nn.optimizers import GradientDescent

optimizer = GradientDescent(learning_rate=0.01, depth=5)
X, Y, x, y = generate_binary_data()
activation = Relu()
scale = Sigmoid()
loss_function = CrossEntropy()
metrics = Accuracy()
net = NeuralNetwork([784, 512, 258, 64, 1], activation, scale, loss_function, optimizer)
trainer = BinaryTrainer(activation, scale, loss_function, metrics, net)
history = trainer.train(X, Y, x, y, 20)
trainer.plot(history)
