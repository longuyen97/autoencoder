from drivers.trainer import AutoencoderTrainer
from nn.activations import Relu, LinearActivation
from nn.losses import MeanAbsoluteError
from nn.nn import NeuralNetwork
from nn.metrics import Accuracy
from nn.data import generate_binary_data
import matplotlib.pyplot as plt
from nn.optimizers import GradientDescent

optimizer = GradientDescent(learning_rate=0.01, depth=8)
X, Y, x, y = generate_binary_data()
activation = Relu()
scale = LinearActivation()
loss_function = MeanAbsoluteError()
metrics = Accuracy()
net = NeuralNetwork([784, 512, 256, 128, 128, 256, 512, 784], activation, scale, loss_function, optimizer)
trainer = AutoencoderTrainer(activation, scale, loss_function, metrics, net)
history = trainer.train(X, Y, x, y, 20)
trainer.plot(history)


sample = x[:, 0].reshape((784, 1))
logits, activations = net.forward(sample)
z2 = logits["Z2"].reshape((16, 16))
a7 = activations["A7"].reshape((28, 28))
f, axarr = plt.subplots(1,3)
axarr[0].imshow(sample.reshape((28, 28)), cmap="gray")
axarr[1].imshow(z2, cmap="gray")
axarr[2].imshow(a7, cmap="gray")
for i in range(3):
    axarr[i].set_xticks([], minor=False)
    axarr[i].set_yticks([], minor=False)
plt.show()