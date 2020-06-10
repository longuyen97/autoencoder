from nn.nn import NeuralNetwork
from nn.data import generate_categorical_data
from nn.activations import Relu, Softmax
from nn.losses import CategoricalCrossEntropy

X, _ = generate_categorical_data()
net = NeuralNetwork([X.shape[0], 512, 256, 512, X.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
for i in range(100):
    loss = net.train(X, X)
    print(f"Epoch {i} Loss {loss.item()}")
