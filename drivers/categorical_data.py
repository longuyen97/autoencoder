import matplotlib.pyplot as plt
import numpy as np
from nn.data import generate_categorical_data
from nn.nn import NeuralNetwork
from nn.activations import Relu, Softmax
from nn.losses import CategoricalCrossEntropy
from nn.metrics import Accuracy
from tqdm import tqdm

X, Y, x, y = generate_categorical_data()
activation = Relu()
scale = Softmax()
loss_function = CategoricalCrossEntropy()
accuracy = Accuracy()
net = NeuralNetwork([X.shape[0], 256, Y.shape[0]], activation, scale, loss_function)

data = dict()
data["loss"] = []
data["val_loss"] = []
data["acc"] = []
data["val_acc"] = []

epochs = 20
progress = tqdm(range(epochs))
for i in progress:
    loss = net.train(X, Y, learning_rate=0.1)
    data["loss"].append(loss.item())
    prediction = net.predict(X)
    argmax_prediction = np.argmax(prediction, axis=0)
    ground_truth = np.argmax(Y, axis=0)
    acc = accuracy.compute(argmax_prediction, ground_truth)
    data["acc"].append(acc.item())
    val_prediction = net.predict(x)
    val_prediction_argmax = np.argmax(val_prediction, axis=0)
    val_ground_truth = np.argmax(y, axis=0)
    val_acc = accuracy.compute(val_prediction_argmax, val_ground_truth)
    val_loss = loss_function.compute(y, val_prediction)
    data["val_acc"].append(val_acc.item())
    data["val_loss"].append(val_loss.item())
    progress.set_description(f"Epoch {i} loss {loss.item()} acc {acc.item()} val_loss {val_loss.item()} val_acc {val_acc.item()}")

x = [i for i in range(epochs)]
plt.figure(figsize=(10, 10))
plt.plot(x, data["loss"], label="loss")
plt.plot(x, data["val_loss"], label="val_loss")
plt.plot(x, data["acc"], label="acc")
plt.plot(x, data["val_acc"], label="val_acc")
plt.legend()
plt.show()
