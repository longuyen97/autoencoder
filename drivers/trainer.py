import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, activation, scale, loss_function, metrics, net):
        self.activation = activation
        self.scale = scale
        self.loss_function = loss_function
        self.metrics = metrics
        self.net = net
        self.epochs = 0

    def train(self, X, Y, x, y, epochs):
        self.epochs = epochs
        data = dict()
        data["loss"] = []
        data["val_loss"] = []
        data["acc"] = []
        data["val_acc"] = []
        progress = tqdm(range(epochs))
        for i in progress:
            loss = self.net.train(X, Y)
            data["loss"].append(loss.item())
            prediction = self.net.predict(X)
            argmax_prediction = np.argmax(prediction, axis=0)
            ground_truth = np.argmax(Y, axis=0)
            acc = self.metrics.compute(argmax_prediction, ground_truth)
            data["acc"].append(acc.item())
            val_prediction = self.net.predict(x)
            val_prediction_argmax = np.argmax(val_prediction, axis=0)
            val_ground_truth = np.argmax(y, axis=0)
            val_acc = self.metrics.compute(val_prediction_argmax, val_ground_truth)
            val_loss = self.loss_function.compute(y, val_prediction)
            data["val_acc"].append(val_acc.item())
            data["val_loss"].append(val_loss.item())
            progress.set_description(
                f"Epoch {i} loss {loss.item()} acc {acc.item()} val_loss {val_loss.item()} val_acc {val_acc.item()}")
        return data

    def plot(self, data):
        x = [i for i in range(self.epochs)]
        plt.figure(figsize=(10, 10))
        plt.plot(x, data["loss"], label="loss")
        plt.plot(x, data["val_loss"], label="val_loss")
        plt.plot(x, data["acc"], label="acc")
        plt.plot(x, data["val_acc"], label="val_acc")
        plt.legend()
        plt.show()


class BinaryTrainer(Trainer):
    def train(self, X, Y, x, y, epochs):
        self.epochs = epochs
        data = dict()
        data["loss"] = []
        data["val_loss"] = []
        data["acc"] = []
        data["val_acc"] = []
        progress = tqdm(range(epochs))
        for i in progress:
            loss = self.net.train(X, Y)
            data["loss"].append(loss[0][0])

            prediction = np.round(self.net.predict(X))
            acc = self.metrics.compute(Y[0], prediction[0])
            data["acc"].append(acc.item())

            val_prediction = self.net.predict(x)
            val_acc = self.metrics.compute(y[0], np.round(val_prediction[0]))
            val_loss = self.loss_function.compute(y, val_prediction)
            data["val_loss"].append(val_loss.item())
            data["val_acc"].append(val_acc.item())

            progress.set_description(
                f"Epoch {i} loss {loss.item()} acc {acc.item()} val_loss {val_loss.item()} val_acc {val_acc.item()}")
        return data


class AutoencoderTrainer(Trainer):
    def train(self, X, Y, x, y, epochs):
        self.epochs = epochs
        data = dict()
        data["loss"] = []
        data["val_loss"] = []
        data["acc"] = []
        data["val_acc"] = []
        progress = tqdm(range(epochs))
        for i in progress:
            loss = self.net.train(X, X)
            data["loss"].append(np.mean(loss))
            val_prediction = self.net.predict(x)
            val_loss = self.loss_function.compute(x, val_prediction)
            data["val_loss"].append(np.mean(val_loss))
            progress.set_description(f"Epoch {i} loss {np.mean(loss)} val_loss {np.mean(val_loss)} ")
        return data

    def plot(self, data):
        x = [i for i in range(self.epochs)]
        plt.figure(figsize=(10, 10))
        plt.plot(x, data["loss"], label="loss")
        plt.plot(x, data["val_loss"], label="val_loss")
        plt.legend()
        plt.show()


class RegressionTrainer(Trainer):
    def train(self, X, Y, x, y, epochs):
        self.epochs = epochs
        data = dict()
        data["loss"] = []
        data["val_loss"] = []
        data["acc"] = []
        data["val_acc"] = []
        progress = tqdm(range(epochs))
        for i in progress:
            loss = self.net.train(X, Y)
            data["loss"].append(np.mean(loss))
            val_prediction = self.net.predict(x)
            val_loss = self.loss_function.compute(y, val_prediction)
            data["val_loss"].append(np.mean(val_loss))
            progress.set_description(f"Epoch {i} loss {np.mean(loss)} val_loss {np.mean(val_loss)} ")
        return data

    def plot(self, data):
        x = [i for i in range(self.epochs)]
        plt.figure(figsize=(10, 10))
        plt.plot(x, data["loss"], label="loss")
        plt.plot(x, data["val_loss"], label="val_loss")
        plt.legend()
        plt.show()