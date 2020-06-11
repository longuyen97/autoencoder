# Neuronal networks with Numpy 

This project is my Neural Network playground with Numpy.

### Requirements
- numpy==1.18.5
- opencv-python==4.2.0.34
- mnist==0.2.2
- matplotlib==3.2.1

# Neural Networks
- Multi class classification
- Binary classification
- Real valued regression
- Autoencoder
- Stacked (Deep) Autoencoder
- Error Backpropagation Multi-Layer Perceptron
- Webcam Integration

### Categorical classification

```python
from nn.data import generate_categorical_data
from nn.activations import Relu, Softmax
from nn.losses import CategoricalCrossEntropy
from nn.nn import NeuralNetwork

X, Y = generate_categorical_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 128, Y.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
loss = net.train(X, Y)
```

### Binary classification

```python
from nn.nn import NeuralNetwork
from nn.data import generate_binary_data
from nn.activations import Relu, Sigmoid
from nn.losses import BinaryCrossEntropy

X, Y = generate_binary_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 128, 1], Relu(), Sigmoid(), BinaryCrossEntropy())
loss = net.train(X, Y)
```

### Autoencoder

```python
from nn.data import generate_categorical_data
from nn.activations import Relu, Softmax
from nn.losses import MeanSquaredError
from nn.nn import NeuralNetwork

X, Y = generate_categorical_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 512, X.shape[0]], Relu(), Softmax(), MeanSquaredError())
loss = net.train(X, X)
```
