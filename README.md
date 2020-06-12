# Neuronal networks with Numpy 

This project is my Neural Network playground with Numpy.


![alt-text](images/autoencoder-result.png)

### Requirements
- numpy==1.18.5
- autograd==1.3

# Neural Networks
- Multi class classification
- Binary classification
- Real valued regression
- Auto-Encoder
- Stacked (Deep) Auto-Encoder
- Error Backpropagation Multi-Layer Perceptron

### Categorical classification

```python
from nn.data import generate_categorical_data
from nn.activations import Relu, Softmax
from nn.losses import CrossEntropy
from nn.nn import NeuralNetwork

X, Y = generate_categorical_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 128, Y.shape[0]], Relu(), Softmax(), CrossEntropy())
loss = net.train(X, Y)
```

Model's learning curve

![alt-text](images/categorical.png)
### Binary classification

```python
from nn.nn import NeuralNetwork
from nn.data import generate_binary_data
from nn.activations import Relu, Sigmoid
from nn.losses import CrossEntropy

X, Y = generate_binary_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 128, 1], Relu(), Sigmoid(), CrossEntropy())
loss = net.train(X, Y)
```

Model's learning curve

![alt-text](images/binary.png)
### Autoencoder

```python
from nn.data import generate_categorical_data
from nn.activations import Relu, LinearActivation
from nn.losses import MeanAbsoluteError
from nn.nn import NeuralNetwork

X, Y = generate_categorical_data()
net = NeuralNetwork([X.shape[0], 512, 256, 256, 512, X.shape[0]], Relu(), LinearActivation(), MeanAbsoluteError())
loss = net.train(X, X)
```

Model's learning curve

![alt-text](images/autoencoder.png)
### Regression problem

```python
from nn.data import generate_regression_data
from nn.activations import Relu, LinearActivation
from nn.losses import MeanAbsoluteError
from nn.nn import NeuralNetwork

X, Y = generate_regression_data()
net = NeuralNetwork([X.shape[0], 32, 1], Relu(), LinearActivation(), MeanAbsoluteError())
loss = net.train(X, Y)
```
![alt-text](images/regression.png)