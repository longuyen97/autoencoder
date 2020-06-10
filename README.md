# Neuronal networks with Numpy 

This project is my Neural Network playground with Numpy.

# Neural Networks
- Autoencoder
- Stacked (Deep) Autoencoder
- Error Backpropagation Multi-Layer Perceptron
- Webcam Integration

### Categorical classification

```python
from nn.data import generate_categorical_data, generate_binary_data, generate_data
from nn.activations import Relu, Softmax, CategoricalCrossEntropy, BinaryCrossEntropy, Sigmoid

net = NeuralNetwork([X.shape[0], 256, Y.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
X, Y = generate_categorical_data()
loss = net.train(X, Y)
```
