import cv2
from nn.nn import NeuralNetwork
from nn.activations import Relu, Softmax
from nn.losses import CategoricalCrossEntropy
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
flatted = gray.flatten()
net = NeuralNetwork([flatted.shape[0], 512, 256, 512, flatted.shape[0]], Relu(), Softmax(), CategoricalCrossEntropy())
while cv2.waitKey(1) != 27:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flatted = gray.flatten()
    net.train(flatted, flatted)
    prediction = net.predict(flatted)
    prediction = np.argmax(prediction)
    prediction = prediction.reshape(gray.shape)
    cv2.imshow('frame', prediction)

cap.release()
cv2.destroyAllWindows()
