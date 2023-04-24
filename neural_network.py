import numpy as np
from dense import Dense
from activation import Activation, relu, relu_prime, sigmoid, sigmoid_prime

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def compute_loss_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y_true, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_true, y_pred)
            loss_gradient = self.compute_loss_prime(y_true, y_pred)
            self.backward(loss_gradient, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
