import numpy as np
from neural_network import NeuralNetwork
from dense import Dense
from activation import Activation, relu, relu_prime, sigmoid, sigmoid_prime

# Generate a sample dataset
X = np.random.randn(100, 2)
y_true = np.array([0 if x1 * x2 < 0 else 1 for x1, x2 in X]).reshape(-1, 1)

# Create a neural network
nn = NeuralNetwork()
nn.add_layer(Dense(2, 8))
nn.add_layer(Activation(relu, relu_prime))
nn.add_layer(Dense(8, 1))
nn.add_layer(Activation(sigmoid, sigmoid_prime))

# Train the neural network
nn.train(X, y_true, epochs=2000, learning_rate=0.1)

# Test the neural network
y_pred = nn.forward(X)
y_pred = (y_pred > 0.5).astype(int)

accuracy = np.mean(y_true == y_pred)
print(f'Accuracy: {accuracy}')

