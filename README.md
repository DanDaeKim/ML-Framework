# Simple Machine Learning Framework

A basic neural network framework built from scratch in Python. This project demonstrates the core concepts of creating a machine learning framework, including layers, activation functions, forward and backward propagation, and training.

## Features

- Basic neural network architecture
- Dense (fully connected) layers
- Activation functions: ReLU and Sigmoid
- Mean Squared Error (MSE) loss function
- Gradient descent optimization

## Installation

1. Clone this repository:
  - git clone https://github.com/DanDaeKim/ML-Framework.git

2. Change to the project directory:
  - cd simple_ml_framework
  
3. Create a virtual environment and activate it:
  - python -m venv venv
  - source venv/bin/activate # or "venv\Scripts\activate" on Windows
  
4. Install the required packages:
  - pip install numpy


## Usage

1. Import the necessary classes and functions from the framework:

```python
import numpy as np
from neural_network import NeuralNetwork
from dense import Dense
from activation import Activation, relu, relu_prime, sigmoid, sigmoid_prime

2. Generate a sample dataset, create a neural network, add layers, and train the network:
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

3. Test the neural network:
y_pred = nn.forward(X)
y_pred = (y_pred > 0.5).astype(int)

accuracy = np.mean(y_true == y_pred)
print(f'Accuracy: {accuracy}')

# Future Improvements
Add support for more layer types, such as convolutional and recurrent layers
Implement additional activation functions and loss functions
Use advanced optimization techniques like Adam or RMSProp
Add support for loading and saving models
Integrate a more efficient library like TensorFlow or PyTorch for underlying operations


