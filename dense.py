from layer import Layer

class Dense(Layer):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros((1, output_units))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient
