from layer import Layer

class Activation(Layer):
    def __init__(self, activation_func, activation_func_prime):
        super().__init__()
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation_func(input_data)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_func_prime(self.input)
