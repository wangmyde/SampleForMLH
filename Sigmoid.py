import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.activations = 1. / (1. + np.exp( - self.input_tensor))
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * self.activations * (1. - self.activations)

