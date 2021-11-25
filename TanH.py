import numpy as np


class TanH:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        #print('input_tensor=', input_tensor)
        self.input_tensor = input_tensor
        self.activations = np.tanh(self.input_tensor)
        #print('activiation.shape=', self.activations.shape)
        return self.activations

    def backward(self, error_tensor):
        #print('error_tensor=', (error_tensor * (1. - self.activations**2)))
        return error_tensor * (1. - self.activations**2)

