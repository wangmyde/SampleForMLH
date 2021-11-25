import numpy as np
import copy
from . import Base


class FullyConnected(Base.Base_class):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        #print(self.weights)
        self.delta = 1.
        self.error_tensor = None
        self.optimizer = None
        self.input_tensor = None
        # self.gradient_weights = None
        self.gradient_weights = np.zeros_like(self.weights)

    def initialize(self, weights_initializer, bias_initializer):
        weights, bias = np.vsplit(self.weights, [-1])
        weights = weights_initializer.initialize(weights)
        bias = bias_initializer.initialize(bias)
        self.weights = np.vstack((weights, bias))

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        bias = np.ones(input_tensor.shape[0])
        self.input_tensor = np.column_stack((input_tensor, bias))
        #print(self.input_tensor)
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_tensor_pre_layer = np.dot(self.error_tensor, self.weights.T)
        error_tensor_pre_layer = np.delete(error_tensor_pre_layer, -1, 1)
        self.gradient_weights = np.dot(self.input_tensor.T, self.error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.gradient_weights)
        return error_tensor_pre_layer

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
