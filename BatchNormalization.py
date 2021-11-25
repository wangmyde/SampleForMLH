import numpy as np
import copy
from . import Base


class BatchNormalization(Base.Base_class):
    def __init__(self, channels=0):
        super().__init__()
        self.channels = channels
        self.phase = Base.Phase.train
        self.weights = None
        self.bias = None
        self.epsilon = 1e-8
        self.alpha = 1.
        self.mean = None
        self.var = None
        self.optimizer = None
        self.bias_optimizer = None
        self.delta = 1

    def initialize(self, weights_initializer, bias_initializer):
        pass

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.batch_size, self.input_size = input_tensor.shape
        if self.phase == Base.Phase.train:
            if self.channels == 0:
                self.input_tensor = input_tensor
                return self.normalization(self.input_tensor)
            else:
                self.input_tensor = input_tensor.reshape(np.int(input_tensor.shape[1]/self.channels) * input_tensor.shape[0], self.channels)
                self.input_normed = self.normalization(self.input_tensor)
                return self.input_normed.reshape((self.batch_size, self.input_size))

        if self.phase == Base.Phase.test:
            if self.channels == 0:
                self.input_normed = (input_tensor - self.mean) / np.sqrt(self.var + self.epsilon)
                return self.weights * self.input_normed + self.bias
            else:
                input_tensor = input_tensor.reshape(np.int(input_tensor.shape[1]/self.channels) * input_tensor.shape[0], self.channels)
                self.input_normed = (input_tensor - self.mean) / np.sqrt(self.var + self.epsilon)
                self.input_normed = self.weights * self.input_normed + self.bias
                return self.input_normed.reshape((self.batch_size, self.input_size))

    def normalization(self, input_tensor):
        if self.weights is None:
            self.weights = np.ones((1, input_tensor.shape[1]))
            self.bias = np.zeros((1, input_tensor.shape[1]))
            self.mean = np.zeros((1, input_tensor.shape[1]))
            self.var = np.zeros((1, input_tensor.shape[1]))
        mean = np.mean(input_tensor, axis=0).reshape(1, input_tensor.shape[1])
        var = np.var(input_tensor, axis=0).reshape(1, input_tensor.shape[1])
        self.mean = (1 - self.alpha) * self.mean + self.alpha * mean
        self.var = (1 - self.alpha) * self.var + self.alpha * var
        self.input_normed = (input_tensor - mean) / np.sqrt(var + self.epsilon)
        self.input_normed = self.weights * self.input_normed + self.bias
        return self.input_normed

    def backward(self, error_tensor):
        if self.channels == 0:
            return self.gradient_calculate(error_tensor)
        else:
            error_tensor = error_tensor.reshape(np.int(error_tensor.shape[1]/self.channels) * error_tensor.shape[0], self.channels)
            return self.gradient_calculate(error_tensor).reshape((self.batch_size, self.input_size))

    def gradient_calculate(self, error_tensor):
        batch_size, input_size = error_tensor.shape
        gradient_weights = np.sum(error_tensor * self.input_normed, axis=0).reshape(1, error_tensor.shape[1])
        gradient_bias = np.sum(error_tensor, axis=0).reshape(1, error_tensor.shape[1])
        gradient_normed = error_tensor * self.weights
        gradient_var = np.sum(gradient_normed * (self.input_tensor - self.mean) * (-1. / 2) * np.power(self.var + self.epsilon, -3./2), axis=0).reshape(1, error_tensor.shape[1])
        gradient_mean = np.sum(gradient_normed * (-1. / np.sqrt(self.var + self.epsilon)), axis=0).reshape(1, error_tensor.shape[1])
        error_tensor_pre_layer = gradient_normed * (1. / np.sqrt(self.var + self.epsilon)) \
                                + gradient_var * 2. * (self.input_tensor - self.mean) / batch_size + gradient_mean / batch_size
        self._gradient_bias = gradient_bias
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, gradient_bias)
        self._gradient_weights = gradient_weights
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, gradient_weights)
        return error_tensor_pre_layer

    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias
