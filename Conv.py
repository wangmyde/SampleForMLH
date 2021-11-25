import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import correlate
from scipy.signal import correlate2d
from scipy.signal import correlate as correlate1d
import copy
from . import Base


class Conv(Base.Base_class):
    def __init__(self, input_image_shape, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.delta = 1.
        shape = (num_kernels, *convolution_shape)   # 4,3,5,8
        self.weights = np.random.random(shape)
        self.bias = np.random.random(num_kernels)
        self.optimizer = None
        self.bias_optimizer = None
        self.input_tensor = None
        self.output_shape = []
        image_shape = self.input_image_shape[1:]
        for input_dim, stride in zip(image_shape, self.stride_shape):
            rounded = np.int(np.ceil(np.float(input_dim) / np.float(stride)))
            self.output_shape.append(rounded)
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

    def initialize(self, weights_initializer, bias_initializer):
                                                  #4,3,5,8
        self.weights = weights_initializer.initialize(self.weights)
        self.bias = bias_initializer.initialize(self.bias)

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        #print(self.output_shape)
        #print(input_tensor.shape)
        output_size = np.prod(self.output_shape)
        output_tensor = np.zeros((batch_size, self.num_kernels * output_size))
        #print('o_t = ', output_tensor.shape)
        for b in range(batch_size):
            input_image = input_tensor[b, :]
            input_image = input_image.reshape(*self.input_image_shape)
            #input_image.reshape(3, 10, 14)
            #print('imag shape = ', input_image.shape)
            for k, kernel in enumerate(self.weights):
                convresult = convolve(input_image, kernel, mode='constant')
                mid = np.int(np.floor(self.weights.shape[1] / 2.))
                convresult = convresult[mid]
                result = np.zeros(self.output_shape)
                if np.size(self.output_shape) > 1:
                    for p in range(self.output_shape[0]):
                        for q in range(self.output_shape[1]):
                            result[p,q] = convresult[p * self.stride_shape[0], q * self.stride_shape[1]]
                else:
                    for p in range(self.output_shape[0]):
                        result = convresult[p * self.stride_shape[0]]
                start = k * output_size
                end = (k + 1) * output_size
                output_tensor[b, start:end] = result.flatten()
                output_tensor[b, start:end] += self.bias[k]
        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        shape = (batch_size, np.prod(self.input_image_shape))
        error_tensor_below = np.zeros(shape)
        spatial_output = np.prod(self.input_image_shape[1:])
        gradient_weights = np.zeros_like(self.weights)
        gradient_bias = np.zeros_like(self.bias)
        for b in range(batch_size):
            error_image = np.zeros((self.num_kernels, *self.input_image_shape[1:]))
            error = error_tensor[b, :]
            error = error.reshape((self.num_kernels, *self.output_shape))
            for i in range(self.num_kernels):
                gradient_bias[i] += np.sum(error[i, :])
            if np.size(self.output_shape) > 1:
                for p in range(self.output_shape[0]):
                    for q in range(self.output_shape[1]):
                        error_image[:, p * self.stride_shape[0], q * self.stride_shape[1]] += error[:, p, q]
            else:
                for p in range(self.output_shape[0]):
                    error_image[:, p * self.stride_shape[0]] += error[:, p]

            num_channel = self.weights.shape[1]
            for i, rev in zip(range(num_channel), reversed(range(num_channel))):
                kernel = self.weights[:, i, ...]
                result = correlate(error_image, kernel, mode='constant', cval=0.)
                mid = np.int(np.floor(self.weights.shape[0] / 2.))
                result = result[mid]
                start = rev * spatial_output
                end = (rev + 1) * spatial_output
                error_tensor_below[b, start:end] = result.flatten()

            activations = self.input_tensor[b, :]
            activations = activations.reshape(self.input_image_shape)   #3,5,7
            #print('test1', error_image.shape[0])

            for o in range(activations.shape[0]):   #z = 3
                for k in range(error_image.shape[0]):   # num_k = 3
                    padding = tuple((np.int(np.floor(x / 2.)), np.int(np.floor((x / 2.) - 0.49))) for x in self.weights.shape[2:])
                    activation = np.lib.pad(activations[o], padding, 'constant', constant_values=0.)
                    #print('test',activation.shape)
                    if len(activations[o].shape) > 1:
                        result = correlate2d(activation, error_image[k], mode='valid')
                    else:
                        result = correlate1d(activation, error_image[k], mode='valid')

                    for c in range(len(result.shape)):
                        result = np.flip(result, axis=c)
                    index = (self.weights.shape[1]-1) - o
                    gradient_weights[k, index, ...] += result
            self._gradient_bias = gradient_bias
            if self.bias_optimizer is not None:
                self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, gradient_bias)
            self._gradient_weights = gradient_weights
            if self.optimizer is not None:
                self.weights = self.optimizer.calculate_update(self.delta, self.weights, gradient_weights)
        return error_tensor_below

    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias
