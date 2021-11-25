import numpy as np
from . import Base


class Pooling(Base.Base_class):
    def __init__(self, input_image_shape, stride_shape, pooling_shape):
        super().__init__()
        self.img_z, self.img_y, self.img_x = input_image_shape
        self.stride_y, self.stride_x = stride_shape
        self.pooling_y, self.pooling_x = pooling_shape
        self.out_x = (self.img_x - self.pooling_x) // self.stride_x + 1  # floor division 2 // 3 = 0
        self.out_y = (self.img_y - self.pooling_y) // self.stride_y + 1

    def forward(self, input_tensor):
        self.batch = input_tensor.shape[0]
        self.input_tensor = input_tensor.reshape(self.batch, self.img_z, self.img_y, self.img_x)
        #print('test0', self.input_tensor)
        self.out = np.zeros((self.batch, self.img_z, self.out_y, self.out_x))
        for b in range(self.batch):
            for c in range(self.img_z):
                for i in range(self.out_y):
                    for j in range(self.out_x):
                        self.out[b, c, i, j] = np.max(self.input_tensor[b, c, i * self.stride_y : i * self.stride_y + self.pooling_y, \
                                                                            j * self.stride_x : j * self.stride_x + self.pooling_x])
        #print('test1', self.out)
        self.input_tensor_new = self.out.reshape(self.batch, self.img_z * self.out_y * self.out_x)
        return self.input_tensor_new

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.batch, self.img_z, self.out_y, self.out_x)
        error_tensor_low = np.zeros_like(self.input_tensor)
        #print(self.new_y, self.new_x)
        #print('test0', error_tensor)
        #print('test1', self.input_tensor)
        for b in range(self.batch):
            for c in range(self.img_z):
                for i in range(self.out_y):
                    for j in range(self.out_x):
                        x_pool = self.input_tensor[b, c, i * self.stride_y : i * self.stride_y + self.pooling_y, \
                                                        j * self.stride_x : j * self.stride_x + self.pooling_x]
                        mask = (x_pool == np.max(x_pool))
                        error_tensor_low[b, c, i * self.stride_y : i * self.stride_y + self.pooling_y, \
                                        j * self.stride_x : j * self.stride_x + self.pooling_x] += mask * error_tensor[b, c, i, j]
        error_tensor_low = error_tensor_low.reshape(self.batch, self.img_z * self.img_y * self.img_x)
        return error_tensor_low
