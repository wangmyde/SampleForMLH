import numpy as np
from . import Base


class SoftMax(Base.Base_class):
    def __init__(self):
        super().__init__()
        self.y_hat = None

    def forward(self, input_tensor, label_tensor):
        self.y_hat = self.predict(input_tensor)
        return np.sum(np.multiply(label_tensor, -np.log(self.y_hat)))

    def predict(self, input_tensor):
        exp_X = np.exp(input_tensor - np.max(input_tensor))
        sum_exp = np.sum(exp_X, axis=1, keepdims=True)
        return exp_X / sum_exp

    def backward(self, label_tensor):
        return self.y_hat - label_tensor
