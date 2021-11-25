import numpy as np
from . import Base


class ReLU(Base.Base_class):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor[self.input_tensor < 0.] = 0.
        return self.input_tensor

    def backward(self, error_tensor):
        np.place(error_tensor, self.input_tensor <= 0., 0.)
        return error_tensor
