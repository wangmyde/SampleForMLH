import numpy as np
from . import Base


class Dropout(Base.Base_class):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.phase = Base.Phase.train
        self.mask = None

    def forward(self, input_tensor):
        if self.phase == Base.Phase.train:
            #print('test1', Base.Phase.train)
            self.mask = np.random.choice([0,1], size=input_tensor.shape, p=[1-self.probability, self.probability])
            #print('mask=', self.mask)
            #print('dropout_forward=',(input_tensor * self.mask) / self.probability)
            return (input_tensor * self.mask) / self.probability

        if self.phase == Base.Phase.test:
            #print('input_tensor=', input_tensor)
            return input_tensor

    def backward(self, error_tensor):
        #if self.phase == Base.Phase.train:
        return error_tensor * self.mask
        #if self.phase == Base.Phase.test:
            #return error_tensor
