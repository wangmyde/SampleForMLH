import numpy as np


class Constant():
    def __init__(self, cons):
        self.cons = cons

    def initialize(self,weights):
        return self.cons * np.ones(weights.shape)


class UniformRandom():
    def __init__(self):
        pass

    def initialize(self,weights):
        return np.random.uniform(0, 1, weights.shape)


class Xavier():
    def __init__(self):
        pass

    def initialize(self, weights):
        sigma = np.sqrt(2.) / np.sqrt((weights.shape[0]) + (weights.shape[1]))
        #print(sigma)
        return np.random.normal(0, sigma, weights.shape)
        #return sigma * np.random.randn(weights.shape[0], weights.shape[1])


