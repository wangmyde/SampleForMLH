from Layers.FullyConnected import *
from Layers.Conv import *
from Layers import *
import pickle


class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.regularizer = None

    def append_trainable_layer(self, layer):
        layer.initialize(self.weights_initializer, self.bias_initializer)
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)

    def set_phase(self, phase):
        for layer in self.layers:
            layer.phase = phase

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        regularization_loss = 0

        for layer in self.layers:
            if self.regularizer is not None:
                if isinstance(layer, FullyConnected):
                    weights, bias = np.vsplit(layer.weights, [-1])
                    regularization_loss += layer.optimizer.regularizer.norm(weights)
                if isinstance(layer, Conv):
                    regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
            # regularization_loss += layer.optimizer.norm(layer.weights)
            self.input_tensor = layer.forward(self.input_tensor)
        self.input_tensor += regularization_loss
        self.loss.append(self.loss_layer.forward(self.input_tensor, self.label_tensor))

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        self.set_phase(Base.Phase.train)
        for i in range(iterations):
            #print('ite=', i)
            self.forward()
            self.backward()

    def test(self, input_tensor):
        self.set_phase(Base.Phase.test)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        predict_tensor = self.loss_layer.predict(input_tensor)
        return predict_tensor

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data_layer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data_layer = None


def save(filename, net):
    with open(filename, 'wb') as file:
        pickle.dump(net, file)
        file.close()


def load(filename, data_layer):
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    net.data_layer = data_layer
    return net
