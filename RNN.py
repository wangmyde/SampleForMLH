import numpy as np
from . import TanH, Sigmoid, FullyConnected


class RNN:
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        self.input_size = input_size        #13
        self.hidden_size = hidden_size      #7
        self.bptt_length = bptt_length      #9 = batch_size
        self.output_size = output_size      #5
        self.tbptt = False
        self.weigths = np.random.uniform(0, 1, (hidden_size + input_size + 1, hidden_size))   #Whh + Whx
        self.weigths_cur = np.random.uniform(0, 1, (hidden_size + 1, output_size))   #Why
        self.fc1 = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.fc2 = FullyConnected.FullyConnected(hidden_size, output_size)
        self.Sigmoid = Sigmoid.Sigmoid()

        self.TanH = TanH.TanH()
        self.ht0 = 0


    def toggle_memory(self):
        self.tbptt = True
        pass


    def forward(self, input_tensor):
        #print(input_tensor.shape)
        self.input_tensor = input_tensor
        self.batch_size, input_size = input_tensor.shape
        self.ht = np.zeros((self.batch_size + 1,self.hidden_size))
        #print("self.ht.shape:", self.ht.shape)
        self.fc1_output = np.zeros((self.batch_size,self.hidden_size))
        self.fc1_input = np.zeros((self.batch_size, self.input_size + self.hidden_size))
        #self.fc2_weights = np.zeros((self.hidden_size+self.input_size +1, self.hidden_size))
        self.Tanh_activations = np.zeros((self.batch_size,self.hidden_size))
        if self.tbptt:
            self.ht[0, :] = self.ht0

        for ite in range(self.batch_size):
            #print(self.ht[ite,:])
            #print(input_tensor[ite,:])

            self.fc1_input[ite, :] = np.hstack((self.ht[ite,:], input_tensor[ite,:]))
            #self.fc1.weights = self.weigths
            self.fc1_output[ite,:] = self.fc1.forward(self.fc1_input[ite, :].reshape((1, self.hidden_size + self.input_size)))
            #self.weigths = self.fc1.weights

            self.ht[ite+1, :] = self.TanH.forward(self.fc1_output[ite,:])
            self.Tanh_activations[ite,:] = self.TanH.activations
        self.ht0 = self.ht[-1, :]
        fc2_input = np.delete(self.ht, 0, axis=0)
        fc2_output = self.fc2.forward(fc2_input)
        output = self.Sigmoid.forward(fc2_output)
        return output

    def backward(self, error_tensor):
        #print(error_tensor.shape)
        self.gradient_weights_cur = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights = np.zeros((self.hidden_size+self.input_size + 1, self.hidden_size))
        tanh_prime = 1 - self.ht[1::,:]**2
        #print("tan_prime:",tan_prime, tan_prime.shape)
        grad_h0 = np.zeros(self.hidden_size).reshape(1, self.hidden_size)
        backprop_error_tensor = np.zeros((self.batch_size, self.input_size))
        counter = 0

        for t in reversed(range(self.batch_size)):
            self.fc2.input_tensor = np.hstack((self.ht[t + 1, :], 1)).reshape(1, self.hidden_size + 1)
            #print("self.fc2.input_tensor:",self.fc2.input_tensor)
            grad_h1 = self.fc2.backward(error_tensor[t, :].reshape(1, self.output_size))
            grad_h = grad_h1 + grad_h0
            grad_mid_net = grad_h * tanh_prime[t, :]
            #print("grad_h * tanh_prime[t, :]:", grad_h * tanh_prime[t, :])
            stacked_input = np.hstack((self.ht[t, :], self.))







        # self.error_tensor = error_tensor
        # dot = self.Sigmoid.backward(error_tensor)
        #
        # bias = np.ones(self.ht.shape[0])
        # bias2 = np.ones(self.fc1_input.shape[0])
        # fc2_input_tensor = np.column_stack((self.ht, bias))
        # fc1_input_tensor = np.column_stack((self.fc1_input, bias2))
        # fc1_back_output = np.zeros((self.batch_size, self.hidden_size+self.input_size))
        # fc2_back_output = self.fc2.backward(dot)
        # self.gradient_weights_cur = self.fc2.get_gradient_weights()
        # counter = 0
        # for ite in reversed(range(self.batch_size + 1)):
        #
        #     if ite == self.batch_size:
        #         self.dht = np.ones((self.batch_size + 1,self.hidden_size))
        #         self.fc2.input_tensor = fc2_input_tensor[ite, :].reshape((1,self.hidden_size + 1))
        #         self.dht[ite, :] = fc2_back_output[ite-1, :]
        #     else:
        #         self.fc2.input_tensor = fc2_input_tensor[ite + 1, :].reshape((1,self.hidden_size + 1))
        #         self.TanH.activations = self.Tanh_activations[ite, :]
        #         self.fc1.input_tensor = fc1_input_tensor[ite, :].reshape((1,self.hidden_size+self.input_size+1))
        #         fc1_back_output[ite, :] = self.fc1.backward(self.TanH.backward(self.dht[ite+1, :].reshape(1,self.hidden_size)))
        #         dht, dxt = np.hsplit(fc1_back_output, [self.hidden_size])
        #         self.dht[ite, :] = dht[ite, :] + fc2_back_output[ite,:]
        #         counter += 1
        #         if counter <= self.batch_size:
        #             self.gradient_weights += self.fc1.get_gradient_weights()
        # print("dht:", dht, dht.shape)
        # if self.optimizer is not None:
        #     self.fc2.weights = self.optimizer.calculate_update(self.delta,  self.fc2.weights, self.gradient_weights_cur)
        #     self.fc1.weights = self.optimizer.calculate_update(self.delta, self.fc1.weights, self.gradient_weights)
        #
        #
        # return dxt

    def get_gradient_weights(self):
        return self.gradient_weights


    def get_weights(self):
        return self.weigths


    def set_weights(self, weights):
        self.weigths = weights


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def initialize(self, weights_initializer, bias_initializer):
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)


