import numpy as np
from . import TanH, Sigmoid, FullyConnected


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_length = bptt_length
        self.tbptt = False
        self.fc1 = FullyConnected.FullyConnected(hidden_size + input_size, 4 * hidden_size)
        self.fc2 = FullyConnected.FullyConnected(hidden_size, output_size)
        self.sigmoid_f = Sigmoid.Sigmoid()
        self.sigmoid_i = Sigmoid.Sigmoid()
        self.sigmoid_o = Sigmoid.Sigmoid()
        self.sigmoid_y = Sigmoid.Sigmoid()
        self.tanh_c = TanH.TanH()
        self.tanh_cell = TanH.TanH()

        self.hidden_state_init = 0


    def toggle_memory(self):
        self.tbptt = True

    def forward(self, input_tensor):
        self.batch_size, input_size = input_tensor.shape
        self.hidden_state = np.zeros((self.batch_size + 1,self.hidden_size))
        self.cell_state = np.zeros((self.batch_size + 1, self.hidden_size))
        self.pro_ft = np.zeros((self.batch_size, self.hidden_size))
        self.pro_it = np.zeros((self.batch_size, self.hidden_size))
        self.pro_ot = np.zeros((self.batch_size, self.hidden_size))
        self.pro_ct = np.zeros((self.batch_size, self.hidden_size))
        self.activ_f = np.zeros((self.batch_size, self.hidden_size))
        self.activ_i = np.zeros((self.batch_size, self.hidden_size))
        self.activ_o = np.zeros((self.batch_size, self.hidden_size))
        self.activ_c = np.zeros((self.batch_size, self.hidden_size))
        self.fc1_output = np.zeros((self.batch_size, 4 * self.hidden_size))
        self.fc1_input = np.zeros((self.batch_size, self.input_size + self.hidden_size))
        fc2_output = np.zeros((self.batch_size, self.output_size))
        if self.tbptt:
            self.hidden_state[0, :] = self.hidden_state_init

        for ite in range(self.batch_size):
            self.fc1_input[ite, :] = np.hstack((self.hidden_state[ite, :], input_tensor[ite, :]))

            self.fc1_output[ite, :] = self.fc1.forward(self.fc1_input[ite, :].reshape((1, self.hidden_size + self.input_size)))
            ft, it, ct, ot = np.hsplit(self.fc1_output[ite, :], 4)
            self.pro_ft[ite, :] = self.sigmoid_f.forward(ft.reshape(1,self.hidden_size))
            self.activ_f[ite, :] = self.sigmoid_f.activations
            self.pro_it[ite, :] = self.sigmoid_i.forward(it.reshape(1, self.hidden_size))
            self.activ_i[ite, :] = self.sigmoid_i.activations
            self.pro_ot[ite, :] = self.sigmoid_o.forward(ot.reshape(1, self.hidden_size))
            self.activ_o[ite, :] = self.sigmoid_o.activations
            self.pro_ct[ite, :] = self.tanh_c.forward(ct.reshape(1, self.hidden_size))
            self.activ_c[ite, :] = self.tanh_c.activations
            self.cell_state[ite + 1, :] = self.cell_state[ite, :] * self.pro_ft[ite, :] + self.pro_it[ite, :] * self.pro_ct[ite, :]

            self.hidden_state[ite + 1, :] = self.tanh_cell.forward(self.cell_state[ite+1, :].reshape(1, self.hidden_size)) * self.pro_ot[ite, :]
            self.hidden_state_init = self.hidden_state[-1, :]
        fc2_input = np.delete(self.hidden_state, 0, axis=0)
        fc2_output = self.fc2.forward(fc2_input)
        output_tensor = self.sigmoid_y.forward(fc2_output)

        return output_tensor

    def backward(self, error_tensor):
        self.gradient_weights = np.zeros((self.hidden_size + self.input_size + 1, 4* self.hidden_size))
        delta_ht = np.zeros((self.hidden_size))
        fc2_back_output = self.fc2.backward(self.sigmoid_y.backward(error_tensor))  #delta_M
        error_tensor_pre = np.zeros((self.batch_size, self.input_size))
        bias = np.ones(self.fc1_input.shape[0])
        fc1_input_tensor = np.column_stack((self.fc1_input, bias))
        counter = 0
        for ite in reversed(range(self.batch_size)):
            self.sigmoid_f.activations = self.activ_f[ite,:]
            self.sigmoid_i.activations = self.activ_i[ite, :]
            self.sigmoid_o.activations = self.activ_o[ite, :]
            self.tanh_c.activations = self.activ_c[ite, :]
            delta_ct = delta_ht * self.pro_ot[ite,:]*(1. - self.cell_state[ite+1, :] **2)
            delta_it = self.sigmoid_i.backward((delta_ct * self.pro_ct[ite,:]).reshape(1,self.hidden_size))
            delta_c_hat_t = self.tanh_c.backward((delta_ct * self.pro_it[ite,:]).reshape(1,self.hidden_size))
            delta_ot = self.sigmoid_o.backward((delta_ht * np.tanh(self.cell_state[ite+1,:])).reshape(1,self.hidden_size))
            delta_ft = delta_ht * self.pro_ot[ite, :]
            fc1_back_input = np.vstack((delta_ft, delta_it, delta_c_hat_t, delta_ot)).reshape(1, 4*self.hidden_size)
            self.fc1.input_tensor = fc1_input_tensor[ite, :].reshape(1, self.hidden_size + self.input_size +1)
            fc1_back_output = self.fc1.backward(fc1_back_input)
            delta_ht, error_tensor_pre[ite,:] = np.hsplit(fc1_back_output, [self.hidden_size])
            delta_ht = delta_ht + fc2_back_output[ite, :]
            counter += 1
            if counter <= self.batch_size:
                self.gradient_weights += self.get_weights()

        return error_tensor_pre

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_weights(self):
        return self.fc1.get_weights()

    def set_weights(self, weights):
        self.fc1.set_weights(weights)

    def set_optimizer(self, optimizer):
        self.fc1.set_optimizer(optimizer)
        self.fc2.set_optimizer(optimizer)

    def initialize(self, weights_initializer, bia_initializer):
        self.fc1.initialize(weights_initializer, bia_initializer)
        self.fc2.initialize(weights_initializer, bia_initializer)
