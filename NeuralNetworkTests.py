import unittest
from Layers import *
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import datetime


class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.delta = 0.1
        layer.set_optimizer(Optimizers.Sgd(1))
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size,:] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestTanH(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = 1 - 2 / (np.exp(2*self.input_tensor) + 1)

        layer = TanH.TanH()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = TanH.TanH()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, -1.)

    def test_gradient(self):
        layers = list()
        layers.append(TanH.TanH())
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = 0.5 * (1. + np.tanh(self.input_tensor / 2.))

        layer = Sigmoid.Sigmoid()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = Sigmoid.Sigmoid()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)

    def test_gradient(self):
        layers = list()
        layers.append(Sigmoid.Sigmoid())
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss = layer.forward(input_tensor, self.label_tensor)

        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss = layer.forward(input_tensor, self.label_tensor)

        # test a specific value here
        self.assertAlmostEqual(float(loss), 909.8875105980)

    def test_regression_backward_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertGreaterEqual(element, 1 / 3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss = layer.forward(input_tensor, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)

    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertGreaterEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertLessEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        difference = Helpers.gradient_check([layer], input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.batch_size, self.categories))
        layer = SoftMax.SoftMax()
        prediction = layer.predict(input_tensor)
        expected_values = [[0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241]]
        np.testing.assert_almost_equal(prediction, expected_values)


class TestOptimizers(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(1., result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))

    def test_sgd_with_momentum(self):
        optimizer = Optimizers.SgdWithMomentum(1., 0.9)

        result = optimizer.calculate_update(1., 1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(1., result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]))

    def test_adam(self):
        optimizer = Optimizers.Adam(1., 0.01, 0.02, 1e-8)

        result = optimizer.calculate_update(1., 1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(1., result, 1.)
        np.testing.assert_almost_equal(result, np.array([-0.99999998000000034]))


class TestInitializers(unittest.TestCase):
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = np.random.random_sample((output_size, input_size + 1))

        def initialize(self, initializer):
            self.weights = initializer.initialize(self.weights)

    def setUp(self):
        self.batch_size = 9
        self.input_size = 200
        self.output_size = 50

    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        weights_before_init = layer.weights.copy()
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return weights_before_init, weights_after_init

    def test_uniform_shape(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        self.assertEqual(weights_before_init.shape, weights_after_init.shape)
        self.assertFalse(np.allclose(weights_before_init, weights_after_init))

    def test_uniform_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        p_value = stats.kstest(weights_after_init.flat, 'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_xavier_shape(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.Xavier())

        self.assertEqual(weights_before_init.shape, weights_after_init.shape)
        self.assertFalse(np.allclose(weights_before_init, weights_after_init))

    def test_xavier_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.Xavier())

        scale = np.sqrt(2.) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)


class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:

        @staticmethod
        def initialize(weights):
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 3 * 10 * 14
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 3 * 11 * 15
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        conv = Conv.Conv(self.input_shape, (1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape[1], self.spatial_input_shape * self.num_kernels)
        self.assertEqual(self.batch_size, output_tensor.shape[0])

    def test_forward_size_stride(self):
        conv = Conv.Conv(self.input_shape, (3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape[1], 28 * self.num_kernels)
        self.assertEqual(self.batch_size, output_tensor.shape[0])

    def test_forward_size_stride_uneven_image(self):
        conv = Conv.Conv(self.uneven_input_shape, (3, 2), self.kernel_shape, self.num_kernels + 1)
        input_tensor = np.array(range(self.uneven_input_size * (self.batch_size + 1)), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, np.prod(self.uneven_input_shape))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape[1], 32 * (self.num_kernels+1))
        self.assertEqual(self.batch_size+1, output_tensor.shape[0])

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 10, 14), (1, 1), (1, 3, 3), 1)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv.Conv((3, 10, 14), (1, 1), (3, 3, 3), 1)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3 , 10, 14))
        input_tensor[:,0] = tensor.copy()
        input_tensor[:,1] = tensor.copy()
        input_tensor[:,2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2)

    def test_1D_forward_size(self):
        conv = Conv.Conv((3, 15), [2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(45 * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, 45)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape[1], 8 * self.num_kernels)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        conv = Conv.Conv(self.input_shape, (1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], np.prod(self.input_shape))
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_backward_size_stride(self):
        conv = Conv.Conv(self.input_shape, (3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], np.prod(self.input_shape))
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_1D_backward_size(self):
        conv = Conv.Conv((3, 15), [2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(45 * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, 45)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], 45)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_1x1_convolution(self):
        conv = Conv.Conv(self.input_shape, (1, 1), (3, 1, 1), self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape[1], self.spatial_input_shape * self.num_kernels)
        self.assertEqual(output_tensor.shape[0], self.batch_size)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], np.prod(self.input_shape))
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_layout_preservation(self):
        conv = Conv.Conv(self.input_shape, (1, 1), (3, 3, 3), 1)
        conv.initialize(TestConv.TestInitializer(), Initializers.Constant(0.0))
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, np.prod(self.input_shape))
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor[:, 140:280])), 0.)

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 105)))
        layers = list()
        layers.append(Conv.Conv((3, 5, 7), (1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(FullyConnected.FullyConnected(35 * self.hidden_channels, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        description = 'Gradient_convolution_input'
        Helpers.plot_difference(self.plot, description, (15, 7), difference, self.directory)
        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2,105)))
        layers = list()
        layers.append(Conv.Conv((3, 5, 7), (1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(FullyConnected.FullyConnected(35*self.hidden_channels, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        description = 'Gradient_convolution_weights'
        Helpers.plot_difference(self.plot, description, (9, 9), difference.reshape(1, 81), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 105)))
        layers = list()
        layers.append(Conv.Conv((3, 5, 7), (2, 2), (3, 3, 3), self.hidden_channels))
        layers.append(FullyConnected.FullyConnected(12*self.hidden_channels, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        description = 'Gradient_convolution_weights'
        Helpers.plot_difference(self.plot, description, (9, 9), difference.reshape(1, 81), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 105)))
        layers = list()
        layers.append(Conv.Conv((3, 5, 7), (1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(FullyConnected.FullyConnected(35 * self.hidden_channels, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        description = 'Gradient_convolution_bias'
        Helpers.plot_difference(self.plot, description, (1, 3), difference.reshape(1, 3), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_stride(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 210)))
        layers = list()
        layers.append(Conv.Conv((3, 5, 14), (1, 2), (3, 3, 3), 1))
        layers.append(FullyConnected.FullyConnected(35, self.categories))
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)

        description = 'Gradient_strided_convolution_input'
        Helpers.plot_difference(self.plot, description, (15, 14), difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_update(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        conv = Conv.Conv(self.input_shape, (3, 2), self.kernel_shape, self.num_kernels)
        conv.set_optimizer(Optimizers.SgdWithMomentum(1, 0.9))
        conv.initialize(Initializers.Xavier(), Initializers.Constant(0.1))
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))


class TestPooling(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(None)
        self.layers.append(SoftMax.SoftMax())

        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_shape(self):
        layer = Pooling.Pooling(self.input_shape, (2, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 12])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = Pooling.Pooling(self.input_shape, (2, 1), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 24])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = Pooling.Pooling(self.input_shape, (3, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 6])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_gradient_stride(self):
        self.layers[0] = Pooling.Pooling(self.input_shape, (2, 2), (2, 2))
        self.layers[1] = (FullyConnected.FullyConnected(12, self.categories))

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        description = 'Gradient_Pooling_inputs'
        Helpers.plot_difference(self.plot, description, self.plot_shape, difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        self.layers[0] = Pooling.Pooling(self.input_shape, (2, 1), (2, 2))
        self.layers[1] = FullyConnected.FullyConnected(24, self.categories)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        description = 'Gradient_overlapping_Pooling_inputs'
        Helpers.plot_difference(self.plot, description, self.plot_shape, difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):

        self.layers[0] = Pooling.Pooling(self.input_shape, (3, 2), (2, 2))
        self.layers[1] = FullyConnected.FullyConnected(6, self.categories)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        description = 'Gradient of subsampling Pooling with respect to inputs'
        Helpers.plot_difference(self.plot, description, self.plot_shape, difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_layout_preservation(self):
        pool = Pooling.Pooling(self.input_shape, (1, 1), (1, 1))
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, self.input_size)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling.Pooling(input_shape, (2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, np.prod(input_shape))

        result = pool.forward(input_tensor)
        expected_result = np.array([[4], [13]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling.Pooling(input_shape, (2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, np.prod(input_shape))

        result = pool.forward(input_tensor)
        expected_result = np.array([[5, 21], [7, 23], [13, 29], [15, 31]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.individual_delta = 0.2
        self.delta = 0.1
        self.regularizer_strength = 1337
        self.shape = (4, 5)

    def test_L1(self):
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1
        weights_tensor = regularizer.calculate(weights_tensor)

        expected = np.ones(self.shape) * self.regularizer_strength
        expected[1:3, 2:4] *= -1

        difference = np.sum(np.abs(weights_tensor - expected))
        self.assertLessEqual(difference, 1e-10)

    def test_L2(self):
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor = regularizer.calculate(weights_tensor)

        difference = np.sum(np.abs(weights_tensor - np.ones(self.shape) * self.regularizer_strength))
        self.assertLessEqual(difference, 1e-10)

    def test_L1_with_optimizer(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Sgd(self.delta)
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(self.individual_delta, weights_tensor, np.ones(self.shape)*2)
        expected_result = np.ones(self.shape) * (1-(self.regularizer_strength*self.delta*self.individual_delta)-(self.delta*self.individual_delta) * 2)

        self.assertAlmostEqual(result[0, 0], expected_result[0, 0])
        self.assertAlmostEqual(result[0, 0]*self.shape[0]*self.shape[1], expected_result[0, 0]*self.shape[0]*self.shape[1])

    def test_L2_with_optimizer(self):
        weights_tensor = np.ones(self.shape)

        optimizer = Optimizers.Sgd(self.delta)
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(self.individual_delta, weights_tensor, np.ones(self.shape)*3)
        expected_result = np.ones(self.shape) * (1 - (self.regularizer_strength * self.delta * self.individual_delta) - (self.delta * self.individual_delta) * 3)

        self.assertAlmostEqual(result[0, 0], expected_result[0, 0])
        self.assertAlmostEqual(result[0, 0]*self.shape[0]*self.shape[1], expected_result[0, 0]*self.shape[0]*self.shape[1])


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10

        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_forward_trainTime(self):
        drop_layer = Dropout.Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 2)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean, 1. * self.input_size, places=1)

    def test_forward_testTime(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.phase = Base.Phase.test
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size)

    def test_backward(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)

        self.assertEqual(np.max(output), 1)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean, .5 * self.input_size, places=1)

class TestBatchNorm(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 200
        self.channels = 2
        self.input_shape = (self.channels, 3, 3)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(0)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T

        self.categories = 5
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        self.layers.append(SoftMax.SoftMax())

        self.plot_shape = (self.input_shape[1], self.input_shape[0] * np.prod(self.input_shape[2:]))

    @staticmethod
    def _channel_moments(tensor, channels):
        tensor = tensor.reshape(np.int(tensor.shape[1]/channels) * tensor.shape[0], channels)
        mean = np.mean(tensor, axis=0)
        var = np.var(tensor, axis=0)
        return mean, var

    def test_forward_shape(self):
        layer = BatchNormalization.BatchNormalization()
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward_shape_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward(self):
        layer = BatchNormalization.BatchNormalization()
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones(var.shape[0]))), 0)

    def test_forward_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor)
        mean, var = TestBatchNorm._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0)

    def test_forward_train_phase(self):
        layer = BatchNormalization.BatchNormalization()
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_train_phase_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor, self.channels)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_test_phase(self):
        layer = BatchNormalization.BatchNormalization()
        layer.forward(self.input_tensor)
        layer.phase = Base.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_forward_test_phase_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        layer.forward(self.input_tensor)
        layer.phase = Base.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_gradient(self):
        self.layers[0] = BatchNormalization.BatchNormalization()

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        description = 'Gradient_batch_norm_inputs'
        Helpers.plot_difference(self.plot, description, self.plot_shape, difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_gradient_weights(self):
        self.layers[0] = BatchNormalization.BatchNormalization()
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)

        description = 'Gradient_batch_norm_weights'
        Helpers.plot_difference(self.plot, description, (np.prod(self.input_shape[:-1]), self.channels), difference.reshape((np.prod(self.input_shape), 1)), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias(self):
        self.layers[0] = BatchNormalization.BatchNormalization()
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)

        description = 'Gradient_batch_norm_bias'
        #Helpers.plot_difference(self.plot, description, (np.prod(self.input_shape[:-1]), self.channels), difference.reshape((np.prod(self.input_shape), 1)), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        description = 'Gradient_batch_norm_convolutional_inputs'
        #Helpers.plot_difference(self.plot, description, self.plot_shape, difference, self.directory)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)

        description = 'Gradient_batch_norm_convolutional_weights'
        #Helpers.plot_difference(self.plot, description, (2, 1), difference.reshape((2, 1)), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)

        description = 'Gradient_batch_norm_convolutional_bias'
        #Helpers.plot_difference(self.plot, description, (2, 1), difference.reshape((2, 1)), self.directory)
        self.assertLessEqual(np.sum(difference), 1e-6)

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 13
        self.output_size = 5
        self.hidden_size = 7
        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

        self.categories = 4
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_forward_stateful(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size, self.batch_size)

        input_vector = np.random.rand(self.input_size, 1).T
        input_tensor = np.tile(input_vector, (2, 1))

        output_tensor = layer.forward(input_tensor)

        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)

    def test_forward_stateful_TBPTT(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        layer.toggle_memory()
        output_tensor_first = layer.forward(self.input_tensor)
        output_tensor_second = layer.forward(self.input_tensor)
        self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0)

    def test_backward_size(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        layer.delta = 0.1
        layer.set_optimizer(Optimizers.Sgd(1))
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.output_size, self.batch_size]).T
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN.RNN(self.input_size, self.hidden_size, self.categories, self.batch_size)
        layer.initialize(Initializers.Xavier(), Initializers.Xavier())
        layers.append(layer)
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = RNN.RNN(self.input_size, self.hidden_size, self.categories, self.batch_size)
        layer.initialize(Initializers.Xavier(), Initializers.Xavier())
        layers.append(layer)
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = RNN.RNN(100000, 100, 1, self.batch_size)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 13
        self.output_size = 5
        self.hidden_size = 7
        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

        self.categories = 4
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_forward_stateful(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size, self.batch_size)

        input_vector = np.random.rand(1, self.input_size)
        input_tensor = np.tile(input_vector, (2, 1))

        output_tensor = layer.forward(input_tensor)

        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)

    def test_forward_stateful_TBPTT(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        layer.toggle_memory()
        output_tensor_first = layer.forward(self.input_tensor)
        output_tensor_second = layer.forward(self.input_tensor)
        self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0)

    def test_backward_size(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size, self.batch_size)
        layer.delta = 0.1
        layer.set_optimizer(Optimizers.Sgd(1))
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.output_size, self.batch_size]).T
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.categories, self.batch_size)
        layer.initialize(Initializers.Xavier(), Initializers.Xavier())
        layers.append(layer)
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.categories, self.batch_size)
        layer.initialize(Initializers.Xavier(), Initializers.Xavier())
        layers.append(layer)
        layers.append(SoftMax.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = LSTM.LSTM(100000, 100, 1, self.batch_size)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)


class TestNeuralNetwork(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-4),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData()
        net.loss_layer = SoftMax.SoftMax()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)

        net.train(4000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_momentum(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.SgdWithMomentum(1e-4, 0.8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData()
        net.loss_layer = SoftMax.SoftMax()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Momentum.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Momentum, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999, 1e-8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData()
        net.loss_layer = SoftMax.SoftMax()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_ADAM.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using ADAM, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999, 1e-8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData()
        net.loss_layer = SoftMax.SoftMax()
        net.append_trainable_layer(BatchNormalization.BatchNormalization())
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Batchnorm.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999, 1e-8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData()
        net.loss_layer = SoftMax.SoftMax()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(Dropout.Dropout(0.3))

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Dropout.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)


class TestConvNet(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_digit_data(self):
        adam = Optimizers.Adam(5e-3, 0.98, 0.999, 1e-8)
        self._perform_test(adam, TestConvNet.iterations, 'ADAM', False, False)

    def test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999, 1e-8)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'L2_regularizer', False, False)

    def test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Optimizers.Adam(5e-3, 0.98, 0.999, 1e-8)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(sgd_with_l1, TestConvNet.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999, 1e-8)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Optimizers.Adam(1e-2, 0.98, 0.999, 1e-8)
        self._perform_test(adam, TestConvNet.iterations, 'Batch_norm', False, True)

    def test_all(self):
        sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999, 1e-8)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          Initializers.Xavier(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 150
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = SoftMax.SoftMax()

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization(1))

        cl_1 = Conv.Conv(input_image_shape, conv_stride_shape, convolution_shape, num_kernels)
        net.append_trainable_layer(cl_1)
        cl_1_output_shape = (*input_image_shape[1:], num_kernels)

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization(num_kernels))

        net.layers.append(ReLU.ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, np.int(fcl_1_input_size / 2))
        net.append_trainable_layer(fcl_1)

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization())

        if dropout:
            net.layers.append(Dropout.Dropout(0.3))

        net.layers.append(ReLU.ReLU())

        fcl_2 = FullyConnected.FullyConnected(np.int(fcl_1_input_size / 2), np.int(fcl_1_input_size / 3))
        net.append_trainable_layer(fcl_2)

        net.layers.append(ReLU.ReLU())

        fcl_3 = FullyConnected.FullyConnected(np.int(fcl_1_input_size / 3), categories)
        net.append_trainable_layer(fcl_3)

        net.train(iterations)
        if TestConvNet.plot:
            fig = plt.figure('Loss function for training a Convnet on the Digit dataset ' + description)
            plt.plot(net.loss, '-x')
            plt.ylim(ymin=0, ymax=450)
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description + ".pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using ' + description + ' we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        print('\nOn the UCI ML hand-written digits dataset using ' + description + ' we achieve an accuracy of: ' + str(accuracy * 100.) + '%')
        self.assertGreater(accuracy, 0.3)

if __name__ == "__main__":
    plot = True
    directory = 'plots'
    log = 'log.txt'
    with open(log, 'a') as f:
        print('Run ' + str(datetime.datetime.now()), file=f)
    if not os.path.exists(directory):
        os.makedirs(directory)
    TestNeuralNetwork.plot = plot
    TestConv.plot = plot
    TestPooling.plot = plot
    TestConvNet.plot = plot
    TestBatchNorm.plot = plot
    TestNeuralNetwork.directory = directory
    TestConv.directory = directory
    TestPooling.directory = directory
    TestConvNet.directory = directory
    TestBatchNorm.directory = directory
    TestNeuralNetwork.log = log
    TestConv.log = log
    unittest.main()
    with open(log, 'a') as f:
        print('Finished Tests' + str(datetime.datetime.now()), file=f)
