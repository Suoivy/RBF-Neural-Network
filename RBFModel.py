""""
Date:2017.4.27
Neural Network design homework
Using 3-phases-RBF NN to regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

"""

import numpy as np
import time
from sklearn.cluster import KMeans
from scipy.spatial import distance

import matplotlib.pyplot as plt


class RBFModel():
    # config model value
    """
     Input: data_input(array(IN_value_num,data_num))
            data_output(array(OUT_value_num,data_num)
            hidden_unit_number(N)
    """

    def __init__(self, data_input = np.array([[0],[0]]), data_output = np.array([[0],[0]]), hidden_units_number = 1):

        self.input = data_input
        self.output = data_output
        self.data_number = self.input.shape[1]

        self.input_units_number = self.input.shape[0]
        self.hidden_units_number = hidden_units_number
        self.output_units_number = self.output.shape[0]

        self.center, self.spread, self.weight, self.bias = np.array([[], [], [], []])
        self.radial_basis_input = np.array([])

        self.radial_basis_function = self.sigmoid_activation
        self.radial_basis_function_gradient = self.sigmoid_gradient
        self.output_activation = self.none_activation
        self.output_activation_gradient = self.none_gradient

        self.loss_function = self.L2_loss
        self.loss_function_gradient = self.L2_loss_gradient

        self.optimizer = self.BGD_optimizer

        self.eval_input = np.array([])
        self.eval_output = np.array([])

    # Initialize values  #3-Layers BP
    """
     input: activation_function(function) activation_gradient(function)
    """

    def initialize_parameters(self, radial_basis_function, radial_basis_function_gradient,
                              output_activation, output_activation_gradient):

        self.center = 8 * np.random.rand(self.hidden_units_number, self.input_units_number) - 4
        self.spread = 0.2 * np.random.rand(self.hidden_units_number, 1) + 0.1
        self.weight = 0.2 * np.random.rand(self.output_units_number, self.hidden_units_number) - 0.1
        self.bias = 0.2 * np.random.rand(self.output_units_number, 1) - 0.1

        self.set_activation(radial_basis_function, radial_basis_function_gradient, output_activation,
                            output_activation_gradient)

    # Set activation function
    def set_activation(self, radial_basis_function, radial_basis_function_gradient, output_activation,
                       output_activation_gradient):

        self.radial_basis_function = radial_basis_function
        self.radial_basis_function_gradient = radial_basis_function_gradient
        self.output_activation = output_activation
        self.output_activation_gradient = output_activation_gradient

    # Cluster hidden unit center using Kmeans
    def cluster_center(self):

        estimator = KMeans(n_clusters = self.hidden_units_number)
        estimator.fit(self.input.T)
        self.center = estimator.cluster_centers_
        alldist = distance.cdist(self.center, self.center, 'euclidean')
        dist = np.where(alldist == 0, alldist.max()+1, alldist)
        self.spread = dist.min(axis=1).reshape(self.hidden_units_number, 1)

    # Set evaluate dataset
    def set_evaluate_dataset(self, samp_input, samp_output):
        self.eval_input = samp_input
        self.eval_output = samp_output

    # train model
    """
     Input: loss_function(function)
            loss_gradient(function)
            optimizer(function)
            learn_error(float64)
            max_iteration(int64)
    """

    def train(self, loss_function, loss_gradient, optimizer, learn_error, iteration, evaluate=False,
              **option_hyper_param):

        self.loss_function = loss_function
        self.loss_function_gradient = loss_gradient
        self.optimizer = optimizer

        train_losses = []
        eval_losses = []
        param = []
        elapsed_time = 0

        # plt.ion()
        # train_fig = plt.figure()
        # loss_plt = train_fig.add_subplot(1, 1, 1)
        # loss_plt.set_title('train_loss')
        # loss_plt.set_xlable('train_iter')
        # loss_plt.set_ylable('loss')

        for iter in range(iteration):

            last_time = time.time()

            # Back propagation and Optimizer
            loss = self.optimizer(param, option_hyper_param)

            iter_time = time.time() - last_time
            elapsed_time = elapsed_time + iter_time
            train_losses.append(loss)
            # loss_plt.plot(loss, 'b-')
            # plt.draw()

            if evaluate:
                results, eval_loss = self.evaluate(self.eval_input, self.eval_output)
                eval_losses.append(eval_loss)

            if iter % 100 == 0:
                print('train iteration:%d, train loss:%f, iter time:%f, elapsed time:%f' % (
                    iter, loss, iter_time, elapsed_time))

            if loss < learn_error:
                break

        # plt.ioff()
        # plt.show()
        if evaluate:
            return train_losses, eval_losses, results
        else:
            return train_losses

    # Forward graph configure network output
    def _forward(self, input_=np.array([])):

        if len(input_) == 0:
            input_ = self.input

        self.distance = distance.cdist(input_.T, self.center, 'euclidean').T
        self.radial_basis_input = self.distance / self.spread
        hidden_output = self.radial_basis_function(self.radial_basis_input)
        network_output = self.output_activation(self.weight.dot(hidden_output) + self.bias)

        return hidden_output, network_output

    # Back graph configure network gradient
    def _backward(self, hidden_output, network_output):

        hidden_gradient = self.loss_function_gradient(network_output, self.output) * self.output_activation_gradient(
            network_output)
        input_gradient = self.weight.T.dot(hidden_gradient) * self.radial_basis_function_gradient(self.radial_basis_input)

        delta_weight = hidden_gradient.dot(hidden_output.T) / self.data_number
        delta_bias = hidden_gradient.dot(np.ones((self.data_number, 1))) / self.data_number
        ldist = np.tile(self.input, (self.hidden_units_number, 1)) - self.center.reshape(self.input_units_number*self.hidden_units_number, 1)  # shape: i*h,n
        ldist_gradient = (input_gradient / self.spread) / (-self.distance)
        delta_center = (ldist * ldist_gradient.repeat(self.input_units_number, 0)).sum(axis=1).reshape(self.hidden_units_number, self.input_units_number) / self.data_number
        delta_spread = (input_gradient * (-self.distance * np.power(self.spread, -2))).dot(np.ones((self.data_number, 1))) / self.data_number

        return delta_center, delta_spread, delta_weight, delta_bias

    # evaluate model
    def evaluate(self, input, output, ):

        hidden_output, network_output = self._forward(input)

        loss = self.loss_function(network_output, output)

        return network_output, loss

    # predict
    def predict(self, input):

        output = self._forward(input)

        return output

    ### Optimizer Functions ###
    # Batch Gradient Descent (BGD)
    def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

        # Initialize variables
        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('BGD_optimizer have no "learn_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        extended_delta = learn_rate * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # Momentum
    def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

        # Initialize variables
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Momentum_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('Momentum_optimizer have no "momentum_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        extended_delta = param[0]
        extended_delta = momentum_rate * extended_delta + learn_rate * extended_gradient
        param[0] = extended_delta
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # Nesterov Accelerated Gradient(NAG)
    def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):

        # Initialize variables
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('NAG_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('NAG_optimizer have no "momentum_rate" hyper-parameter')
            return

        # Forward propagation
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)
        extended_delta = param[0]
        self.center, self.spread, self.weight, self.bias = self.split_weights(
            extended_variables - momentum_rate * extended_delta)
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)

        # Updata variables
        extended_delta = momentum_rate * extended_delta + learn_rate * extended_gradient
        param[0] = extended_delta
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # Adagrad
    def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):

        # Initialize variables
        delta = 10e-7
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Adagrad_optimizer have no "learn_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_gradient = accumulated_gradient + extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # Adadelta
    def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):

        # Initialize variables
        delta = 10e-7
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient
            param.append(np.zeros(1))  # accumulated square delta

        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('Adadelta_optimizer have no "decay_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_delta = param[1]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        extended_delta = np.sqrt(accumulated_delta + delta) / np.sqrt(accumulated_gradient + delta) * extended_gradient
        accumulated_delta = decay_rate * accumulated_delta + (1 - decay_rate) * extended_delta * extended_delta
        param[0] = accumulated_gradient
        param[1] = accumulated_delta
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # RMSProp
    def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):

        # Initialize variables
        delta = 10e-6
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('RMSProp_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('RMSProp_optimizer have no "decay_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # RMSProp_with_Nesterov
    def RMSProp_Nesterov_optimizer(self, param,
                                   hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):

        # Initialize variables
        delta = 10e-6
        if len(param) == 0:
            param.append(np.zeros(1))  # last delta_weights and delta_biases
            param.append(np.zeros(1))  # accumulated square gradient

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay_rate = hyper_param['decay_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "decay_rate" hyper-parameter')
            return
        try:
            momentum_rate = hyper_param['momentum_rate']
        except:
            print('RMSProp_Nesterov_optimizer have no "momentum_rate" hyper-parameter')
            return

        # Forward propagation
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)
        extended_delta = param[0]
        self.center, self.spread, self.weight, self.bias = self.split_weights(
            extended_variables - momentum_rate * extended_delta)
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        accumulated_gradient = param[1]
        accumulated_gradient = decay_rate * accumulated_gradient + (
                1 - decay_rate) * extended_gradient * extended_gradient
        param[1] = accumulated_gradient
        extended_delta = learn_rate / np.sqrt(accumulated_gradient + delta) * extended_gradient
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    # Adam
    def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):

        # Initialize variables
        delta = 10e-8
        if len(param) == 0:
            param.append(np.zeros(1))  # accumulated gradient
            param.append(np.zeros(1))  # accumulated square gradient
            param.append(0)  # train steps

        try:
            learn_rate = hyper_param['learn_rate']
        except:
            print('Adam_optimizer have no "learn_rate" hyper-parameter')
            return
        try:
            decay1_rate = hyper_param['decay1_rate']
        except:
            print('Adam_optimizer have no "decay1_rate" hyper-parameter')
            return
        try:
            decay2_rate = hyper_param['decay2_rate']
        except:
            print('Adam_optimizer have no "decay2_rate" hyper-parameter')
            return

        # Forward propagation
        hidden_output_, network_output_ = self._forward()

        # Loss function
        loss_ = self.loss_function(network_output_, self.output)

        # Backward propagation
        delta_center, delta_spread, delta_weight, delta_bias = self._backward(hidden_output_, network_output_)
        extended_gradient = self.extend_variables(delta_center, delta_spread, delta_weight, delta_bias)
        extended_variables = self.extend_variables(self.center, self.spread, self.weight, self.bias)

        # Updata variables
        accumulated_gradient = param[0]
        accumulated_square_gradient = param[1]
        step = param[2] + 1
        accumulated_gradient = decay1_rate * accumulated_gradient + (1 - decay1_rate) * extended_gradient
        accumulated_square_gradient = decay2_rate * accumulated_square_gradient + (
                1 - decay2_rate) * extended_gradient * extended_gradient
        param[0] = accumulated_gradient
        param[1] = accumulated_square_gradient
        param[2] = step + 1
        extended_moment1 = accumulated_gradient / (1 - np.power(decay1_rate, step))
        extended_moment2 = accumulated_square_gradient / (1 - np.power(decay2_rate, step))
        extended_delta = learn_rate * extended_moment1 / (np.sqrt(extended_moment2) + delta)
        extended_variables = extended_variables - extended_delta

        self.center, self.spread, self.weight, self.bias = self.split_weights(extended_variables)

        return loss_

    ### Activation Functions ###
    # sigmoid
    def sigmoid_activation(self, input_):

        output_ = 1 / (1 + np.exp(-input_))

        return output_

    def sigmoid_gradient(self, input_):

        output_ = input_ * (1 - input_)

        return output_

    # tanh (Bipolar sigmoid)
    def tanh_activation(self, input_):

        output_ = (1 - np.exp(-input_)) / (1 + np.exp(-input_))

        return output_

    def tanh_gradient(self, input_):

        output_ = 0.5 * (1 - input_ * input_)

        return output_

    # ReLU
    def ReLU_activation(self, input_):

        output_ = np.where(input_ < 0, 0, input_)

        return output_

    def ReLU_gradient(self, input_):

        output_ = np.where(input_ > 0, 1, 0)

        return output_

    # Softmax
    def softmax_activation(self, input_):

        output_ = np.exp(input_ - input_.max(axis=0)) / np.sum(np.exp(input_ - input_.max(axis=0)), axis=0)

        return output_

    def softmax_gradient(self, input_):

        output_ = input_ * (self.output - input_)

        return output_

    # None Activation
    def none_activation(self, input_):

        output_ = input_

        return output_

    def none_gradient(self, input_):

        output_ = 1

        return output_

    # RBF Gaussian Function
    def Gaussian_basis(self, input_):

        output_ = np.exp(-np.power(input_, 2))

        return output_

    def Gaussian_gradient(self, input_):

        output_ = -2 * input_ * np.exp(-np.power(input_, 2))

        return output_

    # RBF Reflected sigmoid Function
    def Reflected_sigmoid_basis(self, input_):

        output_ = 1 / (1 + np.exp(np.power(input_, 2)))

        return output_

    def Reflected_sigmoid_gradient(self, input_):

        output_ = -2 * input_ * np.exp(np.power(input_, 2)) / np.power(1 + np.exp(np.power(input_, 2)), 2)

        return output_


    ### Loss Functions ###
    # output_: Network predict output   output: dataset output
    # L2_loss (RMSE)
    def L2_loss(self, output_, output):

        loss = np.sum(0.5 * np.power(output_ - output, 2))

        return loss

    def L2_loss_gradient(self, output_, output):

        loss_gradient = output_ - output

        return loss_gradient

    # cross_entropy loss

    def cross_entropy_loss(self, output_, output):

        loss = -np.sum(np.log(output_) * output)

        return loss

    def cross_entropy_gradient(self, output_, output):

        loss_gradient = -output / output_

        return loss_gradient

    # sigmoid_cross_entropy loss

    # softmax cross_entropy loss

    ### utilities ###
    def extend_variables(self, center_, spread_, weight_, bias_):

        variables = np.concatenate(
            (center_.reshape(-1), spread_.reshape(-1), weight_.reshape(-1), bias_.reshape(-1)))

        return variables

    def split_weights(self, variables):

        center_mark = self.input_units_number * self.hidden_units_number
        spread_mark = center_mark + self.hidden_units_number
        weight_mark = spread_mark + self.hidden_units_number * self.output_units_number
        center = variables[: center_mark].copy()
        center = center.reshape(self.hidden_units_number, self.input_units_number)
        spread = variables[center_mark: spread_mark].copy()
        spread = spread.reshape(self.hidden_units_number, 1)
        weight = variables[spread_mark: weight_mark].copy()
        weight = weight.reshape(self.output_units_number, self.hidden_units_number)
        bias = variables[weight_mark:].copy()
        bias = bias.reshape(self.output_units_number, 1)

        return center, spread, weight, bias

    def save_model(self, path='model'):
        np.save(path, [self.center, self.spread, self.weight, self.bias])

    def load_model(self, path):
        parameters = np.load(path)

        self.center = parameters[0]
        self.spread = parameters[1]
        self.weight = parameters[2]
        self.bias = parameters[3]

        self.input_units_number = self.center.shape[1]
        self.hidden_units_number = self.center.shape[0]
        self.output_units_number = self.weight.shape[0]
