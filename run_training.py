""""
Date:2017.4.27
Neural Network design homework
Using 3-layers-RBF NN to classification and regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

run_training
using different activation function, loss function and optimizer to train 3-layers-RBF classify network

"""

import numpy as np
import os
import time
from RBFModel import RBFModel

import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 50000

# generate dataset to classify
def gen_regression_data(numbers):

    noise_var = 0.1
    noise = noise_var*np.random.rand(1, numbers)
    sample_input = 8 * np.random.rand(1, numbers) - 4
    sample_output = 1.1 * (1-sample_input + 2 * np.power(sample_input, 2)) * np.exp(-np.power(sample_input, 2)/2) + noise

    return sample_input, sample_output

def visualization(input_x, input_y):
    samp = plt.figure()
    ax = samp.add_subplot(1, 1, 1)
    ax.plot(input_x, input_y, 'k +')
    plt.show()



# generate dataset to regression
# def gen_regress_data():


# main function
def main():
    ### Optimizers ###
    # BGD_optimizer
    # def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def BGD():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.BGD_optimizer, 0.01,
                                iteration, learn_rate=0.2)

    # Momentum_optimizer
    # def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def Momentum():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Momentum_optimizer, 0.01,
                                iteration, learn_rate=0.1, momentum_rate=0.9)

    # Nesterov Accelerated Gradient(NAG_optimizer)
    # def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def NAG():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.NAG_optimizer, 0.01,
                                iteration, learn_rate=0.1, momentum_rate=0.9)

    # Adagrad_optimizer
    # def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def Adagrad():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adagrad_optimizer, 0.01,
                                iteration, learn_rate=0.1)

    # Adadelta_optimizer
    # def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):
    def Adadelta():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adadelta_optimizer, 0.01,
                                iteration, decay_rate=0.9)

    # RMSProp
    # def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):
    def RMSProp():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.RMSProp_optimizer, 0.01,
                                iteration, learn_rate=0.01, decay_rate=0.9)

    # RMSProp_with_Nesterov
    # def RMSProp_Nesterov_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):
    def RMSProp_Nesterov():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.RMSProp_Nesterov_optimizer,
                                0.01, iteration, learn_rate=0.01, momentum_rate=0.9, decay_rate=0.9)

    # Adam
    # def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):
    def Adam():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adam_optimizer, 0.01,
                                iteration, learn_rate=0.01, decay1_rate=0.9, decay2_rate=0.999)

    ### Activation ###
    # Gaussian
    def Gaussian():
        regress.initialize_parameters(regress.Gaussian_basis, regress.Gaussian_gradient,
                                         regress.none_activation, regress.none_gradient)

    # Reflected_sigmoid
    def Reflected_sigmoid():
        regress.initialize_parameters(regress.Reflected_sigmoid_basis, regress.Reflected_sigmoid_gradient,
                                         regress.none_activation, regress.none_gradient)

    train_samples_x, train_samples_y = gen_regression_data(100)
    visualization(train_samples_x, train_samples_y)

    regress = RBFModel(train_samples_x, train_samples_y, 10)

    iteration = 500000
    withcluster = False

    optimizer = {'BGD': BGD, 'Momentum': Momentum, 'NAG': NAG, 'Adagrad': Adagrad, 'Adadelta': Adadelta,
                 'RMSProp': RMSProp, 'RMSProp_Nesterov': RMSProp_Nesterov, 'Adam': Adam}
    activation = {'Gaussian': Gaussian, 'Reflected_sigmoid': Reflected_sigmoid}

    line = {'BGD': 'b-', 'Momentum': 'r-', 'NAG': 'g-', 'Adagrad': 'k-', 'Adadelta': 'y-', 'RMSProp': 'c-',
            'RMSProp_Nesterov': 'm-', 'Adam': 'k--'}

    if not os.path.exists('train-loss'):
        os.mkdir('train-loss')
    if not os.path.exists('models'):
        os.mkdir('models')

    for iscluster in {True, False}:
        withcluster = iscluster
        if withcluster:
            method = '-3Phase'
        else:
            method = '-Gradient'

        for act in activation:

            #plt.ion()
            opt_fig = plt.figure(1)
            opt_plt = opt_fig.add_subplot(1, 1, 1)
            opt_plt.set_ylim([0, 1])
            opt_plt.set_title(act + '-' + method)
            opt_fig.suptitle('Optimizers Comparision')
            losses = {}
            times = {}

            for opt in optimizer:

                losses[opt] = np.array([100])
                times[opt] = 0

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set_ylim([0, 1])
                ax.set_title(opt + ' - ' + act + method + '-train-loss')

                for iter in range(3):

                    activation[act]()

                    if withcluster:
                        regress.cluster_center()

                    begin = time.time()
                    loss = optimizer[opt]()
                    timecost = time.time() - begin

                    if loss[len(loss) - 1] < losses[opt][len(losses[opt]) - 1]:
                        losses[opt] = loss
                        times[opt] = timecost

                        regress.save_model('models/' + act + '-' + opt + method + '-model')

                    color = 0.2 * iter + 0.1
                    ax.plot(loss, color=(color, color, color))
                    plt.draw()
                ax.annotate('loss=%.3f,time=%.2f' % (losses[opt][len(losses[opt]) - 1], times[opt]),
                            xy=(iteration - 1, losses[opt][len(losses[opt]) - 1]), xytext=(-150, 25),
                            textcoords='offset points')
                plt.savefig('train-loss/' + act + '-' + opt + method + '.png', dpi=400, bbox_inches='tight')

                np.save('train-loss/' + act + '-' + opt + method, losses[opt])
                opt_plt.plot(losses[opt], line[opt],
                             label=opt + '   loss=%.4f,time=%.2f' % (losses[opt][len(losses[opt]) - 1], times[opt]))
                plt.draw()
                opt_plt.legend(loc='best', fontsize='small')
            opt_fig.savefig('train-loss/' + act + method + '-Optimizers.png', dpi=400, bbox_inches='tight')

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
