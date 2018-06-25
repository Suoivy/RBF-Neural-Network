""""
Date:2017.4.20
Neural Network design homework
Using 3-layers-BP NN to classification and regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

run_training
using different activation function, loss function and optimizer to train 3-layers-BP classify network

"""

import numpy as np
import os
import time
from RBFModel import BPModel

import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 50000

# generate dataset to classify
def gen_classify_data(numbers):
    sample_input = (np.random.rand(2, numbers) - 0.5) * 4
    sample_output = np.array([[], [], []])

    for i in range(numbers):
        sample = sample_input[:, i]
        x = sample[0]
        y = sample[1]

        if ((x > -1) & (x < 1)) == 1:
            if ((y > x / 2 + 1 / 2) & (y < 1)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [1], [0]]), axis=1)
            elif ((y < -0.5) & (y > -1.5)) == 1:
                sample_output = np.append(sample_output, np.array([[0], [0], [1]]), axis=1)
            else:
                sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)
        else:
            sample_output = np.append(sample_output, np.array([[1], [0], [0]]), axis=1)

    return sample_input, sample_output

def visualization(input_x, input_y):
    samp = plt.figure()
    type = np.where(input_y.T == input_y.max())[1] +1
    plt.scatter(input_x[0], input_x[1], c = 'g', linewidths = type)
    plt.show()


# generate dataset to regression
# def gen_regress_data():


# main function
def main():
    ### Optimizers ###
    # BGD_optimizer
    # def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def BGD():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.BGD_optimizer, 0.01,
                                iteration, learn_rate=0.2)

    # Momentum_optimizer
    # def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def Momentum():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Momentum_optimizer, 0.01,
                                iteration, learn_rate=0.1, momentum_rate=0.9)

    # Nesterov Accelerated Gradient(NAG_optimizer)
    # def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def NAG():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.NAG_optimizer, 0.01,
                                iteration, learn_rate=0.1, momentum_rate=0.9)

    # Adagrad_optimizer
    # def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def Adagrad():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adagrad_optimizer, 0.01,
                                iteration, learn_rate=0.1)

    # Adadelta_optimizer
    # def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):
    def Adadelta():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adadelta_optimizer, 0.01,
                                iteration, decay_rate=0.9)

    # RMSProp
    # def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):
    def RMSProp():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_optimizer, 0.01,
                                iteration, learn_rate=0.01, decay_rate=0.9)

    # RMSProp_with_Nesterov
    # def RMSProp_Nesterov_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):
    def RMSProp_Nesterov():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.RMSProp_Nesterov_optimizer,
                                0.01, iteration, learn_rate=0.01, momentum_rate=0.9, decay_rate=0.9)

    # Adam
    # def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):
    def Adam():
        return classifier.train(classifier.L2_loss, classifier.L2_loss_gradient, classifier.Adam_optimizer, 0.01,
                                iteration, learn_rate=0.01, decay1_rate=0.9, decay2_rate=0.999)

    ### Activation ###
    # sigmoid
    def sigmoid():
        classifier.initialize_parameters(classifier.sigmoid_activation, classifier.sigmoid_gradient,
                                         classifier.sigmoid_activation, classifier.sigmoid_gradient)

    # tanh
    def tanh():
        classifier.initialize_parameters(classifier.tanh_activation, classifier.tanh_gradient,
                                         classifier.tanh_activation, classifier.tanh_gradient)

    # ReLU
    def ReLU():
        classifier.initialize_parameters(classifier.ReLU_activation, classifier.ReLU_gradient,
                                         classifier.ReLU_activation, classifier.ReLU_gradient)

    train_samples_x, train_samples_y = gen_classify_data(200)
    visualization(train_samples_x, train_samples_y)

    classifier = BPModel(train_samples_x, train_samples_y, 10)

    iteration = 5000

    optimizer = {'BGD': BGD, 'Momentum': Momentum, 'NAG': NAG, 'Adagrad': Adagrad, 'Adadelta': Adadelta,
                 'RMSProp': RMSProp, 'RMSProp_Nesterov': RMSProp_Nesterov, 'Adam': Adam}
    activation = {'sigmoid': sigmoid, 'tanh': tanh, 'ReLU': ReLU}

    line = {'BGD': 'b-', 'Momentum': 'r-', 'NAG': 'g-', 'Adagrad': 'k-', 'Adadelta': 'y-', 'RMSProp': 'c-',
            'RMSProp_Nesterov': 'm-', 'Adam': 'k--'}

    if not os.path.exists('train-loss'):
        os.mkdir('train-loss')
    if not os.path.exists('models'):
        os.mkdir('models')

    for act in activation:

        plt.ion()
        opt_fig = plt.figure()
        opt_plt = opt_fig.add_subplot(1, 1, 1)
        opt_plt.set_ylim([0, 40])
        opt_plt.set_title(act)
        opt_fig.suptitle('Optimizers Comparision')
        losses = {}
        times = {}

        for opt in optimizer:

            losses[opt] = np.array([100])
            times[opt] = 0

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(opt + ' - ' + act+'train-loss')
            ax.set_ylim([0, 40])

            for iter in range(1):

                activation[act]()

                begin = time.time()
                loss = optimizer[opt]()
                timecost = time.time() - begin

                if loss[len(loss) - 1] < losses[opt][len(losses[opt]) - 1]:
                    losses[opt] = loss
                    times[opt] = timecost

                    classifier.save_model('models/' + act + '-' +opt + '-model')

                color = 0.2 * iter + 0.1
                ax.plot(loss, color=(color, color, color))
                plt.draw()
            ax.annotate('loss=%.3f,time=%.2f'%(losses[opt][len(losses[opt]) - 1],times[opt]),xy=(iteration-1,losses[opt][len(losses[opt]) - 1]), xytext=(-150,25), textcoords='offset points')
            plt.savefig('train-loss/' + act + '-' + opt + '.png', dpi=400, bbox_inches='tight')

            np.save('results/'+act+opt, losses[opt])
            opt_plt.plot(losses[opt], line[opt], label=opt + '   loss=%.4f,time=%.2f'%(losses[opt][len(losses[opt]) - 1],times[opt]))
            plt.draw()
            opt_plt.legend(loc='best', fontsize='small')
        opt_fig.savefig('train-loss/' + act + '-Optimizers.png', dpi=400, bbox_inches='tight')

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
