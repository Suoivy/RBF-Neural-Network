""""
Date:2017.4.28
Neural Network design homework
Using 3-layers-RBF NN to classification and regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

run_evaluation
evaluate model with evaluation dataset while training model.
visualize the evaluation loss with training iteration

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

def visualization(sample_x, sample_y):
    samp = plt.figure()
    ax = samp.add_subplot(1, 1, 1)
    ax.plot(sample_x, sample_y, 'k +')
    plt.show()

def compare_visual(sample_x, sample_y, predict_y):
    comp = plt.figure()
    comp_ax = comp.add_subplot(1, 1, 1)
    comp_ax.plot(sample_x, sample_y, 'k--')
    comp_ax.plot(sample_x, predict_y, 'b+')
    plt.draw()



# generate dataset to regression
# def gen_regress_data():


# main function
def main():
    ### Optimizers ###
    # BGD_optimizer
    # def BGD_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def BGD():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.BGD_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.2)

    # Momentum_optimizer
    # def Momentum_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def Momentum():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Momentum_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1, momentum_rate=0.9)

    # Nesterov Accelerated Gradient(NAG_optimizer)
    # def NAG_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9}):
    def NAG():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.NAG_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1, momentum_rate=0.9)

    # Adagrad_optimizer
    # def Adagrad_optimizer(self, param, hyper_param={'learn_rate': 0.01}):
    def Adagrad():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adagrad_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.1)

    # Adadelta_optimizer
    # def Adadelta_optimizer(self, param, hyper_param={'decay_rate': 0.9}):
    def Adadelta():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adadelta_optimizer, 0.01,
                                iteration, evaluate, decay_rate=0.9)

    # RMSProp
    # def RMSProp_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay_rate': 0.9}):
    def RMSProp():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.RMSProp_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.01, decay_rate=0.9)

    # RMSProp_with_Nesterov
    # def RMSProp_Nesterov_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'momentum_rate': 0.9, 'decay_rate': 0.9}):
    def RMSProp_Nesterov():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.RMSProp_Nesterov_optimizer,
                                0.01, iteration, evaluate, learn_rate=0.01, momentum_rate=0.9, decay_rate=0.9)

    # Adam
    # def Adam_optimizer(self, param, hyper_param={'learn_rate': 0.01, 'decay1_rate': 0.9, 'decay2_rate': 0.999}):
    def Adam():
        return regress.train(regress.L2_loss, regress.L2_loss_gradient, regress.Adam_optimizer, 0.01,
                                iteration, evaluate, learn_rate=0.01, decay1_rate=0.9, decay2_rate=0.999)

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
    eval_samples_x = np.arange(-4, 4, 0.08).reshape(1, -1)
    eval_samples_y = 1.1 * (1-eval_samples_x + 2 * np.power(eval_samples_x, 2)) * np.exp(-np.power(eval_samples_x, 2)/2)

    regress = RBFModel(train_samples_x, train_samples_y, 10)
    regress.set_evaluate_dataset(eval_samples_x, eval_samples_y)

    iteration = 500000
    evaluate = True  # Train model and evaluate with evaluate_samples simultaneously
    withcluster = True

    optimizer = {'BGD': BGD, 'Momentum': Momentum, 'NAG': NAG, 'Adagrad': Adagrad, 'Adadelta': Adadelta,
                 'RMSProp': RMSProp, 'RMSProp_Nesterov': RMSProp_Nesterov, 'Adam': Adam}
    activation = {'Gaussian': Gaussian, 'Reflected_sigmoid': Reflected_sigmoid}

    line = {'BGD': 'b-', 'Momentum': 'r-', 'NAG': 'g-', 'Adagrad': 'k-', 'Adadelta': 'y-', 'RMSProp': 'c-',
            'RMSProp_Nesterov': 'm-', 'Adam': 'k--'}

    if not os.path.exists('evaltrain-loss'):
        os.mkdir('evaltrain-loss')
    if not os.path.exists('evalmodels'):
        os.mkdir('evalmodels')

    for iscluster in {True, False}:
        withcluster = iscluster
        if withcluster:
            method = '-3Phase'
        else:
            method = '-Gradient'

        for act in activation:

            #plt.ion()
            opt_fig = plt.figure()
            opt_plt = opt_fig.add_subplot(1, 1, 1)
            opt_plt.set_ylim([0, 5])
            opt_plt.set_title(act)
            opt_fig.suptitle('Optimizers Comparision')
            losses = {}
            times = {}

            for opt in optimizer:

                losses[opt] = np.array([100])
                times[opt] = 0

                fig = plt.figure()
                if evaluate:
                    ax = fig.add_subplot(2, 1, 1)
                    val = fig.add_subplot(2, 1, 2)
                    val.set_title('evaluation-loss', fontsize='small')
                else:
                    ax = fig.add_subplot(1, 1, 1)

                ax.set_title(opt + ' - ' +  act + method +'-train-loss')
                ax.set_ylim([0, 5])

                for iter in range(1):

                    activation[act]()

                    if withcluster:
                        regress.cluster_center()

                    begin = time.time()

                    if evaluate:
                        loss, eval_loss, predict = optimizer[opt]()
                        val.plot(eval_loss)
                        # compare_visual(eval_samples_x, eval_samples_y, predict)
                    else:
                        loss = optimizer[opt]()

                    timecost = time.time() - begin

                    #results, eval_loss = regress.evaluate(eval_samples_x, eval_samples_y)

                    if loss[len(loss) - 1] < losses[opt][len(losses[opt]) - 1]:
                        losses[opt] = loss
                        times[opt] = timecost

                        regress.save_model('evalmodels/' + act + '-' + opt + method + '-model')

                    color = 0.2 * iter + 0.1
                    ax.plot(loss, color=(color, color, color))
                    plt.draw()
                ax.annotate('loss=%.3f,time=%.2f'%(losses[opt][len(losses[opt]) - 1],times[opt]),xy=(iteration-1,losses[opt][len(losses[opt]) - 1]), xytext=(-150,25), textcoords='offset points')
                plt.savefig('evaltrain-loss/' + act + '-' + opt + method  + '.png', dpi=400, bbox_inches='tight')

                np.save('evaltrain-loss/' + act + '-' + opt + method, losses[opt])
                opt_plt.plot(losses[opt], line[opt], label=opt + '   loss=%.4f,time=%.2f'%(losses[opt][len(losses[opt]) - 1],times[opt]))
                plt.draw()
                opt_plt.legend(loc='best', fontsize='small')
            opt_fig.savefig('evaltrain-loss/' + act + method + '-Optimizers.png', dpi=400, bbox_inches='tight')

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
