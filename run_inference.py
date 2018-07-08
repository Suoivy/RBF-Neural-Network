""""
Date:2017.4.28
Neural Network design homework
Using 3-layers-RBF NN to classification and regression
author:Suo Chuanzhe
email: suo_ivy@foxmail.com

run_inference
load the training model and infer the output
visualize the training and prediction data.

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

def compare_visual(sample_x, sample_y, predict_y,save=False, path='inference/inference-compare.png',
                   title='inference comparison', loss=0):
    comp = plt.figure()
    comp.suptitle(title)
    comp_ax = comp.add_subplot(1, 1, 1)
    #pred_ax = comp.add_subplot(2, 1, 2)
    comp_ax.set_title('Black:Groud truth; Blue:Predict; loss= %.5f' % loss, fontsize='small')
    comp_ax.plot(sample_x, sample_y, 'k.')
    comp_ax.plot(sample_x, predict_y, 'b +')
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    #plt.show()



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
        regress.set_activation(regress.Gaussian_basis, regress.Gaussian_gradient,
                                      regress.none_activation, regress.none_gradient)

        # Reflected_sigmoid

    def Reflected_sigmoid():
        regress.set_activation(regress.Reflected_sigmoid_basis, regress.Reflected_sigmoid_gradient,
                                      regress.none_activation, regress.none_gradient)

    eval_samples_x = np.arange(-4, 4, 0.08).reshape(1,-1)
    eval_samples_y = 1.1 * (1-eval_samples_x + 2 * np.power(eval_samples_x, 2)) * np.exp(-np.power(eval_samples_x, 2)/2)
    visualization(eval_samples_x, eval_samples_y)

    regress = RBFModel()

    evaluate = True  # Train model and evaluate with evaluate_samples simultaneously
    withcluster = True

    optimizer = {'BGD': BGD, 'Momentum': Momentum, 'NAG': NAG, 'Adagrad': Adagrad, 'Adadelta': Adadelta,
                 'RMSProp': RMSProp, 'RMSProp_Nesterov': RMSProp_Nesterov, 'Adam': Adam}
    activation = {'Gaussian': Gaussian, 'Reflected_sigmoid': Reflected_sigmoid}

    line = {'BGD': 'b-', 'Momentum': 'r-', 'NAG': 'g-', 'Adagrad': 'k-', 'Adadelta': 'y-', 'RMSProp': 'c-',
            'RMSProp_Nesterov': 'm-', 'Adam': 'k--'}

    if not os.path.exists('evalinference'):
        os.mkdir('evalinference')
    if not os.path.exists('models'):
        os.mkdir('models')

    filelist = []
    for file in os.listdir(os.getcwd() + '/evalmodels'):
        file_path = os.path.join(os.getcwd() + '/evalmodels', file)
        if os.path.isdir(file_path):
            pass
        else:
            name, extension = os.path.splitext(file)
            if extension == '.npy':
                filelist.append(file_path)

    for file in filelist:
        filename = os.path.basename(file)

        act_function = filename.split('-')[0]
        opti = filename.split('-')[1]
        method = filename.split('-')[2]

        regress.load_model(file)
        activation[act_function]()

        predict_y, loss = regress.evaluate(eval_samples_x, eval_samples_y)
        compare_visual(eval_samples_x, eval_samples_y, predict_y, save=True,
                       path='evalinference/' + str(act_function) + '-' + str(opti) + '-' + str(method) + '-inference.png',
                       title=str(act_function) + '-' + str(opti) + '-' + str(method) + '-comparison', loss=loss)

        print(loss)



if __name__ == "__main__":
    main()
