import random
import numpy as np
import scipy.ndimage
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import odl.contrib.tensorflow
import dicom as dc
from scipy.misc import imresize
import tensorflow as tf
import util as ut

from forward_models import ct
from forward_models import denoising

from data_pips import LUNA
from networks import improved_binary_classifier
from networks import UNet
from networks import fully_convolutional
from data_pips import BSDS


# This class provides methods necessary
class generic_framework(object):
    model_name = 'no_model'

    # set the noise level used for experiments
    noise_level = 0.01

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return UNet(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    # the parameter alpha determines the weight of the adv net compared to l2 loss.
    # 0 corresponds to pure adversarial loss, 1 to pure l2 loss
    def __init__(self, alpha):
        self.alpha = alpha
        self.data_pip = self.get_Data_pip()
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()

        name = platform.node()
        path_prefix = ''
        if name == 'LAPTOP-E6AJ1CPF':
            path_prefix=''
        elif name == 'motel':
            path_prefix='/local/scratch/public/sl767/OptimalTransportDenoising/'
        self.path = path_prefix+'Saves/{}/{}/{}/Weight_{}/'.format(self.model.name, self.noise_level, self.model_name, self.alpha)


        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # generate needed folder structure
        self.generate_folders()

    # method to generate training data given the current model type
    def generate_training_data(self, batch_size, training_data = True):
        y = np.empty((batch_size, self.measurement_space[0], self.measurement_space[1], self.colors), dtype='float32')
        x_true = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')
        fbp = np.empty((batch_size, self.image_space[0], self.image_space[1], self.colors), dtype='float32')

        for i in range(batch_size):
            if training_data:
                image = self.data_pip.load_data(training_data=True)
            else:
                image = self.data_pip.load_data(training_data=False)
            for k in range(self.data_pip.colors):
                data = self.model.forward_operator(image[...,k])

                # add white Gaussian noise
                noisy_data = data + np.random.normal(size= self.measurement_space) * self.noise_level

                fbp [i, ..., k] = self.model.inverse(noisy_data)
                x_true[i, ..., k] = image[...,k]
                y[i, ..., k] = noisy_data
        return y, x_true, fbp

    # puts in place the folders needed to save the results obtained with the current model
    def generate_folders(self):
        paths = {}
        paths['Image Folder'] = self.path + 'Images'
        paths['Saves Folder'] = self.path + 'Data'
        paths['Logging Folder'] = self.path + 'Logs'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    def save(self, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=global_step)
        print('Progress saved')

    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    ### generic method for subclasses
    def deploy(self, true, guess, measurement):
        pass

# The postprocessing framework with unsupervised adversarial loss
class postprocessing(generic_framework):
    model_name = 'PostProcessing'

    # The batch size
    batch_size = 16
    # weight of soft relaxation regulariser adversarial net
    lmb = 10
    # default_adv_steps
    def_adv_steps = 8
    # default_gen_steps
    def_gen_steps = 1

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return UNet(size=size, colors=colors)

    def get_Data_pip(self):
        return BSDS()

    def get_model(self, size):
        return denoising(size=size)

    def get_adversarial_network(self):
        return improved_binary_classifier(size=self.image_size, colors=self.colors)

    def reconstruction_model(self, fbp, measurement):
        return self.network.net(fbp)

    def loss_model(self, measurement, reconstruction, truth):
        transport_loss = self.model.tensorflow_operator(reconstruction) - measurement
        return tf.reduce_mean(tf.reduce_sum(tf.square(transport_loss), axis=(1, 2, 3)))

    def set_adv_steps(self, amount = None):
        if amount == None:
            return self.def_adv_steps
        else:
            return amount

    def set_gen_steps(self, amount = None):
        if amount == None:
            return self.def_gen_steps
        else:
            return amount


    def __init__(self, alpha, learning_rate):
        # call superclass init
        super(postprocessing, self).__init__(alpha=alpha)
        adversarial_net = self.get_adversarial_network()

        # set learning rate
        self.learning_rate = learning_rate

        # set training parameters
        self.adv_steps = self.set_adv_steps()
        self.gen_steps = self.set_gen_steps()

        ### generator training
        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.guess = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                dtype=tf.float32)
        self.measurement = tf.placeholder(shape=[None, self.measurement_space[0],
                                                 self.measurement_space[1], self.data_pip.colors],
                                dtype=tf.float32)
        self.alpha_num = tf.placeholder(dtype=tf.float32)
        # network output
        with tf.variable_scope('Forward_model'):
            self.out = self.reconstruction_model(self.guess, self.measurement)
        # compute loss
        # transport loss: L2 loss squared
        with tf.variable_scope('adversarial_loss'):
            self.adv = tf.reduce_mean(adversarial_net.net(self.out))
            self.l2_loss = self.loss_model(measurement=self.measurement, reconstruction=self.out, truth=self.true)
            self.loss = (1-self.alpha_num) * self.adv + self.alpha_num * self.l2_loss
            self.normed_wass = self.adv - tf.reduce_mean(adversarial_net.net(self.true))


        # track L2 loss to ground truth for quality control
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.out- self.true), axis=(1, 2, 3))))

        # optimizer for Reconstruction network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # apply gradient clipping
        plain_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = plain_optimizer.compute_gradients(self.loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                 scope='Forward_model'))
        clipped_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        self.optimizer = plain_optimizer.apply_gradients(clipped_grad, global_step=self.global_step)

        # logging tools
        with tf.name_scope('Generator_training'):

            # get the batch size
            batch_s = tf.cast(tf.shape(self.true)[0], tf.float32)

            # loss analysis
            tf.summary.scalar('Overall_Loss', self.loss)
            tf.summary.scalar('Wasserstein_Loss', self.normed_wass)
            tf.summary.scalar('L2_Loss', self.l2_loss)

            tf.summary.image('Ground_Truth', self.true, max_outputs=2)
            tf.summary.image('FBP', ut.cut_image_tf(self.guess), max_outputs=2)
            tf.summary.image('Reconstruction', ut.cut_image_tf(self.out), max_outputs=2)

            gradients_distributional = tf.gradients(batch_s *self.adv, self.out)[0]
            gradients_transport = tf.gradients(batch_s *self.l2_loss, self.out)[0]
            tf.summary.scalar('Wasserstein_Loss_Grad', ut.tf_l2_norm(gradients_distributional))
            tf.summary.scalar('Transport_Loss_Grad', ut.tf_l2_norm(gradients_transport))

            # quality analysis
            tf.summary.scalar('Quality', self.quality)

        ### adversarial network training

        self.random_uint = tf.placeholder(shape=[None], dtype=tf.float32)
        self.ground_truth = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.network_guess = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                dtype=tf.float32)

        with tf.variable_scope('adversarial_loss'):
            self.net_was = adversarial_net.net(self.network_guess)
            self.truth_was = adversarial_net.net(self.ground_truth)
            # Wasserstein loss
            self.wasserstein_loss = tf.reduce_mean(self.truth_was - self.net_was)

            # intermediate point
            random_uint_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.random_uint, axis=1), axis=1), axis=1)
            self.inter = tf.multiply(self.network_guess, random_uint_exp) + \
                         tf.multiply(self.ground_truth, 1 - random_uint_exp)
            self.inter_was = adversarial_net.net(self.inter)
            # calculate derivative at intermediate point
            self.gradient_was = tf.gradients(self.inter_was, self.inter)[0]
            # ensure gradient norm bounds
            self.regulariser_was = tf.reduce_mean(tf.square(tf.nn.relu(tf.sqrt(
                tf.reduce_sum(tf.square(self.gradient_was), axis=(1, 2, 3))) - 1)))
            self.loss_was = self.wasserstein_loss + self.lmb * self.regulariser_was

        self.optimizer_adv = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_was,
                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adversarial_loss'))

        with tf.name_scope('Adversarial_training'):
            tf.summary.scalar('Data_Loss', self.wasserstein_loss)
            tf.summary.scalar('Regularizer', self.regulariser_was)
            tf.summary.scalar('Overall_Loss', self.loss_was)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def train_adversarial(self, steps):
        for k in range(steps):
            measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=True)
            out = self.sess.run(self.out, feed_dict={self.true: x_true, self.measurement: measurement,
                                                     self.guess: guess, self.alpha_num: self.alpha})
            epsilon = np.random.uniform(size=(self.batch_size))
            self.sess.run(self.optimizer_adv, feed_dict={self.random_uint: epsilon, self.ground_truth: x_true,
                                                         self.network_guess: out})

    def train_generator(self, steps, pretrain=False):
        if pretrain:
            alpha = 1
        else:
            alpha = self.alpha
        for k in range(steps):
            measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=True)
            self.sess.run(self.optimizer, feed_dict={self.true: x_true, self.measurement: measurement,
                                                     self.guess: guess, self.alpha_num: alpha})

    def evaluate(self):
        # get test data
        measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=False)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=(self.batch_size))
        out = self.sess.run(self.out, feed_dict={self.true: x_true, self.measurement: measurement,
                                                 self.guess: guess, self.alpha_num: self.alpha})

        iteration, adv_loss, \
        trans_loss, loss_was, \
        summary, quality = self.sess.run([self.global_step, self.adv,
                                          self.l2_loss, self.loss_was, self.merged, self.quality],
                                         feed_dict={self.random_uint: epsilon, self.ground_truth: x_true,
                                                    self.network_guess: out, self.true: x_true,
                                                    self.measurement: measurement,
                                                    self.guess: guess, self.alpha_num: self.alpha})
        print('Iteration: {}, mu: {}, Quality: {}, Wass. Dis.: {}, '
              'Transp. Loss: {}, Wass. Loss.:{}'.format(iteration, self.alpha, quality, loss_was,
                                                        trans_loss, adv_loss))
        self.writer.add_summary(summary, iteration)

    def train(self, steps):
        for k in range(steps):
            self.train_adversarial(self.adv_steps)
            self.train_generator(self.gen_steps)
            if k % 20 == 0:
                self.evaluate()

        self.save(self.global_step)

    def pretrain(self, steps):
        for k in range(steps):
            self.train_generator(1, pretrain=True)
            if k%20 == 0:
                self.evaluate()
        self.save(self.global_step)

    # estimate good regularisation parameter alpha in unsupervised case. Heuristic: Ground truth is a stable point,
    # so alpha = 1 / [1+ 2 ||A^t (Ax-y)||]
    def find_reg_parameter(self):
        measurement, x_true, guess = self.generate_training_data(32, training_data=True)
        Ax_y = self.model.forward_operator(x_true) - measurement
        data_er = ut.l2_norm(self.model.adjoint(Ax_y))
        return 1/(1+2*data_er)

    # estimate good regularisation parameter in supervised case. Heuristic: Reconstruction reaches level beta of
    # the error of the naive reconstruction and is a stable point. Hence
    # alpha = 1/ [1+ 2 beta ||x_fbp - x_true||]
    def find_reg_parameter_supervised(self, beta = 0.2):
        measurement, x_true, guess = self.generate_training_data(32, training_data=True)
        C = beta * ut.l2_norm(x_true - guess)
        return 1/(1+2*C)

# Iterative scheme for reconstruction
class iterative_recon(postprocessing):
    model_name = 'Iterative_Scheme'

    def get_network(self, size, colors):
        return fully_convolutional(size, colors)

    def reconstruction_model(self, fbp, measurement):
        x = fbp
        for k in range(4):
            # get gradient of data term
            grad = self.model.tensorflow_adjoint_operator(self.model.tensorflow_operator(x) - measurement)
            # network with gradient of data term and current guess as input
            with tf.variable_scope('Iteration_' + str(k)):
                x = self.network.net(tf.concat((grad, x), axis=3))
        return x

# Supervised postprocessing, leveraged with L2 distance to ground truth
class postprocessing_supervised(postprocessing):
    model_name = 'Postprocessing_Supervised'
    def loss_model(self, measurement, reconstruction, truth):
        return tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction-truth), axis=(1, 2, 3)))

# Supervised postprocessing, leveraged with L2 distance to ground truth
class iterative_recon_supervised(iterative_recon):
    model_name = 'Iterative_Scheme_Supervised'
    def loss_model(self, measurement, reconstruction, truth):
        return tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction - truth), axis=(1, 2, 3)))

# TV reconstruction
class total_variation(generic_framework):
    model_name = 'TV'

    # TV hyperparameters
    noise_level = 0.01
    def_lambda = 0.0013

    def __init__(self):
        # call superclass init
        super(total_variation, self).__init__()
        self.space = self.model.get_odl_space()
        self.operator = self.model.get_odl_operator()
        self.range = self.operator.range

    def tv_reconstruction(self, y, param=def_lambda):
        # the operators
        gradients = odl.Gradient(self.space, method='forward')
        broad_op = odl.BroadcastOperator(self.operator, gradients)
        # define empty functional to fit the chambolle_pock framework
        g = odl.solvers.ZeroFunctional(broad_op.domain)

        # the norms
        l1_norm = param * odl.solvers.L1Norm(gradients.range)
        l2_norm_squared = odl.solvers.L2NormSquared(self.range).translated(y)
        functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

        # Find parameters
        op_norm = 1.1 * odl.power_method_opnorm(broad_op)
        tau = 10.0 / op_norm
        sigma = 0.1 / op_norm
        niter = 500

        # find starting point
        x = self.range.element(self.model.inverse(y))

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x

    def find_TV_lambda(self, lmd):
        amount_test_images = 32
        y, true, cor = self.generate_training_data(amount_test_images)
        for l in lmd:
            error = np.zeros(amount_test_images)
            or_error = np.zeros(amount_test_images)
            for k in range(amount_test_images):
                recon = self.tv_reconstruction(y[k, ..., 0], l)
                error[k] = np.sum(np.square(recon - true[k, ..., 0]))
                or_error[k] = np.sum(np.square(cor[k, ..., 0] - true[k, ..., 0]))
            total_e = np.mean(np.sqrt(error))
            total_o = np.mean(np.sqrt(or_error))
            print('Lambda: ' + str(l) + ', MSE: ' + str(total_e) + ', OriginalError: ' + str(total_o))


