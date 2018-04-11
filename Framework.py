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
from networks import binary_classifier
from networks import UNet
from networks import fully_convolutional
from data_pips import BSDS


# This class provides methods necessary
class generic_framework(object):
    model_name = 'no_model'
    experiment_name = 'default_experiment'

    # set the noise level used for experiments
    noise_level = 0.01

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return binary_classifier(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)


    def __init__(self):
        self.data_pip = self.get_Data_pip()
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.path = 'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name, self.model_name, self.experiment_name)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()

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

    # visualizes the quality of the current method
    def visualize(self, true, fbp, guess, name):
        quality = np.average(np.sqrt(np.sum(np.square(true - guess), axis=(1, 2, 3))))
        print('Quality of reconstructed image: ' + str(quality))
        if self.colors == 1:
            t = true[-1,...,0]
            g = guess[-1, ...,0]
            p = fbp[-1, ...,0]
        else:
            t = true[-1,...]
            g = guess[-1, ...]
            p = fbp[-1, ...]
        plt.figure()
        plt.subplot(131)
        plt.imshow(ut.cut_image(t))
        plt.axis('off')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(ut.cut_image(p))
        plt.axis('off')
        plt.title('PseudoInverse')
        plt.suptitle('L2 :' + str(quality))
        plt.subplot(133)
        plt.imshow(ut.cut_image(g))
        plt.title('Reconstruction')
        plt.axis('off')
        plt.savefig(self.path + name + '.png')
        plt.close()

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

# Framework for postprocessing
class postprocessing(generic_framework):
    model_name = 'PostProcessing'

    # learning rate for Adams
    learning_rate = 0.001
    # The batch size
    batch_size = 64

    # methods to define the models used in framework
    def get_network(self, size, colors):
        return UNet(size=size, colors=colors)

    def get_Data_pip(self):
        return LUNA()

    def get_model(self, size):
        return ct(size=size)

    def __init__(self):
        # call superclass init
        super(postprocessing, self).__init__()

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors], dtype=tf.float32)
        # network output
        self.out = self.network.net(self.y)
        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))
        # optimizer
        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss', self.loss)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def log(self, x, y):
        summary, step = self.sess.run([self.merged, self.global_step],
                                      feed_dict={self.true : x,
                                                 self.y : y})
        self.writer.add_summary(summary, step)

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true : x_true,
                                                    self.y : fbp})
            if k%50 == 0:
                iteration, loss = self.sess.run([self.global_step, self.loss], feed_dict={self.true : x_true,
                                                    self.y : fbp})
                print('Iteration: ' + str(iteration) + ', MSE: ' +str(loss))

                # logging has to be adopted
                self.log(x_true,fbp)
                output = self.sess.run(self.out, feed_dict={self.true : x_true,
                                                    self.y : fbp})
                self.visualize(x_true, fbp, output, 'Iteration_{}'.format(iteration))
        self.save(self.global_step)

    def evaluate(self):
        y, x_true, fbp = self.generate_training_data(self.batch_size)

# implementation of iterative scheme from Jonas and Ozans paper
class iterative_scheme(generic_framework):
    model_name = 'Learned_gradient_descent'

    # hyperparameters
    iterations = 3
    learning_rate = 0.001
    # The batch size
    batch_size = 32

    def get_network(self, size, colors):
        return fully_convolutional(size=size, colors=colors)

    def __init__(self):
        # call superclass init
        super(iterative_scheme, self).__init__()

        # set placeholder for input and correct output
        self.true = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.guess = tf.placeholder(shape=[None, self.image_space[0], self.image_space[1], self.data_pip.colors],
                                   dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.measurement_space[0], self.measurement_space[1], self.data_pip.colors],
                                dtype=tf.float32)


        # network output - iterative scheme
        x = self.guess
        for i in range(self.iterations):
            measurement = self.model.tensorflow_operator(x)
            g_x = self.model.tensorflow_adjoint_operator(self.y - measurement)
            tf.summary.image('Data_gradient', g_x, max_outputs=1)
            tf.summary.scalar('Data_gradient_Norm', tf.norm(g_x))
            # network input
            net_input = tf.concat([x, g_x], axis=3)

            # use the network model defined in
            x_update = self.network.net(net_input)
            tf.summary.scalar('x_update', tf.norm(x_update))
            x = x - x_update
            tf.summary.image('Current_Guess', x, max_outputs=1)
        self.out = x

        # compute loss
        data_mismatch = tf.square(self.out - self.true)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(data_mismatch, axis=(1, 2, 3))))
        # optimizer
        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step)
        # logging tools
        tf.summary.scalar('Loss', self.loss)

        # set up the logger
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/',
                                            self.sess.graph)

        # initialize Variables
        tf.global_variables_initializer().run()

        # load existing saves
        self.load()

    def train(self, steps):
        for k in range(steps):
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            self.sess.run(self.optimizer, feed_dict={self.true : x_true,
                                                    self.y : y,
                                                    self.guess : fbp})
            if k%10 == 0:
                summary, iteration, loss, output = self.sess.run([self.merged, self.global_step, self.loss, self.out],
                                                         feed_dict={self.true : x_true,
                                                                    self.y : y,
                                                                    self.guess : fbp})
                print('Iteration: ' + str(iteration) + ', MSE: ' +str(loss) + ', Original Error: '
                      + str(ut.l2_norm(x_true - fbp)))

                self.writer.add_summary(summary, iteration)
                self.visualize(x_true, fbp, output, 'Iteration_{}'.format(iteration))

        self.save(self.global_step)

class postprocessing_adversarial(generic_framework):
    model_name = 'PostProcessing'

    # learning rate for Adams
    learning_rate = 0.0002
    # learning rate adversarial
    learning_rate_adv = 0.0002
    # weight adv net
    trans_loss_weight = 1
    # The batch size
    batch_size = 16
    # weight of soft relaxation regulariser adversarial net
    lmb= 10
    # default_adv_steps
    def_adv_steps = 12
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
        return binary_classifier(size=self.image_size, colors=self.colors)

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


    def __init__(self):
        # call superclass init
        super(postprocessing_adversarial, self).__init__()
        adversarial_net = self.get_adversarial_network()

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
        # network output
        with tf.variable_scope('Forward_model'):
            self.out = self.network.net(self.guess)
        # compute loss
        # transport loss: L2 loss squared
        with tf.variable_scope('adversarial_loss'):
            self.adv = tf.reduce_mean(adversarial_net.net(self.out))
            transport_loss = self.model.tensorflow_operator(self.out) - self.measurement
            self.trans_loss = tf.reduce_mean(tf.reduce_sum(tf.square(transport_loss), axis=(1, 2, 3)))
            self.loss = self.adv + self.trans_loss_weight * self.trans_loss
            self.normed_wass = self.adv - tf.reduce_mean(adversarial_net.net(self.true))


        # track L2 loss to ground truth for quality control
        self.quality = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.out- self.true), axis=(1, 2, 3))))

        # optimizer for Wasserstein network
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step,
                                                                             var_list=tf.get_collection(
                                                                                 tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                 scope='Forward_model'))

        # logging tools
        with tf.name_scope('Generator_training'):

            # get the batch size
            batch_s = tf.cast(tf.shape(self.true)[0], tf.float32)

            # loss analysis
            tf.summary.scalar('Overall_Loss', self.loss)
            tf.summary.scalar('Distributional_Loss', self.normed_wass)
            tf.summary.scalar('Transport_Loss',
                                                             self.trans_loss_weight * self.trans_loss)
            gradients_distributional = tf.gradients(batch_s *self.adv, self.out)[0]
            gradients_transport = tf.gradients(batch_s *self.trans_loss_weight * self.trans_loss, self.out)[0]
            tf.summary.scalar('Distributional_Loss_Grad', ut.tf_l2_norm(gradients_distributional))
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
                                                     self.guess: guess})
            epsilon = np.random.uniform(size=(self.batch_size))
            self.sess.run(self.optimizer_adv, feed_dict={self.random_uint: epsilon, self.ground_truth: x_true,
                                                         self.network_guess: out})

    def train_generator(self, steps):
        for k in range(steps):
            measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=True)
            self.sess.run(self.optimizer, feed_dict={self.true: x_true, self.measurement: measurement,
                                                     self.guess: guess})


    def train(self, steps):
        for k in range(steps):
            self.train_adversarial(self.adv_steps)
            self.train_generator(self.gen_steps)
            if k % 50 == 0:
                # get test data
                measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=False)
                # generate random distribution for rays
                epsilon = np.random.uniform(size=(self.batch_size))
                out = self.sess.run(self.out, feed_dict={self.true: x_true, self.measurement: measurement,
                                                   self.guess: guess})

                iteration, adv_loss, \
                trans_loss, loss_was, \
                summary, quality = self.sess.run([self.global_step, self.adv,
                                        self.trans_loss, self.loss_was, self.merged, self.quality],
                                    feed_dict={self.random_uint: epsilon, self.ground_truth: x_true,
                                    self.network_guess: out, self.true: x_true, self.measurement: measurement,
                                       self.guess: guess})
                print('Iteration: {}, mu: {}, Quality: {}, Wass. Dis.: {}, '
                      'Transp. Loss: {}, Wass. Loss.:{}'.format(iteration, self.trans_loss_weight, quality, loss_was,
                                                                trans_loss, adv_loss))
                self.writer.add_summary(summary, iteration)

                self.visualize(x_true, guess, out, 'Images/Iteration_{}'.format(iteration))
        self.save(self.global_step)

    # estimate good regularisation parameter lmb = 'trans_loss_weight'. Works for denoising only!
    # heurisitic: grad_y D(y) + lmb ||x - y||^2_2 |_{y = x_true} = 0
    # implies: 1 = lmb 2 ||x-x_true||_2
    def find_reg_parameter(self):
        # estimate ||x-x_true||_2
        # get test data
        measurement, x_true, guess = self.generate_training_data(self.batch_size, training_data=True)
        # mismatch for denoising only!!!
        mismatch = ut.l2_norm(guess - x_true)
        return 1/(2*mismatch)



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


