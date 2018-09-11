import random
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import odl.contrib.tensorflow
from scipy.misc import imresize
import util as ut

class forward_model(object):
    name = 'default'

    def get_image_size(self):
        pass

    def get_measurement_size(self):
        pass

    def forward_operator(self, image):
        pass

    def adjoint(self, measurment):
        pass

    def inverse(self, measurement):
        pass

    def tensorflow_operator(self, input):
        pass

    def tensorflow_adjoint_operator(self, input):
        pass

    def get_odl_space(self):
        pass

    def get_odl_operator(self):
        pass


class ct(forward_model):
    name = 'Computed_Tomography'

    def __init__(self, size):
        self.space = odl.uniform_discr([-64, -64], [64, 64], [size[0], size[1]],
                                  dtype='float32')

        geometry = odl.tomo.parallel_beam_geometry(self.space, num_angles=30)
        op = odl.tomo.RayTransform(self.space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(op)
        self.operator = (1 / opnorm) * op
        self.fbp = (opnorm) * odl.tomo.fbp_op(op)
        self.adjoint_operator = (1 / opnorm)*op.adjoint

        # the spaces
        self.meas_space = self.operator.range.shape
        self.image_space = (128,128)

        # Create tensorflow layer from odl operator
        self.ray_transform = odl.contrib.tensorflow.as_tensorflow_layer(self.operator,
                                                                  'RayTransform')
        self.ray_transform_adj = odl.contrib.tensorflow.as_tensorflow_layer(self.adjoint_operator, 'AdjRayTransform')

    def get_image_size(self):
        return self.space.shape

    def get_measurement_size(self):
        return self.operator.range.shape

    def forward_operator(self, image):
        shape = image.shape
        if len(shape) == 4:
            result = np.zeros(shape=[shape[0], self.meas_space[0], self.meas_space[1],1])
            for k in range(shape[0]):
                input = self.space.element(image[k,...,0])
                result[k,...,0] = self.operator(input)
            return result
        else:
            input = self.space.element(image)
            return self.operator(input)

    def adjoint(self, measurement):
        shape = measurement.shape
        if len(shape) == 4:
            result = np.zeros(shape=[shape[0], self.image_space[0], self.image_space[1], 1])
            for k in range(shape[0]):
                input = self.operator.range.element(measurement[k, ..., 0])
                result[k, ..., 0] = self.adjoint_operator(input)
            return result
        else:
            input = self.operator.range.element(measurement)
            return self.adjoint_operator(input)

    def inverse(self, measurement):
        shape = measurement.shape
        if len(shape) == 4:
            result = np.zeros(shape=[shape[0], self.image_space[0], self.image_space[1], 1])
            for k in range(shape[0]):
                input = self.operator.range.element(measurement[k, ..., 0])
                result[k, ..., 0] = self.fbp(input)
            return result
        else:
            input = self.operator.range.element(measurement)
            return self.fbp(input)


    def tensorflow_operator(self, input):
        return self.ray_transform(input)

    def tensorflow_adjoint_operator(self, input):
        return self.ray_transform_adj(input)

    def get_odl_space(self):
        return self.space

    def get_odl_operator(self):
        return self.operator

class denoising(forward_model):
    name = 'Denoising'

    def __init__(self, size):
        self.size = size
        self.space = odl.uniform_discr([-64, -64], [64, 64], [size[0], size[1]],
                                  dtype='float32')
        self.operator = odl.IdentityOperator(self.space)

    def get_image_size(self):
        return self.size

    def get_measurement_size(self):
        return self.size

    def forward_operator(self, image):
        return image

    def adjoint(self, measurement):
        return measurement

    def inverse(self, measurement):
        return measurement

    def tensorflow_operator(self, input):
        return input

    def tensorflow_adjoint_operator(self, input):
        return input

    def get_odl_space(self):
        return self.space

    def get_odl_operator(self):
        return self.operator
