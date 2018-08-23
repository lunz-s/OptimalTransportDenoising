import tensorflow as tf
from util import lrelu
import util as ut

class binary_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv1')
        # begin convolutional/pooling architecture
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # image size is now size/2
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        # image size is now size/4
        conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        # image size is now size/8
        conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv5')
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
        # image size is now size/16
        conv6 = tf.layers.conv2d(inputs=pool5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv6')

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[0]* self.size[1]/(16*16))
        dimensionality = finishing_size * 128
        reshaped = tf.reshape(conv6, [-1, dimensionality])

        # dense layer for classification
        dense = tf.layers.dense(inputs = reshaped, units = 256, activation=lrelu, reuse=self.reuse, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=self.reuse, name='dense2')

        # change reuse variable for next call of network method
        self.reuse = True

        # Output network results
        return output

class improved_binary_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        # convolutional network for feature extraction
        conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv3', strides=2)
        # image size is now size/2
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv4', strides=2)
        # image size is now size/4
        conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv5', strides=2)
        # image size is now size/8
        conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=lrelu, reuse=self.reuse, name='conv6', strides=2)

        # reshape for classification - assumes image size is multiple of 32
        finishing_size = int(self.size[0]* self.size[1]/(16*16))
        dimensionality = finishing_size * 128
        reshaped = tf.reshape(conv6, [-1, dimensionality])

        # dense layer for classification
        dense = tf.layers.dense(inputs = reshaped, units = 256, activation=lrelu, reuse=self.reuse, name='dense1')
        output = tf.layers.dense(inputs=dense, units=1, reuse=self.reuse, name='dense2')

        # change reuse variable for next call of network method
        self.reuse = True

        # Output network results
        return output


class UNet(object):
    def __init__(self, size, colors, parameter_sharing = True):
        self.colors = colors
        self.size = size
        self.parameter_sharing = parameter_sharing
        self.used = False

    def raw_net(self, input, reuse):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                      padding="same", name='conv1', reuse=reuse, activation=lrelu)
        # 64
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                      padding="same", name='conv2', reuse=reuse, activation=lrelu)
        # 32
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                      padding="same", name='conv3', reuse=reuse, activation=lrelu)
        # 64
        conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv1',
                                           reuse=reuse, activation=lrelu)
        concat1 = tf.concat([conv4, pool1], axis= 3)
        # 128
        conv5 =  tf.layers.conv2d_transpose(inputs=concat1, filters=32, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv2',
                                            reuse=reuse, activation=lrelu)
        concat2 = tf.concat([conv5, input], axis= 3)
        output = tf.layers.conv2d(inputs=concat2, filters=self.colors, kernel_size=[5, 5],
                                  padding="same",name='deconv3',
                                  reuse=reuse,  activation=lrelu)
        return output

    def net(self, input):
        output = self.raw_net(input, reuse=self.used)
        if self.parameter_sharing:
            self.used = True
        return output

class fully_convolutional(UNet):

    def raw_net(self, input, reuse):
        # 128
        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv1', reuse=reuse, activation=lrelu)
        # 64
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5],
                                 padding="same", name='conv2', reuse=reuse, activation=lrelu)
        # 32
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3],
                                 padding="same", name='conv3', reuse=reuse, activation=lrelu)
        # 64
        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[5, 5],
                                 padding="same", name='conv4',reuse=reuse, activation=lrelu)
        output = tf.layers.conv2d(inputs=conv4, filters=self.colors, kernel_size=[5, 5],
                                  padding="same", name='conv6', reuse=reuse)
        return output

### resnet architectures
def apply_conv(x, filters=32, kernel_size=3):
    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            activation=lrelu)

def resblock(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = apply_conv(update, filters=filters)

        skip = tf.layers.conv2d(x, filters=filters, kernel_size=1, padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        return skip + update

def meanpool(x):
    with tf.name_scope('meanpool'):
        x = tf.identity(x)
        return tf.add_n([x[:,::2,::2,:], x[:,1::2,::2,:],
                         x[:,::2,1::2,:], x[:,1::2,1::2,:]]) / 4.

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        return tf.depth_to_space(x, 2)

class resnet_classifier(object):
    def __init__(self, size, colors):
        self.size = size
        self.reuse = False
        self.colors = colors

    def net(self, input):
        with tf.variable_scope('discriminator', reuse=self.reuse):
            with tf.name_scope('pre_process'):
                x = apply_conv(input, filters=64, kernel_size=3)

            with tf.name_scope('x1'):
                x = resblock(x, 64)

            with tf.name_scope('x2'):
                x = resblock(meanpool(x), filters=64) # 1/2

            with tf.name_scope('x3'):
                x = resblock(meanpool(x), filters=128) # 1/4

            with tf.name_scope('x4'):
                x = resblock(meanpool(x), filters=256) # 1/8

            with tf.name_scope('x5'):
                x = resblock(meanpool(x), filters=256) # 1/16

            with tf.name_scope('post_process'):
                flat = tf.contrib.layers.flatten(x)
                flat = tf.layers.dense(flat, 1)

            # change reuse variable for next call of network method
            self.reuse = True

            return flat

class Res_UNet(object):
    def __init__(self, size, colors, parameter_sharing = True):
        self.colors = colors
        self.size = size
        self.parameter_sharing = parameter_sharing
        self.used = False

    def raw_net(self, input, reuse):
        with tf.variable_scope('UNet', reuse=reuse):
            prepro = resblock(input,32)
            # 128
            conv1 =  resblock(prepro, filters=32)
            # 64
            pool1 = meanpool(conv1)
            conv2 = resblock(pool1, filters=64)
            # 32
            pool2 = meanpool(conv2)
            conv3= resblock(pool2, filters=128)
            # 16
            pool3 = meanpool(conv3)
            conv4 = resblock(pool3, filters=128)
            # 32
            up4 = upsample(conv4)
            conv5 = resblock(up4, filters= 128)
            concat1 = tf.concat([conv5, conv3], axis= 3)
            # 64
            up5 = upsample(concat1)
            conv6 = resblock(up5, filters= 64)
            concat2 = tf.concat([conv6, conv2], axis= 3)
            # 128
            up6 = upsample(concat2)
            conv7 = resblock(up6, filters= 32)
            concat3 = tf.concat([conv7, conv1], axis= 3)

            output = resblock(concat3, filters=self.colors)
            return output

    def net(self, input):
        output = self.raw_net(input, reuse=self.used)
        if self.parameter_sharing:
            self.used = True
        return output
