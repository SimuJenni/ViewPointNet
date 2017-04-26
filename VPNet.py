import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import lrelu, up_conv2d, merge
import numpy as np

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 512]
REPEATS = [1, 1, 2, 2, 2]


def vpnet_argscope(activation=lrelu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                   w_reg=0.0001, fix_bn=False):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or test mode
        center: Whether to use centering in batchnorm
        w_reg: Parameter for weight-decay

    Returns:
        An argscope
    """
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.95,
        'epsilon': 0.001,
        'center': center,
    }
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    with slim.arg_scope([slim.fully_connected],
                                        weights_initializer=trunc_normal(0.005)):
                        return arg_sc


class VPNet:
    def __init__(self, num_layers, batch_size, tag='default', vgg_discriminator=True, fix_bn=False):
        """Initialises a VPNet using the provided parameters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
            vgg_discriminator: Whether to use VGG-A instead of AlexNet in the discriminator
        """
        self.name = 'VPNet_{}'.format(tag)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vgg_discriminator = vgg_discriminator
        self.discriminator = VanillaDisc(num_layers=num_layers, fix_bn=fix_bn)

    def net(self, im1, im2, vp1, vp2, reuse=None, training=True):
        """Builds the full VPNet architecture with the given inputs.

        Args:
            img: Placeholder for input images
            reuse: Whether to reuse already defined variables.
            training: Whether in train or test mode

        Returns:
            dec_im: The autoencoded image
            dec_gen: The reconstructed image from cartoon and edge inputs
            disc_out: The discriminator output
            enc_im: Encoding of the image
            gen_enc: Output of the generator
        """
        enc_im1 = self.encoder(im1, reuse=reuse, training=training)
        enc_im2 = self.encoder(im2, reuse=True, training=training)

        enc_im1 = self.hidden_transform(enc_im1, vp2-vp1, reuse=reuse, training=training)
        enc_im2 = self.hidden_transform(enc_im2, vp1-vp2, reuse=True, training=training)

        dec_im1 = self.decoder(enc_im1, reuse=reuse, training=training)
        dec_im2 = self.decoder(enc_im2, reuse=True, training=training)

        enc_dec1 = self.encoder(dec_im1, reuse=True, training=training) #TODO: Maybe set training to False in one usage
        enc_dec2 = self.encoder(dec_im2, reuse=True, training=training)

        enc_dec1 = self.hidden_transform(enc_dec1, vp1-vp2, reuse=True, training=training)  #TODO: Maybe set training to False in one usage
        enc_dec2 = self.hidden_transform(enc_dec2, vp2-vp1, reuse=True, training=training)

        dec_ed1 = self.decoder(enc_dec1, reuse=True, training=training) #TODO: Maybe set training to False in one usage
        dec_ed2 = self.decoder(enc_dec2, reuse=True, training=training)

        # Build input for discriminator
        disc_in_fake = merge(dec_im1, dec_im2, dim=0)
        disc_in_real = merge(dec_ed1, dec_ed2, dim=0)

        disc_out_fake, _ = self.discriminator.discriminate(disc_in_fake, reuse=reuse, training=training)
        disc_out_real, _ = self.discriminator.discriminate(disc_in_real, reuse=True, training=training)

        class_in_real = merge(dec_ed1, dec_ed2, dim=0)
        class_in_fake = merge(dec_im2, dec_im1, dim=0)

        class_out_real = self.discriminator.classify(class_in_real, 3, reuse=reuse, training=training)
        class_out_fake = self.discriminator.classify(class_in_fake, 3, reuse=True, training=training)

        return dec_im1, dec_im2, dec_ed1, dec_ed2, class_out_real, class_out_fake, disc_out_real, disc_out_fake

    def vp_label(self, vp1, vp2):
        return merge(vp1, vp2, dim=0)

    def disc_labels(self):
        labels_real = tf.ones(shape=(self.batch_size*2, ), dtype=tf.int32)
        labels_fake = tf.zeros(shape=(self.batch_size*2, ), dtype=tf.int32)
        return slim.one_hot_encoding(labels_real, 2), slim.one_hot_encoding(labels_fake, 2)

    def build_classifier(self, img, num_classes, reuse=None, training=True):
        """Builds a classifier on top either the encoder, generator or discriminator trained in the AEGAN.

        Args:
            img: Input image
            num_classes: Number of output classes
            reuse: Whether to reuse already defined variables.
            training: Whether in train or test mode

        Returns:
            Output logits from the classifier
        """
        _, model = self.discriminator.discriminate(img, reuse=reuse, training=training, with_fc=False)
        model = self.discriminator.classify(model, num_classes, reuse=reuse, training=training)
        return model

    def encoder(self, net, reuse=None, training=True):
        """Builds an encoder of the given inputs.

        Args:
            net: Input to the encoder (image)
            reuse: Whether to reuse already defined variables
            training: Whether in train or test mode.

        Returns:
            Encoding of the input image.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(padding='SAME', training=training, center=False)):
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))

                return net

    def decoder(self, net, reuse=None, training=True):
        """Builds a decoder on top of net.

        Args:
            net: Input to the decoder (output of encoder)
            reuse: Whether to reuse already defined variables
            training: Whether in train or test mode.

        Returns:
            Decoded image with 3 channels.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(padding='SAME', training=training, center=False)):
                for l in range(0, self.num_layers-1):
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - 2], scope='deconv_{}'.format(l))

                in_shape = net.get_shape().as_list()
                net = tf.image.resize_images(net, (2*in_shape[1], 2*in_shape[2]), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, num_outputs=3, stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net

    def hidden_transform(self, net, d_vp, reuse=None, training=True):
        with tf.variable_scope('transformer', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(training=training, center=False)):
                in_shape = net.get_shape().as_list()
                net = slim.flatten(net)
                net = merge(net, d_vp, dim=1)
                net = slim.fully_connected(net, np.prod(in_shape[1:]), scope='fc_transform')
                net = tf.reshape(net, in_shape)
                return net

    def hidden_transform2(self, net, d_vp, reuse=None, training=True):
        with tf.variable_scope('transformer', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(training=training, center=False)):
                in_shape = net.get_shape().as_list()
                net = slim.flatten(net)
                net = merge(net, d_vp, dim=1)
                net = slim.fully_connected(net, np.prod(in_shape[1:]), scope='fc_transform')
                net = tf.reshape(net, in_shape)
                return net


class VanillaDisc:
    def __init__(self, num_layers=5, fc_activation=tf.nn.relu, fix_bn=False):
        self.num_layers = num_layers
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation

    def classify(self, net, num_out, reuse=None, training=True, with_fc=True):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or test mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        with tf.variable_scope('vp_regressor', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(activation=lrelu, padding='SAME', training=training,
                                               fix_bn=self.fix_bn)):
                f_dims = DEFAULT_FILTER_DIMS
                for l in range(0, self.num_layers):
                    if l == 0:
                        net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1),
                                          normalizer_fn=None)
                    else:
                        net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))
                encoded = net

                if with_fc:
                    # Fully connected layers
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 4096, scope='fc1', trainable=with_fc)
                    net = slim.dropout(net, 0.5, is_training=training)
                    net = slim.fully_connected(net, num_out,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               biases_initializer=tf.zeros_initializer,
                                               trainable=with_fc)
                return net, encoded

    def discriminate(self, net, reuse=None, training=True, with_fc=True):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or test mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(vpnet_argscope(activation=lrelu, padding='SAME', training=training,
                                               fix_bn=self.fix_bn)):
                f_dims = DEFAULT_FILTER_DIMS
                for l in range(0, self.num_layers):
                    if l == 0:
                        net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1),
                                          normalizer_fn=None)
                    else:
                        net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))
                encoded = net

                if with_fc:
                    # Fully connected layers
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 4096, scope='fc1', trainable=with_fc)
                    net = slim.dropout(net, 0.5, is_training=training)
                    net = slim.fully_connected(net, 2,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               biases_initializer=tf.zeros_initializer,
                                               trainable=with_fc)
                return net, encoded
