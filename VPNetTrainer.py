import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import sys

from utils import montage_tf, get_variables_to_train, assign_from_checkpoint_fn
from constants import LOG_DIR

slim = tf.contrib.slim


class VPNetTrainer:
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0002,
                 tag='default'):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.tag = tag
        self.additional_info = None
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.num_train_steps = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}_{}'.format(self.dataset.name, self.model.name, self.tag)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        opts = {'adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate(), beta1=0.5, epsilon=1e-6),
                'sgd+momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate(), momentum=0.9)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {'const': self.init_lr,
                    'linear': self.learning_rate_linear(self.init_lr)}
        return policies[self.lr_policy]

    def get_train_batch(self):
        with tf.device('/cpu:0'):
            # Get the training dataset
            train_set = self.dataset.get_trainset()
            self.num_train_steps = (self.dataset.get_num_train() / self.model.batch_size) * self.num_epochs
            print('Number of training steps: {}'.format(self.num_train_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=8,
                                                                      common_queue_capacity=4 * self.model.batch_size,
                                                                      common_queue_min=self.model.batch_size*2)
            [img, im_size, box, vp] = provider.get(['image', 'im_size', 'bbox', 'viewpoint'])

            # Preprocess data
            img = self.pre_processor.process_train(img, box, im_size)

            # Make batches
            imgs, vps = tf.train.batch([img, vp], batch_size=self.model.batch_size*2, num_threads=8,
                                       capacity=self.model.batch_size*2)
            imgs1, imgs2 = tf.split(0, 2, imgs)
            vps1, vps2 = tf.split(0, 2, vps)

            return imgs1, vps1, imgs2, vps2

    def discriminator_loss(self, disc_out1, disc_out2, disc_labels):
        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        disc_loss = slim.losses.softmax_cross_entropy(disc_out1, disc_labels, scope=disc_loss_scope, weight=1.0)
        disc_loss += slim.losses.softmax_cross_entropy(disc_out2, disc_labels, scope=disc_loss_scope, weight=1.0)
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_total_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Compute accuracy
        predictions = tf.argmax(disc_out1, 1)
        tf.scalar_summary('accuracy/discriminator accuracy',
                          slim.metrics.accuracy(predictions, tf.argmax(disc_labels, 1)))
        return disc_total_loss

    def generator_loss(self, imgs1, dec_ed1, imgs2, dec_ed2, disc_out1, disc_out2,  labels_gen):
        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        gen_disc_loss = slim.losses.softmax_cross_entropy(disc_out1, labels_gen, scope=gen_loss_scope, weight=1.0)
        gen_disc_loss += slim.losses.softmax_cross_entropy(disc_out2, labels_gen, scope=gen_loss_scope, weight=1.0)
        tf.scalar_summary('losses/discriminator loss (generator)', gen_disc_loss)
        gen_ae_loss = tf.contrib.losses.mean_squared_error(predictions=dec_ed1, labels=imgs1, weight=50.0)
        gen_ae_loss += tf.contrib.losses.mean_squared_error(predictions=dec_ed2, labels=imgs2, weight=50.0)
        tf.scalar_summary('losses/autoencoder loss (generator)', gen_ae_loss)
        losses_gen = slim.losses.get_losses(gen_loss_scope)
        losses_gen += slim.losses.get_regularization_losses(gen_loss_scope)
        gen_loss = math_ops.add_n(losses_gen, name='gen_total_loss')
        return gen_loss

    def make_train_op(self, loss, vars2train=None, scope=None):
        if scope:
            vars2train = get_variables_to_train(trainable_scopes=scope)
        train_op = slim.learning.create_train_op(loss, self.optimizer(), variables_to_train=vars2train,
                                                 global_step=self.global_step, summarize_gradients=False)
        return train_op

    def make_summaries(self):
        # Handle summaries
        for variable in slim.get_model_variables():
            tf.histogram_summary(variable.op.name, variable)
        tf.scalar_summary('learning rate', self.learning_rate())

    def make_image_summaries(self, imgs1, dec_im1, dec_ed1, imgs2, dec_im2, dec_ed2):
        tf.image_summary('imgs/imgs1', montage_tf(imgs1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_im1', montage_tf(dec_im1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_ed1', montage_tf(dec_ed1, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/imgs2', montage_tf(imgs2, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_im2', montage_tf(dec_im2, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/dec_ed2', montage_tf(dec_ed2, 1, self.im_per_smry), max_images=1)

    def learning_rate_linear(self, init_lr=0.0002):
        return tf.train.polynomial_decay(init_lr, self.global_step, self.num_train_steps, end_learning_rate=0.0)

    def get_variables_to_train(self, num_conv_train):
        var2train = []
        for i in range(num_conv_train):
            vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(self.model.num_layers - i)],
                                               exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
        vs = slim.get_variables_to_restore(include=['fully_connected'],
                                           exclude=['discriminator/fully_connected'])
        vs = list(set(vs).intersection(tf.trainable_variables()))
        var2train += vs
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        sys.stdout.flush()
        return var2train

    def make_init_fn(self, chpt_path, num_conv2init):
        if num_conv2init == 0:
            return None
        else:
            # Specify the layers of the model you want to exclude
            var2restore = []
            for i in range(num_conv2init):
                vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(i + 1)],
                                                   exclude=['discriminator/fully_connected'])
                var2restore += vs
            init_fn = assign_from_checkpoint_fn(chpt_path, var2restore, ignore_missing_vars=True)
            print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
            sys.stdout.flush()
            return init_fn

    def train(self):
        with self.sess.as_default():
            with self.graph.as_default():
                imgs1, vps1, imgs2, vps2 = self.get_train_batch()

                # Get labels for discriminator training
                labels_disc = self.model.disc_labels()
                labels_gen = self.model.gen_labels()

                # Create the model
                dec_im1, dec_im2, dec_ed1, dec_ed2, disc_out1, disc_out2 = \
                    self.model.net(imgs1, imgs2, vps1, vps2, reuse=None, training=True)

                # Compute losses
                disc_loss = self.discriminator_loss(disc_out1, disc_out2, labels_disc)
                gen_loss = self.generator_loss(imgs1, dec_ed1, imgs2, dec_ed2, disc_out1, disc_out2,  labels_gen)

                # Handle dependencies with update_ops (batch-norm)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    gen_loss = control_flow_ops.with_dependencies([updates], gen_loss)
                    disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

                # Make summaries
                self.make_summaries()
                self.make_image_summaries(imgs1, dec_im1, dec_ed1, imgs2, dec_im2, dec_ed2)

                # Generator training operations
                train_op_gen = self.make_train_op(gen_loss, scope='encoder, decoder, transformer')
                train_op_disc = self.make_train_op(disc_loss, scope='discriminator')

                # Start training
                slim.learning.train(train_op_gen + train_op_disc, self.get_save_dir(),
                                    save_summaries_secs=600,
                                    save_interval_secs=3000,
                                    log_every_n_steps=100,
                                    number_of_steps=self.num_train_steps)
