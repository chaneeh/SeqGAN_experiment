# reference paper: https://arxiv.org/pdf/1609.05473.pdf
#
# original code from LantaoYu
# 
#

import tensorflow as tf
import numpy as np
def linear(input_, output_size, scope=None):

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2d arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

##########################################################################################
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
##########################################################################################
def conv2d(input_, output_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
             k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.relu(deconv)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    return shape

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, h_scale, w_scale):
    _, x_h, x_w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (x_h * h_scale, x_w * w_scale))

class Discriminator(object):
    def __init__(
            self, batch_size, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters,
            decoder_emb_size=32):

        self.input_x = tf.placeholder(tf.float32, [batch_size, sequence_length, vocab_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.k_t = tf.Variable(0., trainable=False, name='k_t')
        self.lambda_k = tf.constant(0.001, name='lambda_k')
        self.gamma = tf.constant(0.5, name='gamma')

        self.input_x_reshape = tf.reshape(self.input_x, [-1, vocab_size])
        with tf.variable_scope('discriminator'):
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W0 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W0")
                self.b0 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0), name="b0")

                self.embedded_chars = tf.matmul(self.input_x_reshape, self.W0) + self.b0
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
                self.embedded_chars_expanded = tf.reshape(self.embedded_chars_expanded, [-1, sequence_length, embedding_size, 1])

            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # <- broadcasting function!
                    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs,3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            with tf.name_scope("decoding"):
                self.h_drop = tf.reshape(self.h_drop, [-1, num_filters_total])
                W1 = tf.Variable(tf.truncated_normal([num_filters_total, decoder_emb_size], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[decoder_emb_size]), name="b1")
                self.decode_starter1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="starter1")

                W2 = tf.Variable(tf.truncated_normal([decoder_emb_size, num_filters_total], stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b2")
                self.decode_starter2 = tf.nn.xw_plus_b(self.decode_starter1, W2, b2, name="starter2")
                self.decode_starter2 = tf.reshape(self.decode_starter2, [self.batch_size, 1, 1, num_filters_total])
                self.decode_h_pool = tf.split(self.decode_starter2, num_filters, axis=3)

            append_full_decoded_h = []
            for filter_size, num_filter, decoded_h in zip(filter_sizes, num_filters, self.decode_h_pool):
                with tf.variable_scope("dec-conv-maxpool-%s" % filter_size):
                    decoded_h = tf.reshape(decoded_h, [self.batch_size, 1, 1, num_filter])
                    decoded_h = upscale(decoded_h, sequence_length - filter_size + 1, 1)
                    decoded_h = tf.image.resize_nearest_neighbor(decoded_h, (sequence_length, 1))
                    full_decoded_h = deconv2d(decoded_h, [self.batch_size, sequence_length, embedding_size, 1], k_h=filter_size, k_w=embedding_size, d_h=1, d_w=embedding_size, name="deconv2d_deconv")
                    append_full_decoded_h.append(full_decoded_h)
            append_full_decoded_h2 = tf.reshape(conv2d(tf.concat(append_full_decoded_h,3), 1), [-1, embedding_size]) #tf. 1.0.1
            append_full_decoded_h3 = tf.reshape(append_full_decoded_h2, [-1, embedding_size])

            with tf.device('/cpu:0'), tf.name_scope("final_embedding"):
                self.W_0 = tf.Variable(tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0), name="W_0")
                self.b_0 = tf.Variable(tf.random_uniform([vocab_size], -1.0, 1.0), name="b0")
                self.recovered_reward = tf.matmul(append_full_decoded_h3, self.W_0) + self.b_0

            self.recovered_reward = tf.nn.softmax(self.recovered_reward)
            self.recovered_reward = tf.reshape(self.recovered_reward, [self.batch_size, sequence_length, vocab_size])

            d_loss_real = []
            d_loss_fake = []
            input_x = tf.split(self.input_x, self.batch_size, 0)
            recovered_reward = tf.split(self.recovered_reward, self.batch_size, 0)
            input_y = tf.split(self.input_y, self.batch_size, 0)
            for input, recovered_reward, label in zip(input_x, recovered_reward, input_y):
                input = tf.reshape(input, [sequence_length, vocab_size])
                recovered_reward = tf.reshape(recovered_reward, [sequence_length, vocab_size])
                label = tf.reshape(label, [num_classes])
                if label[1] == 1:
                    d_loss_real.append(tf.abs(input - recovered_reward))
                else:
                    d_loss_fake.append(tf.abs(input - recovered_reward))

            self.g_loss_for_gen = tf.reshape(d_loss_fake, [self.batch_size, sequence_length, vocab_size])
            self.recovered_reward_for_policy = -tf.reduce_sum(self.g_loss_for_gen, -1) #batch, seq

            self.d_loss_real = tf.reduce_mean(d_loss_real) if len(d_loss_real) != 0 else 0
            self.d_loss_fake = tf.reduce_mean(d_loss_fake) if len(d_loss_fake) != 0 else 0

            self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
            self.g_loss = self.d_loss_fake
            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            d_optimizer = tf.train.AdamOptimizer(1e-4)
            d_optim = d_optimizer.minimize(self.d_loss, var_list=self.params)
            self.train_op = d_optim

            self.balance = self.gamma * self.d_loss_real - self.g_loss
            with tf.control_dependencies([self.train_op]):
                self.k_update = tf.assign(
                    self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

