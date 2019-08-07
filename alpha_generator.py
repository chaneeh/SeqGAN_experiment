# reference paper: https://arxiv.org/pdf/1609.05473.pdf
#
# original code from LantaoYu
# 
#


import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, learning_rate=0.01, reward_gamma=0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.int64_start_token = tf.constant([start_token]*self.batch_size, dtype=tf.int64)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.grad_clip = 5.0


        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            self.g_output_unit = self.create_output_unit(self.g_params)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.nn.softmax(o_t)) #[batch, vocab]
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        self.gen_o = self.gen_o.stack()
        self.gen_o = tf.transpose(self.gen_o, perm=[1, 0, 2]) # batch, seq, vocab

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # pretraining loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        # training updates
        pretrain_opt = self.g_optimizer1(self.learning_rate)
        self.optim = pretrain_opt

        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))



    def pretrain_step(self, sess, x):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x})
        return outputs

    def get_rollout_samples(self, rollout_num, sess, vl_rollout, fast_rollout, discriminator):
        self.sess= sess
        self.rollout_num = rollout_num
        self.vl_rollout = vl_rollout
        self.fast_rollout = fast_rollout
        self.discriminator = discriminator

        gen_oo = [[]]*self.sequence_length
        gen_xx = [[]]*self.sequence_length
        sub_gen_xx = [[]] * (self.sequence_length + 1)
        sub_gen_xx[0] = self.int64_start_token


        def rollout_select(sub_gen_xx, o_t, i):
            o_t = tf.nn.softmax(o_t)
            vl_predictions = self.vl_rollout.get_vl_predictions(self.sess, sub_gen_xx, i+1)
            fast_predictions = self.fast_rollout.get_fast_predictions(self.sess, sub_gen_xx, self.rollout_num, o_t, self.discriminator, i+1)
            gen_ooo = vl_predictions + fast_predictions #[batch, vocab]
            next_token = tf.cast(tf.reshape(np.argmax(gen_ooo, 1), [self.batch_size]),tf.int32)
            #ValueError: Argument must be a dense tensor: [array([4175]), array([3730]), array([2184])] - got shape[3, 1], but wanted[3].
            #the reason for this error by "next_token = np.argmax(gen_ooo, 1)"
            #tf.reshape should get value like dense tensor or list but np is not allowed

            print 'finish_making_each_rollout_next_token'
            return next_token #[batch]


        def generate_rollout(i, x_t, h_tm1, gen_oo, gen_xx, sub_gen_xx):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            next_token = rollout_select(sub_gen_xx, o_t, i)
            gen_xx[i] = next_token
            sub_gen_xx[i+1] = next_token
            gen_oo[i] = tf.nn.softmax(o_t)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            return i+1, x_tp1, h_t, gen_oo, gen_xx, sub_gen_xx

        i = 0
        x_t = tf.nn.embedding_lookup(self.g_embeddings, self.int64_start_token)
        h_t = self.h0
        gen_oo = gen_oo
        gen_xx = gen_xx
        sub_gen_xx = sub_gen_xx
        for _ in range(self.sequence_length):
            i, x_t, h_t, gen_oo, gen_xx, sub_gen_xx = generate_rollout(i, x_t, h_t, gen_oo, gen_xx, sub_gen_xx)

        gen_oo = tf.reshape(gen_oo, [self.sequence_length, self.batch_size, self.num_emb])
        gen_xx = tf.reshape(gen_xx, [self.sequence_length, self.batch_size])
        self.gen_oo = tf.transpose(gen_oo, perm=[1, 0, 2]) #[batch, seq, vocab]
        self.gen_xx = tf.transpose(gen_xx, perm=[1, 0]) #[batch, seq]

        ret1 = self.sess.run(self.gen_oo)
        ret2 = self.sess.run(self.gen_xx)
        return ret1, ret2

    def g_updates(self, sess, rewards):
        self.sess = sess
        self.rewards_for_policy = rewards # batch, seq
        self.g_loss_for_policy = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.gen_xx, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.gen_oo, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards_for_policy, [-1])
        )
        self.g_grad_for_policy, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_for_policy, self.g_params), self.grad_clip)
        self.g_updates_for_policy = self.optim.apply_gradients(zip(self.g_grad_for_policy, self.g_params))
        self.sess.run(self.g_updates_for_policy)

    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def generate_prob(self, sess):
        outputs = sess.run(self.gen_o) #[batch, seq, vocab]
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer1(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

    def g_optimizer2(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

