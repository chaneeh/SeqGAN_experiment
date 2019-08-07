# reference paper: https://arxiv.org/pdf/1609.05473.pdf
#
# original code from LantaoYu
# 
#



import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


class ROLLOUT(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()
        self.g_output_unit = self.create_output_unit()

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])


    def get_vl_predictions(self, sess, sub_gen_xx, tmp_seq_length):

        def _vl_pretrain_recurrence(i, sub_gen_xx, h_tm1, g_predictions):
            x_t = tf.nn.embedding_lookup(self.g_embeddings, sub_gen_xx[i])
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = tf.nn.softmax(o_t)
            return i+1, sub_gen_xx, h_t, g_predictions
        i = 0
        sub_gen_xx = sub_gen_xx
        h0 = self.h0
        g_predictions = []
        for _ in range(tmp_seq_length):
            i, sub_gen_xx, h0, g_predictions = _vl_pretrain_recurrence(i, sub_gen_xx, h0, g_predictions)
        self.vl_predictions = tf.reshape(g_predictions, [self.batch_size, self.num_emb])

        return sess.run(self.vl_predictions)

    def get_fast_predictions(self, sess, sub_gen_xx, rollout_num, o_t, discriminator, tmp_sequence_length, dis_dropout_keep_prob = 0.75):
        gen_ooo = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

        dict = [[[0.0]]*self.num_emb]*self.batch_size # batch * num_emb

        def fast_g_recurrence_1(i, sub_gen_xx, h_tm1, gen_ooo):
            x_t = tf.nn.embedding_lookup(self.g_embeddings, sub_gen_xx[i])
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            gen_ooo = gen_ooo.write(i, tf.nn.softmax(o_t))
            return i+1, sub_gen_xx, h_t, gen_ooo

        def fast_g_recurrence_2(i, x_t, h_tm1, gen_ooo):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_ooo = gen_ooo.write(i, tf.nn.softmax(o_t)) #[seq, batch, vocab]
            return i+1, x_tp1, h_t, gen_ooo

        i = 0
        sub_gen_xx = sub_gen_xx
        h_t = self.h0
        gen_ooo = gen_ooo
        for tmp_sequence_length_ in range(tmp_sequence_length):
             i, sub_gen_xx, h_t, gen_ooo = fast_g_recurrence_1(i, sub_gen_xx, h_t, gen_ooo)

        for rollout_num_ in range(rollout_num):
            log_prob = tf.log(o_t)
            selected_token1 = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32) #[batch]
            selected_token2 = tf.nn.embedding_lookup(self.g_embeddings, selected_token1) #[batch, emb_dim]
            i = i
            x_t = selected_token2
            h_t = h_t
            gen_ooo = gen_ooo
            left_sequence_num = self.sequence_length - i
            for _ in range(left_sequence_num):
                i, x_t, h_t, gen_ooo = fast_g_recurrence_2(i, x_t, h_t, gen_ooo)
            self.gen_ooo = tf.transpose(gen_ooo.stack(), perm=[1,0,2]) # batch, seq, vocab
            self.rollout_per_result = sess.run(self.gen_ooo)

            generator_label = np.array([[1, 0] for _ in range(self.batch_size)])
            feed = {
                discriminator.input_x: self.rollout_per_result,
                discriminator.input_y: generator_label,
                discriminator.dropout_keep_prob: dis_dropout_keep_prob}
            rewards = sess.run(discriminator.recovered_reward_for_policy, feed_dict=feed)

            rewards = tf.reduce_mean(tf.reshape(rewards, [self.batch_size, -1]), [-1])
            rewards = sess.run(rewards)
            selected_token1 = sess.run(selected_token1)

            for batch_num in range(len(selected_token1)):
                dict[batch_num][selected_token1[batch_num]].extend([rewards[batch_num]])

        batch_idx = -1
        for batch_dict in dict:
            batch_idx = batch_idx + 1
            vocab_batch_idx = -1
            for vocab_batch_dict in batch_dict:
                vocab_batch_idx = vocab_batch_idx + 1
                if len(vocab_batch_dict) == 0:
                    dict[batch_idx][vocab_batch_idx] = [0.0]
                else:
                    dict[batch_idx][vocab_batch_idx] = [tf.reduce_sum(vocab_batch_dict)]

        dict = tf.reshape(dict, [self.batch_size, self.num_emb])
        dict = tf.nn.softmax(dict) #[batch, vocab_size]
        ret = sess.run(dict)
        return ret

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

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

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

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

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()
