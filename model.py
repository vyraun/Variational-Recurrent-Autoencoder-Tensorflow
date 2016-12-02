from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pdb

import numpy as np
import tensorflow as tf



def data_type():
  return tf.float32


class VRAE(object):

  def __init__(
    self,
    is_training,
    config,
    seq_len):

    self._is_training = is_training
    self._seq_len = seq_len

    for key in dir(config):
          setattr(self, '_' + key, getattr(config, key))

    self._input_data = tf.placeholder(tf.int32,
      shape=(self._batch_size, self._seq_len))

    with tf.variable_scope("encoder"):
      enc_mean, enc_stddev = self.encoder(self._input_data)
    with tf.variable_scope('dec'):
      outputs = self.decoder(enc_mean, enc_stddev)

    self._outputs = outputs
    self._KL_rate = tf.Variable(0.0, trainable=False, name="KL_rate")
    self._KL_term = self.get_KL_term(enc_mean, enc_stddev)
    self._reconstruction_cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
      outputs, self._input_data))
    cost = self._KL_rate * self._KL_term + self._reconstruction_cost
    self._cost = cost
    loss = cost / self._batch_size

    if not self._is_training:
      return

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      self._max_grad_norm)
    self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

    self._new_KL_rate = tf.placeholder(
      tf.float32, shape=[], name="new_KL_rate")
    self._KL_rate_update = tf.assign(self._KL_rate, self._new_KL_rate)

  def get_KL_term(self, mean, stddev, epsilon=1e-8):
    '''KL_divergence
        get KL divergence of q(z|x) and p(z).
        q(z|x) ~ N(z; mean, stddev^2)
        p(z) ~ N(z; 0, I)
    Args:
      mean: mean of q(z|x)
      stddev: standard deviation of q(z|x)
      epsilon: added to log term to avoid taking logarithm of zero
    Return:
      mini-batch KL divergence that sum over batches
      shape: scalar
    '''
    return tf.reduce_sum(0.5 * (2 * tf.log(stddev + epsilon)
      - tf.square(mean) - tf.square(stddev) + 1.0))
  def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction cost
        get reconstruction cost: log(p(x|z))

    Args:
      output_tensor: tensor produces by decoder
      target_tensor: the target tensor that we want to reconstruct
      epsilon: added to log term to avoid taking logarithm of zero
    Return:
      mini-batch cross entropy that sum over batches
      shape: scalar
    '''
    return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
      output_tensor, target_tensor))

  def encoder(self, input_data):

    single_cell = tf.nn.rnn_cell.BasicLSTMCell(self._enc_dim, forget_bias=1.0,
      state_is_tuple=True)
    cell = single_cell
    if self._enc_num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self._enc_num_layers,
        state_is_tuple=True)

    enc_initial_state = cell.zero_state(self._batch_size, data_type())
    print(cell.state_size)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("encoder_embedding",
        [self._vocab_size, self._embed_dim],
        dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_data)
      assert(cell.state_size[0] ==inputs.get_shape()[2])

    outputs = []
    state = enc_initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(self._seq_len):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (_, state) = cell(inputs[:, time_step, :], state)

    enc_final_state_c, _ = state

    with tf.variable_scope('encoder_to_latent'):
      with tf.variable_scope('mean'):
        w = tf.get_variable("w",[self._enc_dim, self._latent_dim],
          dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_mean = tf.nn.relu(tf.matmul(enc_final_state_c, w) + b)
      with tf.variable_scope('stddev'):
        w = tf.get_variable("w",
          [self._enc_dim, self._latent_dim], dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_stddev = tf.nn.relu(tf.matmul(enc_final_state_c, w) + b)

    return enc_mean, enc_stddev

  def decoder(self, mean=None, stddev=None):
    '''Create decoder network.

      if decoder input(dec_input) is provided,
      then decode from that tensor,
      otherwise sample from the latent space.

    Args:
      mean, stddev:
        a batch of input means and standard deviations to decode.
        both mean and stddev are of shape [batch_size, latent_dim].
        Inputs will pass through a 1-layer feedforward network to fit the
        dimension of the state of the decoder.
        Output of the layer will be the initial state of the decoder
        RNN-LSTM.
    Returns:

    '''
    epsilon = tf.random_normal([self._batch_size, self._latent_dim])
    #epsilon ~ N(0,I)
    if(mean is None or stddev is None):
      input_sample = epsilon
    else:
      input_sample = mean + tf.mul(epsilon, stddev)

    with tf.variable_scope('latent_to_decoder'):
      with tf.variable_scope('cell_state'):
        w = tf.get_variable("w", [self._latent_dim, self._dec_dim],
          dtype=data_type())
        b = tf.get_variable("b", [self._dec_dim], dtype=data_type())
        xw_b = tf.matmul(input_sample, w) + b
        dec_initial_state_c = tf.nn.relu(xw_b)
      with tf.variable_scope('hidden_state'):
        w = tf.get_variable("w", [self._latent_dim, self._dec_dim],
          dtype=data_type())
        b = tf.get_variable("b", [self._dec_dim], dtype=data_type())
        xw_b = tf.matmul(input_sample, w) + b
        dec_initial_state_h = tf.nn.relu(xw_b)

    single_cell = tf.nn.rnn_cell.BasicLSTMCell(self._dec_dim, forget_bias=1.0,
        state_is_tuple=True)
    cell = single_cell
    if self._dec_num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self._dec_num_layers,
        state_is_tuple=True)

    outputs = []
    with tf.variable_scope("RNN"): #an RNNLM
      state = (dec_initial_state_c, dec_initial_state_h)
      word_input = tf.zeros([self._batch_size, self._vocab_size])
      for time_step in range(self._seq_len):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        input_embedding = tf.get_variable("input_embedding",
          [self._vocab_size, self._enc_dim])
        if(time_step == 0):
          cell_input = tf.zeros([self._batch_size, self._enc_dim])
        else:
          cell_input = tf.matmul(word_input, input_embedding)
        cell_output, state = cell(cell_input, state)
        output_embedding = tf.get_variable("output_embedding",
          [self._enc_dim, self._vocab_size])
        word_output = tf.matmul(cell_output, output_embedding)
        word_input = word_output
        outputs.append(word_output)

    outputs = tf.reshape(tf.concat(1, outputs),
      [self._batch_size, self._seq_len, self._vocab_size])
    return outputs

  def assign_KL_rate(self, session, KL_rate):
    session.run(self._KL_rate_update, feed_dict={self._new_KL_rate: KL_rate})

  def generate(self, session, mean=None, stddev=None): #generate output by sampling from latent space
    outputs = self.decoder(self, mean, stddev)
    return session.run(outputs)

  def reconstruct(self, session, input_data): #generate output by sampling from latent space
    return session.run(self._outputs, feed_dict={input_data: input_data})

  @property
  def input_data(self):
    return self._input_data

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def initial_state(self):
    return self._enc_initial_state

  @property
  def outputs(self):
    return self._outputs

  @property
  def KL_term(self):
    return self._KL_term

  @property
  def reconstruction_cost(self):
    return self._reconstruction_cost

  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._learning_rate
