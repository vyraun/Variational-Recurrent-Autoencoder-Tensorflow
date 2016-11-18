from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf



def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32




class VRAE(object):

  def __init__(
    self,
    is_training,
    config,
    input_):

    self._input = input_
    self._is_training = is_training
    self._batch_size = batch_size = _input.batch_size
    self._seq_len = _input.seq_len

    for key in config:
      setattr(self, '_' + key, config[key])

    input_data = tf.placeholder(tf.int32,
      [self._batch_size, self._seq_len])
    with tf.variable_scope("enc"):
      enc_mean, enc_stddev = encoder(input_data)
    with tf.variable_scope('dec'):
      outputs = decoder(enc_mean, enc_stddev)

    loss = get_KL_term(enc_mean, enc_stddev) + get_reconstruction_cost(
      outputs, input_data)
    self._cost = cost = loss / self._batch_size

    if not self._is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      self._max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

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
  def get_reconstruction_cost(ouput_tensor, target_tensor, epsilon=1e-8):
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

    single_cell = tf.nn.rnn.BasicLSTMCell(self._enc_dim, forget_bias=1.0,
      state_is_tuple=True)
    cell = single_cell
    if self._enc_num_layers > 1:
      cell = tf.nn.MultiRNNCell([single_cell] * self._enc_num_layers,
        state_is_tuple=True)

    enc_initial_state = cell.zero_state(self._batch_size, data_type())
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding",
        [self._vocab_size, self._embed_dim],
        dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_data)

    _, enc_final_state = tf.nn.dynamic_rnn(
      cell, inputs, sequence_length=self._seq_len,
        initial_state=enc_initial_state, dtype=data_type())

    with tf.variable_scope('enc'):
      with tf.variable_scope('mean'):
        w = tf.get_variable("w",[self._enc_dim, self._latent_dim],
          dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_mean = tf.matmul(enc_final_state, w) + b
      with tf.variable_scope('stddev'):
        w = tf.get_variable("w",
          [self._enc_dim, self._latent_dim], dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_stddev = tf.matmul(enc_final_state, w) + b

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

    w = tf.get_variable("w", [self._latent_dim, self._dec_dim],
      dtype=data_type())
    b = tf.get_variable("b", [self._dec_dim], dtype=data_type())

    dec_initial_state = tf.matmul(input_sample, w) + b

    single_cell = tf.nn.rnn.BasicLSTMCell(self._dec_dim, forget_bias=1.0,
      state_is_tuple=True)
    cell = single_cell
    if self._dec_num_layers > 1:
      cell = tf.nn.MultiRNNCell([single_cell] * self._dec_num_layers,
        state_is_tuple=True)


    with tf.variable_scope("RNN"): #an RNNLM
      state = dec_initial_state
      initial_input = tf.zeros([self._batch_size, self._dec_dim])
      cell_output = initial_input
      for time_step in range(self._seq_len):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        #cell_output being zero tensor in the first iteration
        (cell_output, state) = cell(cell_output, state)
        outputs.append(cell_output)

    outputs = tf.reshape(tf.concat(1, outputs),
      [self._batch_size, self._seq_len, self._dec_dim])
    return outputs

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._enc_initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


