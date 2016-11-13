import numpy
import tensorflow


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

    for key in config:
      setattr(self, '_' + key, config[key])

    input_data = tf.placeholder(tf.int32, [self._batch_size, self._seq_len])
    with tf.variable_scope("enc"):
      enc_mean, enc_stddev = encoder(input_data)
    with tf.variable_scope('dec'):
      decoder(enc_mean, enc_stddev)

  def encoder(self, input_data):

    lstm_cell = tf.nn.rnn.BasicLSTMCell(self._enc_dim, forget_bias=1.0, state_is_tuple=True)
    if self._is_training and self._keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell, output_keep_prob=self._keep_prob)
    cell = tf.nn.MultiRNNCell([lstm_cell] * self._enc_num_layers, state_is_tuple=True)

    enc_initial_state = cell.zero_state(self._batch_size, data_type())
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [self._vocab_size, self._embed_dim], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_data)
    if self._is_training and self._keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self._keep_prob)

    inputs = [tf.squeeze(input_step, [1])
              for input_step in tf.split(1, self._seq_len, inputs)]

    state = enc_initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(self._seq_len):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    with tf.variable_scope('enc'):
      with tf.variable_scope('mean'):
        w = tf.get_variable("w", [self._enc_dim, self._latent_dim], dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_mean = tf.matmul(state, w) + b
      with tf.variable_scope('stddev'):
        w = tf.get_variable("w", [self._enc_dim, self._latent_dim], dtype=data_type())
        b = tf.get_variable("b", [self._latent_dim], dtype=data_type())
        enc_stddev = tf.matmul(state, w) + b

    return enc_mean, enc_stddev

  def decoder(self, mean=None, stddev=None):
    '''Create decoder network.

      if decoder input(dec_input) is provided, then decode from that tensor,
      otherwise sample from the latent space.

    Args:
      mean, stddev: a batch of input means and standard deviations to decode.
      Inputs will pass through a 1-layer feedforward network to fit the
      dimension of the state of the decoder.
      Output of the layer will be the initial state of the decoder RNN-LSTM.
    Returns:

    '''
    epsilon = tf.random_normal([self._batch_size, self._latent_dim])
    if(mean is None or stddev is None):
      input_sample = epsilon
    else:
      input_sample = mean + epsilon * stddev

    w = tf.get_variable("w", [self._latent_dim, self._dec_dim], dtype=data_type())
    b = tf.get_variable("b", [self._dec_dim], dtype=data_type())

    dec_initial_state = tf.matmul(input_sample, w) + b

    lstm_cell = tf.nn.rnn.BasicLSTMCell(self._dec_dim, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.MultiRNNCell([lstm_cell] * self._enc_num_layers, state_is_tuple=True)
