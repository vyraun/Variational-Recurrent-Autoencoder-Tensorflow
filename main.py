from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import re
import numpy as np
import tensorflow as tf

from model import VRAE



flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "medium",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "cts_TC_l5.txt",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "models",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.001
  max_grad_norm = 5
  enc_num_layers = 1
  enc_dim = 512
  embed_dim = 512
  latent_dim = 16
  dec_dim = 512
  dec_num_layers = 1
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.001
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, data):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  KL_terms = 0.0
  reconstruction_costs = 0.0

  fetches = {
      "KL_term": model.KL_term,
      "reconstruction_cost": model.reconstruction_cost,
      "cost": model.cost,
  }
  if model._is_training:
    fetches["optimizer"] = model.optimizer
  if not model._is_training:
    fetches["outputs"] = model.outputs

  for batch in data:
    batch = tf.convert_to_tensor(batch, dtype=tf.int32)
    print(batch.get_shape())
    vals = session.run(fetches, feed_dict={input_data: batch})
    KL_term = vals["KL_term"]
    reconstruction_cost = vals["reconstruction_cost"]
    cost = vals["cost"]
    costs += cost
    KL_terms += KL_term
    reconstruction_costs = reconstruction_cost
    if not is_training:
      print(fetches["outputs"])


  return costs, KL_terms, reconstruction_costs

def sampling(session, model, id_to_word):
  outputs = model.generate(session)
  outputs = outputs.tolist()
  outputs = [[id_to_word[word] for word in sentence] for sentence in outputs]
  return outputs

def linear_interpolate(session, model, start_pt, end_pt, num_pts, id_to_word):
  pts = []
  for s, e in zip(start_pt.tolist(),end_pt.tolist()):
    pts.append(np.linspace(s, e, num_pts))

  pts = np.array(pts)
  pts = pts.T
  outputs = []
  for pt in pts:
    outputs.append(model.generate(session, mean=pt, stddev=0))
  outputs = outputs.tolist()
  outputs = [[id_to_word[word] for word in sentence] for sentence in outputs]
  return outputs

def reconstruct(session, model, seq_input, word_to_id, id_to_word):
  seq_input = [word_to_id[word] for word in seq_input]
  input_data = tf.convert_to_tensor(seq_input)
  seq_output = model.reconstruct(session, input_data)
  seq_output = seq_output.tolist()
  seq_output = [id_to_word[idx] for idx in seq_output]
  return seq_output

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def char_index_mapping(text):
  print('corpus length:', len(text))
  chars = sorted(list(set(text)))
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))
  return char_indices, indices_char, len(chars)

def get_batch_from_seq_list(seq_list, batch_size):
  '''
  Args:
    seq_list: list of sequences with sequence length seq_len
  Returns:
    list of np.array with shape (batch_size, seq_len)
  '''
  list_of_batches = []
  batch_idx = 0
  while batch_idx < len(seq_list):
    list_of_batches.append(np.array(seq_list[batch_idx:batch_idx + batch_size]))
    batch_idx += batch_size
  return list_of_batches

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  config = get_config()
  eval_config = get_config()
  batch_size = config.batch_size


  text = open(FLAGS.data_path).read()
  seq_list = re.split("[，。\n]+", text)
  cts_seq_len = 5
  word_to_id, id_to_word, num_chars = char_index_mapping(text)
  ind_seq_list = [[word_to_id[char] for char in seq] for seq in seq_list]
  data = get_batch_from_seq_list(ind_seq_list, config.batch_size)
  config.vocab_size = num_chars
  eval_config.vocab_size = num_chars
  eval_config.batch_size = 1

  data_split = (int(len(data) * 0.9) // (batch_size)) * batch_size
  train_data = data[:data_split]
  test_data = data[data_split:]

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = VRAE(is_training=True, config=config, seq_len=cts_seq_len)
      tf.scalar_summary("Training Loss", m.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = VRAE(is_training=True, config=eval_config, seq_len=cts_seq_len)
      tf.scalar_summary("Training Loss", mtest.cost)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        KL_rate = 0
        if(i > 3):
          KL_rate = (i - 3) * 0.1
        m.assign_KL_rate(session, KL_rate)

        train_costs, train_KL_term, train_reconstruction_cost = run_epoch(
          session, m, train_data)
        print("Epoch: %d Train costs: %.3f" % (i + 1, train_costs))
        print("Epoch: %d Train KL divergence: %.3f" % (i + 1, train_KL_term))
        print("Epoch: %d Train reconstruction costs: %.3f"
          % (i + 1, train_reconstruction_cost))

        test_costs, test_KL_term, test_reconstruction_cost = run_epoch(
          session, mtest, test_data)
        print("Epoch: %d test costs: %.3f" % (i + 1, test_costs))
        print("Epoch: %d test KL divergence: %.3f" % (i + 1, test_KL_term))
        print("Epoch: %d test reconstruction costs: %.3f"
          % (i + 1, test_reconstruction_cost))

      sampled_cov = np.identity(config.latent_dim)
      sampled_mean = np.zeros(config.latent_dim)
      start_pt = np.random.multivariate_normal(sampled_mean, sampled_cov)
      end_pt = np.random.multivariate_normal(sampled_mean, sampled_cov)      
      sampling_outputs = linear_interpolate(session, mtest, start_pt, end_pt, 21, id_to_word)
      print("sampling outputs: ", sampling_outputs)
      reconstruct_inputs = seq_list[-32:]
      reconstruct_outputs = []
      for recon_input in reconstruct_inputs:
        reconstruct_outputs.append(reconstruct(
          session,mtest, reconstruct_input, word_to_id, id_to_word))
      for recon_input, recon_output in zip(reconstruct_inputs, reconstruct_outputs):
        print("reconstruct inputs: ", recon_input)
        print("reconstruct outputs: ", recon_output)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
