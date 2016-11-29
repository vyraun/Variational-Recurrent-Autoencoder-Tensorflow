from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from model import VRAE

from tensorflow.models.rnn.ptb import reader #penn treebank


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "ptb",
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
  enc_num_layers = 2
  enc_dim = 512
  embed_dim = 512
  latent_dim = 16
  dec_dim = 512
  dec_num_layers = 2
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
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
  if model.is_training:
    fetches["optimizer"] = model.optimizer

  for batch in data:
    batch_size, seq_len = batch.shape
    feed_dict = {batch_size:batch_size, seq_len: seq_len,
      input_data: batch}
    vals = session.run(fetches, feed_dict=feed_dict)
    KL_term = vals["KL_term"]
    reconstruction_cost = vals["reconstruction_cost"]
    cost = vals["cost"]
    costs += cost
    KL_terms += KL_term
    reconstruction_costs = reconstruction_cost

  return costs, KL_terms, reconstruction_costs

def sampling(session, model, id_to_word):
  outputs = model.generate(session)
  outputs = np.transpose(outputs)
  outputs = outputs.tolist()
  outputs = [[id_to_word[word] for word in sentence] for sentence in outputs]
  return outputs

def reconstruct(session, model, outputs, word_to_id, id_to_word):
  inputs = [[word_to_id[word] for word in sentence] for sentence in inputs]
  input_data = tf.convert_to_tensor(inputs)
  input_data = np.transpose(input_data)
  outputs = model.reconstruct(session, input_data)
  outputs = np.transpose(outputs)
  outputs = outputs.tolist()
  outputs = [[id_to_word[idx] for idx in sentence] for sentence in outputs]
  return outputs



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


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")


  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, word_to_id, id_to_word = raw_data
  train_data = ptb_producer(train_data, config.batch_size)
  valid_data = ptb_producer(valid_data, config.batch_size)
  test_data  = ptb_producer(test_data, eval_config.batch_size)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = VRAE(is_training=True, config=config)
      tf.scalar_summary("Training Loss", m.cost)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = VRAE(is_training=False, config=config)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = VRAE(is_training=False, config=eval_config)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        KL_rate = (i > 3 ? (i - 3)*0.1 : 0)
        m.assign_KL_rate(session, KL_rate)

        train_costs, train_KL_term, train_reconstruction_cost = run_epoch(
          session, m)
        print("Epoch: %d Train costs: %.3f" % (i + 1, train_costs))
        print("Epoch: %d Train KL divergence: %.3f" % (i + 1, train_KL_term))
        print("Epoch: %d Train reconstruction costs: %.3f"
          % (i + 1, train_reconstruction_cost))
        valid_costs, valid_KL_term, valid_reconstruction_cost = run_epoch(
          session, mvalid, train_data)
        print("Epoch: %d Valid costs: %.3f" % (i + 1, valid_costs))
        print("Epoch: %d Valid KL divergence: %.3f" % (i + 1, valid_KL_term))
        print("Epoch: %d Valid reconstruction costs: %.3f"
          % (i + 1, valid_reconstruction_cost))

      test_costs, test_KL_term, test_reconstruction_cost = run_epoch(session, mtest)
      print("Epoch: %d Test costs: %.3f" % (i + 1, test_costs))
      print("Epoch: %d Test KL divergence: %.3f" % (i + 1, test_KL_term))
      print("Epoch: %d Test reconstruction costs: %.3f"
        % (i + 1, test_reconstruction_cost))
      sampling_outputs = sampling(session, mtest, id_to_word)
      print("sampling outputs: ", sampling_outputs)
      reconstruct_inputs = ["I am hungry."]
      reconstruct_outputs = reconstruct(session, mtest, reconstruct_inputs, word_to_id, id_to_word)
      print("reconstruct outputs: ", reconstruct_outputs)


      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
