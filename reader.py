# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

_EOS = "<eos>"
_UNK = "<unk>"
_START_VOCAB = (_EOS, _UNK)

EOS_ID = 0
UNK_ID = 1

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()

def _read_words_to_list_of_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
      return [ s.strip().split() + ['<eos>'] for s in f.read().split('\n')]

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter([w for w in data if w not in _START_VOCAB])

  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  words = _START_VOCAB + words
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words_to_list_of_sentences(filename)
  return [[word_to_id[word] for word in sentence] for sentence in data]

def _group_sentences_into_batches(data, batch_size):
  len_sentence_list = [len(s) for s in data]
  len_sentence_list = list(set(len_sentence_list))
  sent_lists_grouped_by_len = {seq_len: [] for seq_len in len_sentence_list}
  for sent in data:
    sent_lists_grouped_by_len[len(sent)].append(sent)
  batch_sized_sent_list = []
  for seq_len, sent_list in sent_lists_grouped_by_len.iteritems():
    idx = 0
    while idx < len(sent_list):
      batch_sized_sent_list.append(sent_list[idx:(idx+batch_size)])
      idx += batch_size
  return batch_sized_sent_list

def _to_batch_major(data):
  return [np.array(sent_list).transpose() for sent_list in data]

def _get_data_from_path(data_path, word_to_id):
  data = _file_to_word_ids(data_path, word_to_id)[:-1]
  #data = _group_sentences_into_batches(data, batch_size)
  #data = _to_batch_major(data)
  return data

def ptb_raw_data(data_path):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  print(word_to_id)
  train_data = _get_data_from_path(train_path, word_to_id)
  valid_data = _get_data_from_path(valid_path, word_to_id)
  test_data = _get_data_from_path(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
