import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell

from utils import *


def baseline_network(x, n_layers=3, hidden_size=256):
  rnn_layer = MultiRNNCell([GRUCell(hidden_size) for _ in range(n_layers)])
  output_rnn, _ = tf.nn.dynamic_rnn(rnn_layer, x, dtype=tf.float32)
  input_size = tensor_shape(x)[2]
  y1_raw = tf.layers.dense(output_rnn, input_size, activation=tf.nn.relu)
  y2_raw = tf.layers.dense(output_rnn, input_size, activation=tf.nn.relu)
  y1 = y1_raw / (y1_raw + y2_raw + np.finfo(float).eps) * x
  y2 = y2_raw / (y1_raw + y2_raw + np.finfo(float).eps) * x
  return y1, y2


if __name__ == '__main__':
  x = tf.placeholder(tf.float32, [None, 16000, 512])
  y1, y2 = baseline_network(x)
  print(tensor_shape(y1))
  print(tensor_shape(y2))
