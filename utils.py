import librosa
import numpy as np
import tensorflow as tf

def tensor_shape(t):
  return t.get_shape().as_list()
