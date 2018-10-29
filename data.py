import os
import random
import matplotlib.pyplot as plt
import numpy as np
import librosa


def train_test_split():
  names = os.listdir('./data/wav')
  random.seed(9608)
  random.shuffle(names)
  train_ratio = 0.8
  train_names = names[:int(round(len(names) * train_ratio))]
  test_names = names[len(train_names):]
  with open('./data/train.txt', 'w') as f:
    for name in train_names:
      f.write(name + '\n')
  with open('./data/test.txt', 'w') as f:
    for name in test_names:
      f.write(name + '\n')


class Reader:
  def __init__(self, list_path, wav_folder, batch_size, length=16000 * 8):
    """
    Parameters
    ----------
    length: Wanted length for each clip.
    """
    with open(list_path) as f:
      self.names = f.read().splitlines()
    self.wav_folder = wav_folder
    self.list_path = list_path
    self.batch_size = batch_size
    self.data = []
    for name in self.names:
      data, _ = librosa.load(
        os.path.join(self.wav_folder, name),
        sr=None,
        mono=False
      )
      data = self.wav_pad_crop(data, length)
      self.data.append((name, data))

  def wav_pad_crop(self, wav, length):
    while wav.shape[1] < length:
      wav = np.concatenate([wav, wav], 1)
    wav = wav[:,:length]
    return wav

  def next_batch(self):
    batch = random.sample(self.data, self.batch_size)
    x = []
    y1 = []
    y2 = []
    for sample in batch:
      wav = sample[1]
      spec1 = np.abs(librosa.stft(wav[0], n_fft=1024, hop_length=8)).T
      spec2 = np.abs(librosa.stft(wav[1], n_fft=1024, hop_length=8)).T
      mixture = np.abs(librosa.stft(librosa.to_mono(wav), n_fft=1024, hop_length=8))
      y1.append(spec1)
      y2.append(spec2)
      x.append(mixture.T)
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    return x, y1, y2

if __name__ == '__main__':
  r = Reader(
    './data/train.txt',
    './data/wav',
    16
  )
  _, _ = r.next_batch()
