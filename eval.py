import numpy as np
import librosa
import tensorflow as tf
import os
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources

from network import baseline_network
from vctoolkit import pkl_save


class Model:
  def __init__(self, model_path, network):
    self.x = tf.placeholder(tf.float32, [None, None, 513])
    self.y1, self.y2 = network(self.x)

    self.sess = tf.Session()
    self.saver = tf.train.Saver()
    self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

  def process(self, x):
    if len(x.shape) == 2:
      x = np.expand_dims(x, 0)
    y1, y2 = self.sess.run([self.y1, self.y2], {self.x: x})
    y1 = np.squeeze(y1)
    y2 = np.squeeze(y2)
    return y1, y2


def process_folder(src_path, list_path, output_path, model):
  with open(list_path) as f:
    names = f.read().splitlines()
  gnsdr = 0
  gsir = 0
  gsar = 0
  total_length = 0
  for name in tqdm(names, ncols=100):
    wav, _ = librosa.load(
      os.path.join(src_path, name),
      sr=None,
      mono=False
    )
    src1 = librosa.stft(wav[0], n_fft=1024, hop_length=8).T
    phs1 = np.angle(src1)

    src2 = librosa.stft(wav[1], n_fft=1024, hop_length=8).T
    phs2 = np.angle(src2)

    mixture = librosa.to_mono(wav)
    x = np.abs(librosa.stft(mixture, n_fft=1024, hop_length=8))
    mag1_pred, mag2_pred = model.process(x.T)

    wav1_pred = librosa.istft((mag1_pred * np.exp(1.j * phs1)).T, hop_length=8)
    wav2_pred = librosa.istft((mag2_pred * np.exp(1.j * phs2)).T, hop_length=8)

    length = min([wav1_pred.shape[0], wav[0].shape[0]])

    wav1_pred = wav1_pred[:length]
    wav2_pred = wav2_pred[:length]
    wav = wav[:,:length]
    mixture = mixture[:length]

    sdr, sir, sar, _ = bss_eval_sources(
      wav, np.array([wav1_pred, wav2_pred]), False
    )
    sdr_mixed, _, _, _ = bss_eval_sources(
      wav, np.array([mixture, mixture]), False
    )
    nsdr = sdr - sdr_mixed
    gnsdr += length * nsdr
    gsir += length * sir
    gsar += length * sar
    total_length += length

    print(gnsdr / total_length, gsir / total_length, gsar / total_length)

    librosa.output.write_wav(
      os.path.join(output_path, name.replace('.wav', '_accompanies.wav')),
      wav1_pred,
      16000
    )
    librosa.output.write_wav(
      os.path.join(output_path, name.replace('.wav', '_vocal.wav')),
      wav2_pred,
      16000
    )
    librosa.output.write_wav(
      os.path.join(output_path, name.replace('.wav', '.wav')),
      mixture,
      16000
    )

  gnsdr /= total_length
  gsir /= total_length
  gsar /= total_length
  print(gnsdr, gsir, gsar)
  pkl_save('scores.pkl', (gnsdr, gsir, gsar))


def process_folder_example():
  model = Model('./log/baseline', baseline_network)
  process_folder(
    './data/wav',
    './data/test.txt',
    './results/baseline',
    model
  )


def process_single_file(src_path, out_path_pattern, model):
  wav, sr = librosa.load(
    src_path,
    sr=None,
    mono=True
  )
  print('sample rate:', sr)

  spec = librosa.stft(wav, n_fft=1024, hop_length=8)
  mag = np.abs(spec)
  ang = np.angle(spec)
  mag1_pred, mag2_pred = model.process(mag.T)

  wav1_pred = librosa.istft(mag1_pred.T * np.exp(1.j * ang), hop_length=8)
  wav2_pred = librosa.istft(mag2_pred.T * np.exp(1.j * ang), hop_length=8)

  librosa.output.write_wav(
    os.path.join(out_path_pattern % 'accompanies'),
    wav1_pred,
    16000
  )
  librosa.output.write_wav(
    os.path.join(out_path_pattern % 'vocal'),
    wav2_pred,
    16000
  )


def process_single_example():
  model = Model('./log/baseline', baseline_network)
  process_single_file(
    './demo.wav',
    './demo_%s.wav',
    model
  )


if __name__ == '__main__':
  process_folder_example()
