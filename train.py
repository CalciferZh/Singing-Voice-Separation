import tensorflow as tf
import os
import time

from data import Reader
from utils import *
from network import baseline_network


def train_baseline(reader, name, log_root, lrate, momentum=0.9, save_every=100, max_to_keep=100):
  log_dir = os.path.join(log_root, name)
  if os.path.isdir(log_dir):
    restore = True
  else:
    restore = False
    os.makedirs(log_dir)

  print('=====================================================================')
  if restore:
    print('RESTORE FROM OLD SETTINGS: %s' % name)
  else:
    print('START A NEW MODEL: %s' % name)
  print('=====================================================================')

  x_ph = tf.placeholder(tf.float32, [None, reader.sample_rate * reader.n_seconds / 8 + 1, 513])
  y1_pred, y2_pred = baseline_network(x_ph)
  y1_truth_ph = tf.placeholder(tf.float32, tensor_shape(y1_pred))
  y2_truth_ph = tf.placeholder(tf.float32, tensor_shape(y2_pred))
  loss = tf.reduce_mean(tf.square(y1_truth_ph - y1_pred) + tf.square(y2_truth_ph - y2_pred))
  lrate_ph = tf.placeholder(tf.float32, [])
  step_tensor = tf.Variable(0, name='step', trainable=False)
  opt = tf.train.MomentumOptimizer(lrate_ph, momentum)
  train = opt.minimize(loss, global_step=step_tensor)
  summary = tf.summary.scalar('loss', loss)
  writer = tf.summary.FileWriter(log_dir)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver(max_to_keep=max_to_keep)
  sess = tf.Session()
  sess.run(init)

  if restore:
    saver.restore(sess, tf.train.latest_checkpoint(log_dir))

  step =  tf.train.global_step(sess, step_tensor)
  while True:
    tic = time.time()
    batch_x, batch_y1_truth, batch_y2_truth = reader.next_batch()
    fd = {
      x_ph: batch_x,
      y1_truth_ph: batch_y1_truth,
      y2_truth_ph: batch_y2_truth,
      lrate_ph: lrate
    }

    batch_loss, batch_summary, _ = sess.run([loss, summary, train], fd)

    step = tf.train.global_step(sess, step_tensor)
    writer.add_summary(batch_summary, step)

    if step % save_every == 0:
      saver.save(sess, os.path.join(log_dir, '%d.ckpt' % step))

    toc = time.time()
    print(
      '%s: step %d, loss %e, interval %f' % (name, step, batch_loss, toc-tic)
    )


if __name__ == '__main__':
  reader = Reader(
    './data/train.txt',
    './data/wav',
    1,
  )
  train_baseline(
    reader,
    'baseline',
    './log',
    0.0001
  )
