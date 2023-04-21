import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotter

def preprocess(x: tf.Tensor, target_shape, limit_num_samples_to=None):
  if len(target_shape) != 2:
    raise ValueError('target_shape must be a tuple of length 2')
  if len(x.shape) < 4:
    x = tf.expand_dims(x, axis=-1)
  x = tf.image.resize(x, target_shape)
  if limit_num_samples_to is not None:
    random_indices = np.random.choice(x.shape[0], limit_num_samples_to, replace=False)
    x = tf.gather(x, random_indices)
  else:
    x = tf.random.shuffle(x)
  return tf.cast(x, tf.float32) / 127.5 - 1


data = np.load('data/faces.npy')
target_shape = (72, 72)
channels = 3

data = preprocess(data, target_shape=target_shape)
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

i = 0
while True:
  plotter.imshow_rgb_safe(data[i])
  plt.show()
  i += 1