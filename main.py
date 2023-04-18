import keras.datasets.mnist as mnist
from UNet import UNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import noiser
import plotter

def preprocess(x: tf.Tensor, target_shape, limit_num_samples_to=None):
  if len(target_shape) != 2:
    raise ValueError('target_shape must be a tuple of length 2')
  
  # add channel dimension if necessary
  if len(x.shape) < 4:
    x = tf.expand_dims(x, axis=-1)
  
  # resize to desired shape
  x = tf.image.resize(x, target_shape)

  # shuffle data, reduce samples if necessary 
  if limit_num_samples_to is not None:
    random_indices = np.random.choice(x.shape[0], limit_num_samples_to, replace=False)
    x = tf.gather(x, random_indices)
  else:
    x = tf.random.shuffle(x)

  # map data from [0, 255] -> [-1, 1] and return
  return tf.cast(x, tf.float32) / 127.5 - 1

# ask user if they want to save the weights
def ask_to_save():
  saveYN = input('Do you want to save the weights? (y/n): ')
  if saveYN == 'y':
    model.save_weights('models/medium_mnist_weights')

network_code = input('Which network would you like to train?\n\t1. MNIST\n\t2. Shoes\n\t3. Cats and Dogs\n\t4. Faces\n')
target_shape = (32, 32)
channels = 1

# load data
if network_code == '1':
  (x_train, _), (x_test, _) = mnist.load_data()
  data = np.concatenate((x_train, x_test))
  target_shape = (32, 32)
elif network_code == '2':
  data = np.load('data/shoes.npy')
  target_shape = (36, 48)
  channels = 3
elif network_code == '3':
  data = np.load('data/cats_dogs.npy')
  target_shape = (64, 64)
elif network_code == '4':
  data = np.load('data/faces.npy')
  target_shape = (72, 72)
  channels = 3

# normalize to [-1, 1], resize
data = preprocess(data, target_shape=target_shape)

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

image_shape = (target_shape[0], target_shape[1], channels)
model = UNet(channels=channels, image_shape=image_shape)
model.train(data, show_samples=False, show_losses=False, epochs=10, batch_size=16)

plt.ion()
while True:
  samples = model.sample()
  for i, sample in enumerate(samples):
    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    plotter.imshow_rgb_safe(np.squeeze(sample))
    plt.show()
    plt.pause(0.01)
    plt.clf()
    
  final = np.array(np.clip((samples[-1][0] + 1) * 127.5, 0, 255), np.uint8)
  plt.suptitle('Final Image')
  plotter.imshow_rgb_safe(final)
  plt.show()
  plt.pause(0.1)
  plt.clf()