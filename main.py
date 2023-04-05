import keras.datasets.mnist as mnist
from UNet import UNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import noiser

def preprocess(x: tf.Tensor, target_shape, limit_num_samples_to=None):
  if len(target_shape) != 2:
    raise ValueError('target_shape must be a tuple of length 2')
  
  # shuffle data, reduce samples if necessary 
  if limit_num_samples_to is not None:
    random_indices = np.random.choice(x.shape[0], limit_num_samples_to, replace=False)
    x = tf.gather(x, random_indices)
  else:
    np.random.shuffle(x)

  # map data from [0, 255] -> [-1, 1]
  normalized: tf.Tensor = tf.cast(x, tf.float32) / 127.5 - 1

  # add channel dimension if necessary
  if len(normalized.shape) < 4:
    tf.expand_dims(normalized, axis=-1)
  
  # resize to desired shape
  return tf.image.resize(normalized, target_shape)

def imshow_rgb_safe(img: tf.Tensor):
  is_rgb = len(img.shape) == 3 and img.shape[-1] == 3
  plt.imshow(img, cmap='gray' if not is_rgb else None)

# ask user if they want to save the weights
def ask_to_save():
  saveYN = input('Do you want to save the weights? (y/n): ')
  if saveYN == 'y':
    model.save_weights('models/medium_mnist_weights')

network_code = '3'# input('Which network would you like to train?\n\t1. MNIST\n\t2. Shoes\n\t3. Cats and Dogs\n\t4. Faces\n')

# load data
if network_code == '1':
  (x_train, _), (x_test, _) = mnist.load_data()
  data = np.concatenate((x_train, x_test))
elif network_code == '2':
  data = np.load('data/shoes.npy')
elif network_code == '3':
  data = np.load('data/cats_dogs.npy')
elif network_code == '4':
  data = np.load('data/faces.npy')

print(data.shape)

# normalize to [-1, 1], resize to 32x32
data = preprocess(data, target_shape=(160, 160))
print(data.shape)

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

model = UNet(channels=1)
model.train(data, show_samples=False, show_losses=False, epochs=5, batch_size=2)
model.save_weights('models/custom.pkl')

plt.ion()
while True:
  samples = model.sample()
  for i, sample in enumerate(samples):
    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    imshow_rgb_safe(np.squeeze(sample))
    plt.show()
    plt.pause(0.01)
    plt.clf()
    
  final = np.array(np.clip((samples[-1][0] + 1) * 127.5, 0, 255), np.uint8)
  plt.suptitle('Final Image')
  imshow_rgb_safe(final)
  plt.show()
  plt.pause(0.1)
  plt.clf()