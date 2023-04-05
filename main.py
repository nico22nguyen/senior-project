import keras.datasets.mnist as mnist
from UNet import UNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import noiser

def preprocess(x: tf.Tensor, target_shape=(32, 32)):
  normalized: tf.Tensor = tf.cast(x, tf.float32) / 127.5 - 1
  while len(normalized.shape) < 4:
    normalized = tf.expand_dims(normalized, axis=-1)
  return tf.image.resize(normalized, target_shape)

# ask user if they want to save the weights
def ask_to_save():
  saveYN = input('Do you want to save the weights? (y/n): ')
  if saveYN == 'y':
    model.save_weights('models/medium_mnist_weights')

network_code = '4'# input('Which network would you like to train?\n\t1. MNIST\n\t2. Shoes\n\t3. Cats and Dogs\n\t4. Faces\n')

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

print(np.min(data), np.max(data))
exit()

# normalize to [-1, 1], resize to 32x32
data = preprocess(data, target_shape=(256, 256))
print(data.shape)

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

model = UNet(channels=1)
model.train(data, show_samples=False, show_losses=False, epochs=5)
model.save_weights('models/custom.pkl')

plt.ion()
while True:
  samples = model.sample()
  for i, sample in enumerate(samples):
    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    plt.imshow(np.squeeze(sample), cmap='gray')
    plt.show()
    plt.pause(0.01)
    plt.clf()
    
  final = np.array(np.clip((samples[-1][0] + 1) * 127.5, 0, 255), np.uint8)
  plt.suptitle('Final Image')
  plt.imshow(final, cmap='gray')
  plt.show()
  plt.pause(0.1)
  plt.clf()