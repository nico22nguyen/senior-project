import keras.datasets.mnist as mnist
from Frankenstein import UNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import noiser

def preprocess(x):
  normalized = tf.cast(x, tf.float32) / 127.5 - 1
  return tf.image.resize(tf.expand_dims(normalized, axis=-1), (32, 32))

# ask user if they want to save the weights
def ask_to_save():
  saveYN = input('Do you want to save the weights? (y/n): ')
  if saveYN == 'y':
    model.save_weights('models/medium_mnist_weights')

(x_train, _), (x_test, _) = mnist.load_data()
data = np.concatenate((x_train, x_test))[:6000]

# normalize to [-1, 1]
data = preprocess(data)

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

model = UNet(channels=1)
model.train(data, epochs=5)
model.save_weights('models/latest.pkl')

plt.ion()
while True:
  img_list = model.sample()
  for sample in img_list:
    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255)), cmap='gray')
    plt.show()
    plt.pause(0.01)
    plt.clf()
  """
  for i in tqdm(range(noiser.TIMESTEPS-1)):
    t = np.expand_dims(np.array(noiser.TIMESTEPS-i-1, np.int32), 0)
    pred_noise = model(x, t)
    x = ddpm(x, pred_noise, t)
    img_list.append(np.squeeze(np.squeeze(x, 0),-1))

    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255)), cmap='gray')
    plt.show()
    plt.pause(0.01)
    plt.clf()
  """