import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
from UNet import UNet
import tensorflow as tf
import numpy as np

shoes = np.load('data/shoes.npy')
cats_dogs = np.load('data/cats_dogs.npy')
faces = np.load('data/faces.npy')

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255

model = UNet()
model.train(x_train, epochs=3, batch_size=16, show_samples=True, show_losses=True)

# trained
"""
denoised = tf.random.normal(shape=x_train[0].shape)
fig, axs = plt.subplots(10, 10)
for timestep in range(100 - 1, -1, -1):
  denoised = model.sample_timestep(denoised, timestep)[0]

  axs[timestep // 10][timestep % 10].imshow(denoised, cmap='gray')

plt.show()
"""