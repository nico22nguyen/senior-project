import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from UNet import UNet

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255 # tf.expand_dims(x_train, axis=-1) / 255

model = UNet()

model.train(x_train, epochs=1, batch_size=64)
sample = model.sample(tf.random.normal(shape=(9, 28, 28)))

fig, axes = plt.subplots(3, 3)
for img, axis in zip(sample, np.ndarray.flatten(axes)):
  axis.imshow(img, cmap='gray')
  
plt.show()