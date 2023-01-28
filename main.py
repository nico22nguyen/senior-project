import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from diffusion import UNet

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255 # tf.expand_dims(x_train, axis=-1) / 255

model = UNet()
out = model.call(tf.reshape(x_train[:1], [1, 28, 28, 1]))

plt.figure()
plt.imshow(out[0, :, :, 0], cmap='gray')
plt.show()