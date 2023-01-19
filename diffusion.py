import tensorflow as tf
import keras.layers as layers
from keras import Model
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import helpers

class UNet(Model):
  def __init__(self, input_shape, num_downsamples=3):
    super().__init__()
    self.downsample_layers = []
    self.upsample_layers = []
    self.num_downsamples = num_downsamples

    self.initial_shape = input_shape[1]
    self.min_shape = self.initial_shape - 2 ** num_downsamples

    # initialize downsampling layers
    for i in range(num_downsamples):
      num_filters = 2 ** (5 + i)
      self.downsample_layers.append([
        layers.Conv2D(num_filters, 3, padding='same'),
        layers.Conv2D(num_filters, 3, padding='same'),
        helpers.Residual(helpers.AttentionAndGroupNorm()),
        helpers.Downsample() if i < num_downsamples - 1 else helpers.Identity()
      ])
    
    # initialize upsampling layers
    for i in range(num_downsamples):
      num_filters = 1 if num_downsamples - i == 1 else 2 ** (3 + num_downsamples - i)
      self.upsample_layers.append([
        layers.Conv2DTranspose(num_filters, 3, padding='same'),
        layers.Conv2DTranspose(num_filters, 3, padding='same'),
        helpers.Residual(helpers.AttentionAndGroupNorm()),
        helpers.Upsample(num_filters) if i < num_downsamples - 1 else helpers.Identity()
      ])

  def call(self, inputs):
    x = inputs
    skip_connections = []

    # call downsampling layers
    for [conv1, conv2, attention, downsample] in self.downsample_layers:
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      skip_connections.append(x)
      x = downsample(x)

    # bottleneck
    num_filters = 2 ** (4 + self.num_downsamples)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    # attention not working yet
    # x = layers.Attention()(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)

    # call upsampling layers
    for [conv1, conv2, attention, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      x = upsample(x)
    
    return x

timesteps = 100
betas = tf.linspace(0., 1, timesteps)
alphas_cumulative = tf.math.cumprod(1 - betas, axis=0) ** 0.5
def noise_image(image, timestep):
  ones = tf.ones_like(image, dtype=tf.float32)
  random_normal = tf.random.normal(shape=image.shape, mean=0, stddev=ones, dtype=tf.float32)

  mu = (alphas_cumulative ** 0.5)[timestep] * image
  sigma = (1 - alphas_cumulative[timestep]) * random_normal

  return tf.random.normal(shape=image.shape, mean=mu, stddev=sigma, dtype=tf.float32)

def main():
  (x_train, _), (x_test, y_test) = mnist.load_data()
  x_train = x_train / 255 # tf.expand_dims(x_train, axis=-1) / 255

  plt.figure()
  plt.imshow(x_train[0, :, :, 0], cmap='gray')
  model = UNet(x_train.shape)
  out = model.call(x_train[:1])

  plt.figure()
  plt.imshow(out[0, :, :, 0], cmap='gray')
  plt.show()