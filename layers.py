import keras.layers as layers
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

range = np.linspace(0,np.pi,100)
embeddings= tf.stack((tf.math.sin(range), tf.math.cos(range)))

# couldn't get attention to work
class AttentionAndGroupNorm(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    num_groups = 1 # int(inputs.shape[1] / 2)
    x = inputs # layers.Attention()(inputs)
    x = tfa.layers.GroupNormalization(num_groups)(x)
    return x

class Residual(layers.Layer):
  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def call(self, inputs):
    return self.layer(inputs) + inputs

class Downsample(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    x = inputs
    # make sure the input is even so that it can be upsampled properly
    if x.shape[1] % 2 != 0:
      x = layers.ZeroPadding2D(((0, 1), (0, 0)))(x)
    if x.shape[2] % 2 != 0:
      x = layers.ZeroPadding2D(((0, 0), (0, 1)))(x)

    x = layers.MaxPool2D((2, 2))(x)
    
    return x

class Upsample(layers.Layer):
  def __init__(self, filters):
    super().__init__()
    self.filters = filters

  def call(self, inputs, padded_1=False, padded_2=False):
    x = layers.Conv2DTranspose(self.filters, 1, 2)(inputs)

    # remove padding added in downsampling layer if applicable
    if padded_1:
      x = x[:, :-1, :, :]
    if padded_2:
      x = x[:, :, :-1, :]
    
    return x

class Identity(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    return inputs
  
class TimeMLP(layers.Layer):
  def __init__(self, num_channels):
    super().__init__()
    self.dense = layers.Dense(num_channels)

  def call(self, timestep_list):
    embedded_timesteps = tf.transpose(tf.gather(embeddings, timestep_list, axis=1))

    # calculate embeddings
    time_vector = self.dense(embedded_timesteps)

    # expand list to be added with batch of images
    return tf.expand_dims(tf.expand_dims(time_vector, axis=1), axis=1)
  
class Conv2DWithTime(layers.Layer):
  def __init__(self, num_filters, kernel_size):
    super().__init__()
    self.conv1 = layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu')
    self.time1 = TimeMLP(num_filters)
    self.conv2 = layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu')
    self.time2 = TimeMLP(num_filters)

  def call(self, inputs, timestep_list):
    conv1_out = self.conv1(inputs)
    t_e1 = self.time1(timestep_list)
    x = conv1_out + t_e1

    conv2_out = self.conv2(x)
    t_e2 = self.time1(timestep_list)
    x = conv2_out + t_e2

    return x
class Conv2DTransposeWithTime(layers.Layer):
  def __init__(self, num_filters, kernel_size):
    super().__init__()
    self.conv1 = layers.Conv2DTranspose(num_filters, kernel_size, padding='same', activation='relu')
    self.time1 = TimeMLP(num_filters)
    self.conv2 = layers.Conv2DTranspose(num_filters, kernel_size, padding='same', activation='relu')
    self.time2 = TimeMLP(num_filters)

  def call(self, inputs, timestep_list):
    conv1_out = self.conv1(inputs)
    t_e1 = self.time1(timestep_list)
    x = conv1_out + t_e1

    conv2_out = self.conv2(x)
    t_e2 = self.time1(timestep_list)
    x = conv2_out + t_e2

    return x