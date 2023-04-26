import math
import keras.layers as layers
import tensorflow as tf
import tensorflow_addons as tfa

class Sequential(layers.Layer):
  def __init__(self, layers):
    super().__init__()
    self.layers = layers

  def call(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class SiLU(layers.Layer):
  def __init__(self):
    super(SiLU, self).__init__()

  def call(self, x):
    return x * tf.nn.sigmoid(x)
  
class GELU(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, x: tf.Tensor):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))
  
# function courtesy of Vedant Jumle https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac
class SinusoidalPosEmb(layers.Layer):
  def __init__(self, dim, max_positions=10000):
    super(SinusoidalPosEmb, self).__init__()
    self.dim = dim
    self.max_positions = max_positions

  def call(self, x):
    x = tf.cast(x, tf.float32)
    half_dim = self.dim // 2
    emb = math.log(self.max_positions) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = x[:, None] * emb[None, :]

    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

    return emb

class GroupNorm(layers.Layer):
  def __init__(self, num_groups=1):
    super().__init__()
    self.group_norm = tfa.layers.GroupNormalization(num_groups)

  def call(self, inputs):
    return self.group_norm(inputs)

class Residual(layers.Layer):
  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def call(self, inputs):
    return self.layer(inputs) + inputs

class Downsample(layers.Layer):
  def __init__(self):
    super().__init__()
    self.max_pool = layers.MaxPool2D((2, 2))

  def call(self, inputs):
    return self.max_pool(inputs)

class Upsample(layers.Layer):
  def __init__(self, filters):
    super().__init__()
    self.filters = filters
    self.conv_transpose = layers.Conv2DTranspose(filters, 1, 2)

  def call(self, inputs):
    return self.conv_transpose(inputs)

class Identity(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    return inputs
  
class Block(layers.Layer):
  def __init__(self, num_filters, groups=8):
    super().__init__()
    self.conv = layers.Conv2D(num_filters, 3, padding='same')
    self.group_norm = tfa.layers.GroupNormalization(groups, epsilon=1e-5)
    self.activation = SiLU()

  def call(self, inputs, gamma_beta=None):
    x = self.conv(inputs)
    x = self.group_norm(x)

    if gamma_beta is not None:
      gamma, beta = gamma_beta
      x = x * (gamma + 1) + beta

    x = self.activation(x)
    return x
  
class ResnetBlock(layers.Layer):
  def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
    super().__init__()

    self.time_mlp = Sequential([SiLU(), layers.Dense(dim_out * 2)]) if time_emb_dim is not None else None
    self.block1 = Block(dim_out, groups)
    self.block2 = Block(dim_out, groups)
    self.residual_conv = layers.Conv2D(dim_out, 1, strides=1) if dim != dim_out else Identity()

  def call(self, inputs, time_embedding=None):
    gamma_beta=None

    if self.time_mlp is not None and time_embedding is not None:
      time_embedding = self.time_mlp(time_embedding)
      time_embedding = tf.expand_dims(tf.expand_dims(time_embedding, axis=1), axis=1)
      gamma_beta = tf.split(time_embedding, 2, axis=-1)
    
    x = self.block1(inputs, gamma_beta)
    x = self.block2(x)

    return x + self.residual_conv(inputs)