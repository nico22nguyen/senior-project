import keras.layers as layers
import tensorflow_addons as tfa
import tensorflow as tf

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
    return layers.MaxPool2D((2, 2))(inputs)

class Upsample(layers.Layer):
  def __init__(self, filters):
    super().__init__()
    self.filters = filters

  def call(self, inputs):
    return layers.Conv2DTranspose(self.filters, 1, 2)(inputs)

class Identity(layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
    return inputs
  
class TimeMLP(layers.Layer):
  def __init__(self):
    super().__init__()
    # consider using silu here instead
    self.activation = layers.ReLU()
    self.embedder = layers.Dense(28 * 28)

  def call(self, timestep):
    x = self.activation(timestep)
    time_vector = self.embedder(tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1))
    return tf.reshape(time_vector, (28, 28))