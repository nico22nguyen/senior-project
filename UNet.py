import keras.layers as k_layers
from keras import Model
import layers

class UNet(Model):
  def __init__(self, num_downsamples=3):
    super().__init__()
    self.downsample_layers = []
    self.upsample_layers = []
    self.num_downsamples = num_downsamples

    # initialize downsampling layers
    for i in range(num_downsamples):
      num_filters = 2 ** (5 + i)
      self.downsample_layers.append([
        k_layers.Conv2D(num_filters, 3, padding='same'),
        k_layers.Conv2D(num_filters, 3, padding='same'),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Downsample() if i < num_downsamples - 1 else layers.Identity()
      ])
    
    # initialize upsampling layers
    for i in range(num_downsamples):
      num_filters = 1 if num_downsamples - i == 1 else 2 ** (3 + num_downsamples - i)
      self.upsample_layers.append([
        k_layers.Conv2DTranspose(num_filters, 3, padding='same'),
        k_layers.Conv2DTranspose(num_filters, 3, padding='same'),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Upsample(num_filters) if i < num_downsamples - 1 else layers.Identity()
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
    x = k_layers.Conv2D(num_filters, 3, padding='same')(x)
    # attention not working yet
    # x = layers.Attention()(x)
    x = k_layers.Conv2D(num_filters, 3, padding='same')(x)

    # call upsampling layers
    for [conv1, conv2, attention, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      x = upsample(x)
    
    return x