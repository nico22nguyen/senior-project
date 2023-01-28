import random
import tensorflow as tf
import numpy as np
import keras.layers as k_layers
from keras import Model
from diffusion import timesteps, betas, alphas_cumulative, noise_image
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

  def call(self, inputs, timestep):
    # add timestamp embedding
    x = np.expand_dims(inputs.copy(), axis=-1)
    x[-1] = timestep
    
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
    
    # remove timestamp embedding
    return tf.squeeze(x, axis=-1)
  
  def train(self, data, epochs=5, batch_size=32, learning_rate=0.001):
    for epoch in range(epochs):
      for batch in range (0, len(data), batch_size):
        x = data[batch : batch + batch_size]
        timestep = random.randint(0, timesteps - 1)
        with tf.GradientTape() as tape:
          predicted_noise = self.call(x, timestep)
          actual_noise = noise_image(x, timestep)
          loss = tf.reduce_mean(tf.square(predicted_noise - actual_noise))
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        print(f'epoch: {epoch}, batch: {batch}, loss: {loss}')

  def sample_timestep(self, x, timestep):
    sqrt_recip_alphas = ((1 / (1. - betas)) ** 0.5)

    beta = betas[timestep]
    alpha = alphas_cumulative[timestep]
    sqrt_recip_alpha = sqrt_recip_alphas[timestep]

    predicted_mean = sqrt_recip_alpha * (x - beta * self.call(x, timestep) / alpha)

    if timestep == 0:
      return predicted_mean
    
    noise = tf.random.normal(shape=x.shape)
    return predicted_mean + beta * noise
  
  # since the UNet learns how to predict the noise, we don't call the UNet directly to sample,
  # instead we call it to produce the predicted mean, then use that mean to statistically derive the sample
  def sample(self, x):
    denoised = None
    for timestep in range(timesteps - 1, -1, -1):
      denoised = self.sample_timestep(x, timestep)
    return denoised