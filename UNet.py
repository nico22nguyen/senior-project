import tensorflow as tf
import numpy as np
import keras.layers as k_layers
from keras import Model
from noiser import TIMESTEPS, BETAS, alphas_cumulative, noise_images
from plotter import update_losses, update_samples, draw_plots
import layers
import matplotlib.pyplot as plt

# Our implementation of the UNet architecture, first described in Ho et al. https://arxiv.org/pdf/2006.11239.pdf
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
        k_layers.Conv2D(num_filters, 3, padding='same', activation='relu'),
        k_layers.Conv2D(num_filters, 3, padding='same', activation='relu'),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Downsample() if i < num_downsamples - 1 else layers.Identity()
      ])
    
    # initialize upsampling layers
    for i in range(num_downsamples):
      num_filters = 1 if num_downsamples - i == 1 else 2 ** (3 + num_downsamples - i)
      self.upsample_layers.append([
        k_layers.Conv2DTranspose(num_filters, 3, padding='same', activation='relu'),
        k_layers.Conv2DTranspose(num_filters, 3, padding='same', activation='relu'),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Upsample(num_filters) if i < num_downsamples - 1 else layers.Identity()
      ])

    # intialize bottleneck layers
    num_filters = 2 ** (4 + self.num_downsamples)
    self.middle_conv1 = k_layers.Conv2D(num_filters, 3, padding='same')
    self.middle_conv2 = k_layers.Conv2D(num_filters, 3, padding='same')

    # initialize time embedding layer
    self.time_embedder = layers.TimeMLP()

  def call(self, inputs, batch_timestep_list, batch_size=None):
    # add channel dimension
    x = np.expand_dims(inputs.copy(), axis=-1)

    # initialize skip connections list
    skip_connections = []

    # call downsampling layers
    for [conv1, conv2, attention, downsample] in self.downsample_layers:
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      skip_connections.append(x)
      x = downsample(x)

    # bottleneck
    x = self.middle_conv1(x)
    # attention not working yet
    # x = layers.Attention()(x)
    x = self.middle_conv2(x)

    # call upsampling layers
    for [conv1, conv2, attention, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      x = upsample(x)
    
    # pad with zeros if we don't have batch_size samples in the batch, necessary because the input size of the MLP is constant
    # so the last batch will throw an error expecting there to be batch_size samples
    if batch_size is not None and len(batch_timestep_list < batch_size):
      batch_timestep_list = tf.concat([
        batch_timestep_list,
        tf.zeros(batch_size - len(batch_timestep_list), dtype=tf.dtypes.int32)
      ], axis=0)

    # get timestep embeddings
    time_embedding = self.time_embedder(batch_timestep_list)
    # remove zeros if we padded, else this line does nothing (x.shape[0] == time_embedding.shape[0])
    time_embedding = time_embedding[:x.shape[0]]
    
    # remove channel dimension added at the beginning of the function
    x_no_channel_dim = tf.squeeze(x, axis=-1)

    return x_no_channel_dim + time_embedding
  
  def train(self, data, epochs=5, batch_size=32, learning_rate=1e-6, show_samples=False, show_losses=True):
    for epoch in range(epochs):
      for batch in range (0, len(data), batch_size):
        # select a batch of images
        image_batch = data[batch : batch + batch_size]

        # generate num_batches random timesteps
        timesteps = tf.random.uniform([len(image_batch)], 0, TIMESTEPS - 1, dtype=tf.int32)

        with tf.GradientTape() as tape:
          # call the network to produce the outputs
          predicted_noise = self.call(image_batch, timesteps, batch_size)

          # get what the noise should be
          actual_noise = noise_images(image_batch, timesteps)

          # get loss between predicted and actual noise
          loss = tf.reduce_mean(tf.square(predicted_noise - actual_noise))

        # update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        print(f'epoch: {epoch}, batch: {batch}, loss: {loss}')

        # show losses every batch
        if show_losses:
          update_losses(loss)

        # show sample every 10 batches
        if show_samples and batch % (batch_size * 10) == 0:
          update_samples(batch, loss, image_batch[0], actual_noise[0], predicted_noise[0], timesteps[0])
          
        # draw if we need to draw something
        if show_losses or show_samples:
          draw_plots()

  def sample_timestep(self, x, timestep):
    sqrt_recip_alphas = ((1 / (1. - BETAS)) ** 0.5)

    beta = BETAS[timestep]
    alpha = alphas_cumulative[timestep]
    sqrt_recip_alpha = sqrt_recip_alphas[timestep]

    predicted_mean = sqrt_recip_alpha * (x - beta * self.call([x], timestep) / alpha)

    if timestep == 0:
      return predicted_mean
    
    noise = tf.random.normal(shape=x.shape)
    return predicted_mean + beta * noise
  
  # since the UNet learns how to predict the noise, we don't call the UNet directly to sample,
  # instead we call it to produce the predicted mean, then use that mean to statistically derive the sample
  def sample(self, shape):
    denoised = tf.random.normal(shape=shape)
    _, axs = plt.subplots(10, 10)
    for timestep in range(TIMESTEPS - 1, -1, -1):
      denoised = self.sample_timestep(denoised, timestep)[0]

      axs[timestep // 10][timestep % 10].imshow(denoised, cmap='gray')
    plt.show()
    return denoised

# generates and displays samples from the generator
def show_progress(model, subplots, epoch=0, batch=0):
  fig, axes = subplots
  sample = model.sample(np.random.normal(size=(9, 28, 28)))

  if epoch > 0:
    fig.suptitle('Epoch {}, Batch {} samples'.format(epoch, batch))

  for img, axis in zip(sample, np.ndarray.flatten(axes)):
    axis.imshow(img, cmap='gray')

  plt.pause(0.05)
  plt.show(block=False)
