import tensorflow as tf
import numpy as np
from keras.layers import Conv2D
from keras import Model
import noiser
import plotter
import layers
import pickle

# Our implementation of the UNet architecture, first described in Ho et al. https://arxiv.org/pdf/2006.11239.pdf
class UNet(Model):
  def __init__(self, image_shape, num_downsamples=3, batch_size=64):
    super().__init__()
    if len(image_shape) != 3:
      raise ValueError('image_shape must be a 3-tuple in the form of (height, width, channels)')
    self.downsample_layers = []
    self.upsample_layers = []
    self.num_downsamples = num_downsamples
    self.batch_size = batch_size
    self.image_shape = image_shape

    # initialize downsampling layers
    for i in range(num_downsamples):
      num_filters = 2 ** (5 + i)
      self.downsample_layers.append([
        layers.Conv2DWithTime(num_filters, 3),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Downsample() if i < num_downsamples - 1 else layers.Identity()
      ])
    
    # initialize upsampling layers
    for i in range(num_downsamples):
      num_filters = image_shape[-1] if num_downsamples - i == 1 else 2 ** (3 + num_downsamples - i)
      self.upsample_layers.append([
        layers.Conv2DTransposeWithTime(num_filters, 3),
        layers.Residual(layers.AttentionAndGroupNorm()),
        layers.Upsample(num_filters) if i < num_downsamples - 1 else layers.Identity()
      ])

    # initialize bottleneck layers
    num_filters = 2 ** (4 + self.num_downsamples)
    self.middle_conv1 = Conv2D(num_filters, 3, padding='same')
    self.middle_conv2 = Conv2D(num_filters, 3, padding='same')

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=8e-6)

  def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
    out = []
    out.append(self.downsample_layers)
    out.append(self.upsample_layers)
    out.append(self.num_downsamples)
    out.append(self.batch_size)
    out.append(self.image_shape)
    out.append(self.middle_conv1)
    out.append(self.middle_conv2)

    pickle.dump(out, open(filepath, 'wb'))

  def load_weights(self, filepath):
    save_arr = pickle.load(open(filepath, 'rb'))
    self.downsample_layers = save_arr[0]
    self.upsample_layers = save_arr[1]
    self.num_downsamples = save_arr[2]
    self.batch_size = save_arr[3]
    self.image_shape = save_arr[4]
    self.middle_conv1 = save_arr[5]
    self.middle_conv2 = save_arr[6]

  def call(self, inputs, batch_timestep_list):
    x = inputs

    # initialize skip connections list
    skip_connections = []

    # initialize dimension paddings list, used for maintaining image shape through upsamples/downsamples
    dim_paddings = []

    # call downsampling layers
    for i, [conv, attention, downsample] in enumerate(self.downsample_layers):
      x = conv(x, batch_timestep_list)
      x = attention(x)
      skip_connections.append(x)

      # if dimensions are odd, they will be padded in the downsampling layer.
      # We need to keep track of this so that we can remove the padding in the upsampling layer
      padded_dim1 = x.shape[1] % 2 == 1
      padded_dim2 = x.shape[2] % 2 == 1
      if i < self.num_downsamples - 1:
        dim_paddings.append([padded_dim1, padded_dim2])

      x = downsample(x)

    # bottleneck
    x = self.middle_conv1(x)
    # attention not working yet
    # x = layers.Attention()(x)
    x = self.middle_conv2(x)

    # call upsampling layers
    for [conv, attention, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv(x, batch_timestep_list)
      x = attention(x)

      # undo padding from downsampling if necessary
      if len(dim_paddings) > 0:
        padded_1, padded_2 = dim_paddings.pop()
        x = upsample(x, padded_1, padded_2)
      else:
        x = upsample(x)

    # noise for image
    return x

  def get_loss(self, actual, theoretical):
    return tf.reduce_mean(tf.square(actual - theoretical))
  
  def train(self, data, epochs=5, batch_size=48, learning_rate=8e-6, show_samples=False, show_losses=False):
    if len(data.shape) != 4:
      raise ValueError('data must be a 4-tuple in the form of (num_samples, height, width, channels)')
    if show_losses or show_samples:
      plotter.activate_plots()
       
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
      num_batches = (len(data) // batch_size) + 1
      for batch in range (0, len(data), batch_size):
        # select a batch of images
        original_images = data[batch : batch + batch_size]

        # generate num_batches random timesteps
        timesteps = tf.random.uniform([len(original_images)], 1, noiser.TIMESTEPS, dtype=tf.int32)

        # generate noisy images + noise
        ### (minor_noisy_images, used_noise) = noise_images(original_images, timesteps - 1)
        (major_noisy_images, noise) = noiser.noise_images(original_images, timesteps)
        with tf.GradientTape() as tape:

          network_generated_noise = self(major_noisy_images, timesteps) # what was the noise given time + starting ?
          ### theoretical_noise = major_noisy_images - minor_noisy_images # (this was the noise)

          # get loss between predicted and actual noise
          loss = self.get_loss(network_generated_noise, noise)

        # update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        print(f'epoch: {epoch + 1} / {epochs}, batch: {(batch // batch_size) + 1} / {num_batches}, loss: {loss}')

        # show losses every batch
        if show_losses and batch:
          plotter.update_losses(loss)

        # show sample every 10 batches
        if show_samples and batch % (batch_size * 10) == 0:
          plotter.update_samples(batch, epoch, loss, original_images[0], major_noisy_images[0], network_generated_noise[0], timesteps[0])
          
        # draw if we need to draw something
        if show_losses or show_samples:
          plotter.draw_plots()

  def sample_timestep(self, x, timestep):
    offset = 1e-5
    alpha_t = noiser.ALPHAS[timestep]
    sqrt_beta_t = noiser.BETAS[timestep] ** 0.5
    alpha_bar_t = noiser.ALPHA_BAR[timestep]
    recip_sqrt_alpha_t = 1 / (alpha_t ** 0.5)

    predicted_noise = self(x, np.array([timestep]))
    noise_coefficient = (1 - alpha_t) / ((1 - alpha_bar_t + offset) ** 0.5)

    predicted_mean = recip_sqrt_alpha_t * (x - noise_coefficient * predicted_noise)
    if timestep == 0:
      return predicted_mean

    true_noise = tf.random.normal(shape=x.shape)

    return predicted_mean + sqrt_beta_t * true_noise
  
  # since the UNet learns how to predict the noise, we don't call the UNet directly to sample,
  # instead we call it to produce the predicted mean, then use that mean to statistically derive the sample
  def sample(self, num_samples=1):
    samples = []
    for _ in range(num_samples):
      w, h, c = self.image_shape
      denoised = tf.random.normal(shape=(1, w, h, c))

      for timestep in range(noiser.TIMESTEPS - 1, -1, -1):
        denoised = self.sample_timestep(denoised, timestep)
        samples.append(denoised)
    
    return samples