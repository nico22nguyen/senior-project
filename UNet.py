import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Dense
from keras import Model
import noiser
import plotter
import layers
import pickle

# Our implementation of the UNet architecture, first described in Ho et al. https://arxiv.org/pdf/2006.11239.pdf
class UNet(Model):
  def __init__(self, channel_increase_per_downsample=64, dim_multipliers=(1, 2, 4, 8), channels=3, image_shape=(32, 32, 1)):
    super().__init__()
    
    for i in range(len(dim_multipliers))[1:]:
      if dim_multipliers[i] <= dim_multipliers[i - 1]:
        raise ValueError('dim_multipliers must be strictly increasing')

    self.max_dim_multiplier = dim_multipliers[-1]
    self.channels = channels
    self.downsample_layers = []
    self.upsample_layers = []
    self.image_shape = image_shape

    init_dim = channel_increase_per_downsample // 3 * 2
    self.init_conv = Conv2D(init_dim, 7, padding='same')

    dims = [init_dim]
    for multiplier in dim_multipliers:
      dims.append(channel_increase_per_downsample * multiplier)

    dim_in_dim_out = list(zip(dims[:-1], dims[1:]))

    time_dim = channel_increase_per_downsample * 4

    self.time_mlp = layers.Sequential([
      layers.SinusoidalPosEmb(channel_increase_per_downsample),
      Dense(time_dim),
      layers.GELU(),
      Dense(time_dim)
    ])

    # initialize downsampling layers
    for i, (dim_in, dim_out) in enumerate(dim_in_dim_out):
      is_last_layer = i == len(dim_in_dim_out) - 1

      self.downsample_layers.append([
        layers.ResnetBlock(dim_in, dim_out, time_dim),
        layers.ResnetBlock(dim_out, dim_out, time_dim),
        layers.Residual(layers.GroupNorm()),
        layers.Downsample() if not is_last_layer else layers.Identity()
      ])

    mid_dim = dims[-1] # highest dimension (number of channels)
    self.mid_block1 = layers.ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_block2 = layers.ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
    
    # initialize upsampling layers
    for i, (dim_in, dim_out) in enumerate(reversed(dim_in_dim_out[1:])):
      self.upsample_layers.append([
        layers.ResnetBlock(dim_out * 2, dim_in, time_dim),
        layers.ResnetBlock(dim_in, dim_in, time_dim),
        layers.Residual(layers.GroupNorm()),
        layers.Upsample(dim_in)
      ])
      
    self.final_channel_dim = channels
    self.final_resnet = layers.ResnetBlock(channel_increase_per_downsample * 2, channel_increase_per_downsample)
    self.final_conv = Conv2D(channels, 1, padding='same')

  def call(self, inputs, batch_timestep_list):
    if inputs.shape[1] % self.max_dim_multiplier != 0 or inputs.shape[2] % self.max_dim_multiplier != 0:
      raise ValueError(f'input height and width must be a multiple of the largest dimension multiplier: {self.max_dim_multiplier}')
  
    x = self.init_conv(inputs)
    time_embedding = self.time_mlp(batch_timestep_list)

    # initialize skip connections list
    skip_connections = []

    # call downsampling layers
    for [conv1, conv2, groupnorm, downsample] in self.downsample_layers:
      x = conv1(x, time_embedding)
      x = conv2(x, time_embedding)
      x = groupnorm(x)
      skip_connections.append(x)
      x = downsample(x)

    # bottleneck
    x = self.mid_block1(x, time_embedding)
    x = self.mid_block2(x, time_embedding)

    # call upsampling layers
    for [conv1, conv2, groupnorm, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv1(x, time_embedding)
      x = conv2(x, time_embedding)
      x = groupnorm(x)
      x = upsample(x)

    # final convolutions
    x = x + skip_connections.pop()
    x = self.final_resnet(x)
    x = self.final_conv(x)

    # noise for image
    return x

  def get_loss(self, actual, theoretical):
    return tf.reduce_mean(tf.square(actual - theoretical))
  
  def train(self, data: tf.Tensor, epochs=5, batch_size=48, learning_rate=8e-6, show_samples=False, show_losses=False):
    if len(data.shape) != 4:
      raise ValueError('data must be a 4-tuple in the form of (num_samples, height, width, channels)')
    if show_losses or show_samples:
      plotter.activate_plots()
       
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
      progress_bar = tf.keras.utils.Progbar((data.shape[0] // batch_size) + 1)
      losses = []
      for batch in range (0, len(data), batch_size):
        # select a batch of images
        original_images = data[batch : batch + batch_size]

        # generate num_batches random timesteps
        timesteps = tf.random.uniform([len(original_images)], 1, noiser.TIMESTEPS, dtype=tf.int32)

        # generate noisy images + noise
        (noised_images, noise) = noiser.noise_images(original_images, timesteps)
        with tf.GradientTape() as tape:
          network_generated_noise = self(noised_images, timesteps)

          # get loss between predicted and actual noise
          loss = self.get_loss(network_generated_noise, noise)

        # update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        losses.append(loss)
        progress_bar.update((batch // batch_size) + 1, values=[("loss", loss)])

        # show losses every batch
        if show_losses and batch:
          plotter.update_losses(loss)

        # show sample every 10 batches
        if show_samples and batch % (batch_size * 10) == 0:
          plotter.update_samples(batch, epoch, loss, original_images[0], noised_images[0], network_generated_noise[0], timesteps[0])
          
        # draw if we need to draw something
        if show_losses or show_samples:
          plotter.draw_plots()

      # show avg loss for each epoch
      print(f'Epoch {epoch + 1} / {epochs} - Average Loss: {np.mean(losses):.4f}')

  def sample_timestep(self, x: tf.Tensor, timestep):
    offset = 1e-5
    alpha_t = noiser.ALPHAS[timestep]
    sqrt_beta_t = noiser.BETAS[timestep] ** 0.5
    alpha_bar_t = noiser.ALPHA_BAR[timestep]
    recip_sqrt_alpha_t = 1 / (alpha_t ** 0.5)

    predicted_noise = self(x, tf.constant([timestep]))
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
  
  def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
    out = {
      'channels': self.channels,
      'downsample_layers': self.downsample_layers,
      'upsample_layers': self.upsample_layers,
      'init_conv': self.init_conv,
      'time_mlp': self.time_mlp,
      'mid_block1': self.mid_block1,
      'mid_block2': self.mid_block2,
      'final_resnet': self.final_resnet,
      'final_conv': self.final_conv
    }

    pickle.dump(out, open(filepath, 'wb'))

  def load_weights(self, filepath):
    save_dict = pickle.load(open(filepath, 'rb'))

    self.channels = save_dict['channels']
    self.downsample_layers = save_dict['downsample_layers']
    self.upsample_layers = save_dict['upsample_layers']
    self.init_conv = save_dict['init_conv']
    self.time_mlp = save_dict['time_mlp']
    self.mid_block1 = save_dict['mid_block1']
    self.mid_block2 = save_dict['mid_block2']
    self.final_resnet = save_dict['final_resnet']
    self.final_conv = save_dict['final_conv']
    