import pickle
import tensorflow as tf
import keras.layers as nn
from keras import Model
from functools import partial
import plotter
import noiser
import medium_code as layers
import numpy as np

IMAGE_SHAPE = (32, 32, 1)

# unet __init__ definition from medium
# call, train, sample, save are custom
class UNet(Model):
  def __init__(self,
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    resnet_block_groups=8,
  ):
    super().__init__()
    
    # determine dimensions
    self.channels = channels
    
    init_dim = dim // 3 * 2
    self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')
    
    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))
    
    block_klass = partial(layers.ResnetBlock, groups = resnet_block_groups)
    
    # time embeddings
    time_dim = dim * 4
    
    self.time_mlp = layers.Sequential([
        layers.SinusoidalPosEmb(dim),
        nn.Dense(units=time_dim),
        layers.GELU(),
        nn.Dense(units=time_dim)
    ])
    
    # layers
    self.downs = []
    self.ups = []
    num_resolutions = len(in_out)
    
    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)

      self.downs.append([
        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
        layers.Residual(layers.PreNorm(dim_out, layers.LinearAttention(dim_out))),
        layers.Downsample(dim_out) if not is_last else layers.Identity()
      ])

    mid_dim = dims[-1]
    self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_attn = layers.Residual(layers.PreNorm(mid_dim, layers.Attention(mid_dim)))
    self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
    
    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= (num_resolutions - 1)

      self.ups.append([
        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        layers.Residual(layers.PreNorm(dim_in, layers.LinearAttention(dim_in))),
        layers.Upsample(dim_in) if not is_last else layers.Identity()
      ])
    
    self.out_dim = channels
    
    self.final_conv = layers.Sequential([
      block_klass(dim * 2, dim),
      nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
    ], name="output")

  def call(self, inputs, batch_timestep_list):
    x = inputs
    time_embedding = self.time_mlp(batch_timestep_list)

    # initialize skip connections list
    skip_connections = []

    # initialize dimension paddings list, used for maintaining image shape through upsamples/downsamples
    dim_paddings = []

    # call downsampling layers
    for i, [conv1, conv2, attention, downsample] in enumerate(self.downs):
      x = conv1(x, time_embedding)
      x = conv2(x, time_embedding)
      x = attention(x)
      skip_connections.append(x)

      # if dimensions are odd, they will be padded in the downsampling layer.
      # We need to keep track of this so that we can remove the padding in the upsampling layer
      padded_dim1 = x.shape[1] % 2 == 1
      padded_dim2 = x.shape[2] % 2 == 1
      if i < len(self.downs) - 1:
        dim_paddings.append([padded_dim1, padded_dim2])

      x = downsample(x)

    # bottleneck
    x = self.mid_block1(x, time_embedding)
    x = self.mid_attn(x)
    x = self.mid_block2(x, time_embedding)

    # call upsampling layers
    for [conv1, conv2, attention, upsample] in self.ups:
      x = x + skip_connections.pop() # maybe use tf.concat axis=-1 here instead
      x = conv1(x, time_embedding)
      x = conv2(x, time_embedding)
      x = attention(x)

      # undo padding from downsampling if necessary
      if len(dim_paddings) > 0:
        padded_1, padded_2 = dim_paddings.pop()
        x = upsample(x)
      else:
        x = upsample(x)

    # final conv
    x = x + skip_connections.pop()
    x = self.final_conv(x)

    # noise for image
    return x
  

  def get_loss(self, actual, theoretical):
    return tf.reduce_mean(tf.square(actual - theoretical))
  
  # NOTE: my GPU can't handle batch_size >= 64, it runs out of memory
  def train(self, data: tf.Tensor, epochs=5, batch_size=48, learning_rate=8e-6, show_samples=False, show_losses=False):
    if len(data.shape) != 4:
      raise ValueError('data must be a 4-tuple in the form of (num_samples, height, width, channels)')
    if show_losses or show_samples:
      plotter.activate_plots()
       
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
      progress_bar = tf.keras.utils.Progbar(data.shape[0] // batch_size)
      losses = []
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
        
        losses.append(loss)
        progress_bar.update(batch // batch_size, values=[("loss", loss)])

        # show losses every batch
        if show_losses and batch:
          plotter.update_losses(loss)

        # show sample every 10 batches
        if show_samples and batch % (batch_size * 10) == 0:
          plotter.update_samples(batch, epoch, loss, original_images[0], major_noisy_images[0], network_generated_noise[0], timesteps[0])
          
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
      w, h, c = IMAGE_SHAPE
      denoised = tf.random.normal(shape=(1, w, h, c))

      for timestep in range(noiser.TIMESTEPS - 1, -1, -1):
        denoised = self.sample_timestep(denoised, timestep)
        samples.append(denoised)
    
    return samples
  
  def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
    out = []
    out.append(self.channels)
    out.append(self.init_conv)
    out.append(self.time_mlp)
    out.append(self.downs)
    out.append(self.mid_block1)
    out.append(self.mid_attn)
    out.append(self.mid_block2)
    out.append(self.ups)
    out.append(self.final_conv)

    pickle.dump(out, open(filepath, 'wb'))

  def load_weights(self, filepath):
    save_arr: list = pickle.load(open(filepath, 'rb'))
    self.final_conv = save_arr.pop()
    self.ups = save_arr.pop()
    self.mid_block2 = save_arr.pop()
    self.mid_attn = save_arr.pop()
    self.mid_block1 = save_arr.pop()
    self.downs = save_arr.pop()
    self.time_mlp = save_arr.pop()
    self.init_conv = save_arr.pop()
    self.channels = save_arr.pop()