import tensorflow as tf
import keras.layers as k_layers
from keras import Model
from noiser import TIMESTEPS, BETAS, alphas_cumulative, noise_images
from plotter import update_losses, update_samples, draw_plots
import layers

# Our implementation of the UNet architecture, first described in Ho et al. https://arxiv.org/pdf/2006.11239.pdf
class UNet(Model):
  def __init__(self, image_shape, num_downsamples=3):
    super().__init__()
    if len(image_shape) != 3:
      raise ValueError('image_shape must be a 3-tuple in the form of (height, width, channels)')
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
      num_filters = image_shape[-1] if num_downsamples - i == 1 else 2 ** (3 + num_downsamples - i)
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
    self.time_embedder = layers.TimeMLP(image_shape=image_shape)

  def call(self, inputs, batch_timestep_list, batch_size):
    x = inputs.copy()

    # initialize skip connections list
    skip_connections = []

    # initialize dimension paddings list, used for maintaining image shape through upsamples/downsamples
    dim_paddings = []

    # call downsampling layers
    for i, [conv1, conv2, attention, downsample] in enumerate(self.downsample_layers):
      x = conv1(x)
      x = conv2(x)
      x = attention(x)
      skip_connections.append(x)

      # if dimensions are odd, they will be padded in the downsampling layer. We need to keep track of this so that we can remove the padding in the upsampling layer
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
    for [conv1, conv2, attention, upsample] in self.upsample_layers:
      x = x + skip_connections.pop()
      x = conv1(x)
      x = conv2(x)
      x = attention(x)

      # undo padding from downsampling if necessary
      if len(dim_paddings) > 0:
        padded_1, padded_2 = dim_paddings.pop()
        x = upsample(x, padded_1, padded_2)
      else:
        x = upsample(x)
    
    # pad with zeros if we don't have batch_size samples in the batch, necessary because the input size of the MLP is constant
    # so the last batch will throw an error expecting there to be batch_size samples
    if len(batch_timestep_list < batch_size):
      padding = tf.zeros(batch_size - len(batch_timestep_list), dtype=tf.dtypes.int32)
      batch_timestep_list = tf.concat([batch_timestep_list, padding], axis=0)

    # get timestep embeddings
    time_embedding = self.time_embedder(batch_timestep_list)
    # remove zeros if we padded, else this line does nothing (x.shape[0] == time_embedding.shape[0])
    time_embedding = time_embedding[:x.shape[0]]

    return x + time_embedding
  
  def train(self, data, epochs=5, batch_size=32, learning_rate=1e-6, show_samples=False, show_losses=True):
    if len(data.shape) != 4:
      raise ValueError('data must be a 4-tuple in the form of (num_samples, height, width, channels)')
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

    for timestep in range(TIMESTEPS - 1, -1, -1):
      denoised = self.sample_timestep(denoised, timestep)
    
    return denoised