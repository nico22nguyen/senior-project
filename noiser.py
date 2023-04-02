import tensorflow as tf

TIMESTEPS = 200
BETAS = tf.linspace(1e-4, 6e-2, TIMESTEPS)
ALPHAS = 1 - BETAS
ALPHA_BAR = tf.math.cumprod(ALPHAS, axis=0)
ALPHA_BAR = tf.concat((tf.constant([1.]), ALPHA_BAR[:-1]), axis=0)
SQRT_ALPHA_BAR = ALPHA_BAR ** 0.5
SQRT_ONE_MINUS_ALPHA_BAR = (1 - ALPHA_BAR) ** 0.5

def noise_images(images, timesteps, starting_noise=None):
  _images = tf.cast(images, tf.float32)
  if starting_noise is None:
    noise = tf.random.normal(shape=_images.shape)
  else:
    noise = starting_noise

  # get the alphas and one_minus_alphas for each timestep
  sqrt_alpha_bar_t = tf.gather(SQRT_ALPHA_BAR, timesteps)
  sqrt_one_minus_alpha_bar_t = tf.gather(SQRT_ONE_MINUS_ALPHA_BAR, timesteps)

  # modify images based on timestep
  modified_images = tf.einsum('i,ijkl->ijkl', sqrt_alpha_bar_t, _images)

  # modify noise based on timestep
  modified_noise = tf.einsum('i,ijkl->ijkl', sqrt_one_minus_alpha_bar_t, noise)

  # return sum and noise
  return (modified_images + modified_noise, noise)