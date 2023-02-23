import tensorflow as tf

TIMESTEPS = 100
BETAS = tf.linspace(0.0001, 0.02, TIMESTEPS)
alphas_cumulative = tf.math.cumprod(1 - BETAS, axis=0)
alphas_cumulative = tf.concat((tf.constant([1.]), alphas_cumulative[:-1]), axis=0)
sqrt_alphas = alphas_cumulative ** 0.5
sqrt_one_minus_alphas = (1 - alphas_cumulative) ** 0.5

def noise_images(images, _timesteps):
  _images = tf.cast(images, tf.float32)
  ones = tf.ones_like(_images, dtype=tf.float32)
  random_normal = tf.random.normal(shape=_images.shape, mean=0, stddev=ones, dtype=tf.float32)

  # get the alphas and one_minus_alphas for each timestep
  selected_sqrt_alphas = tf.gather(sqrt_alphas, _timesteps)
  selected_sqrt_one_minus_alphas = tf.gather(sqrt_one_minus_alphas, _timesteps)

  # modify images based on timestep
  modified_images = tf.einsum('i,ijk->ijk', selected_sqrt_alphas, _images)

  # modify noise based on timestep
  modified_noise = tf.einsum('i,ijk->ijk', selected_sqrt_one_minus_alphas, random_normal)

  # return sum
  return modified_images + modified_noise