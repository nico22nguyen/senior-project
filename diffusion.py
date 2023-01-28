import tensorflow as tf

timesteps = 100
betas = tf.linspace(0., 1, timesteps)
alphas_cumulative = tf.math.cumprod(1 - betas, axis=0) ** 0.5
def noise_image(image, timestep):
  ones = tf.ones_like(image, dtype=tf.float32)
  random_normal = tf.random.normal(shape=image.shape, mean=0, stddev=ones, dtype=tf.float32)

  mu = (alphas_cumulative ** 0.5)[timestep] * image
  sigma = (1 - alphas_cumulative[timestep]) * random_normal

  return tf.random.normal(shape=image.shape, mean=mu, stddev=sigma, dtype=tf.float32)