import tensorflow as tf
from tensorflow import einsum
from keras import Model, Sequential
from keras.layers import Layer
import keras.layers as nn
import tensorflow_addons as tfa
from einops import rearrange
from functools import partial
from inspect import isfunction

import math
import noiser
import numpy as np
import noiser
from PIL import Image
# helpers functions
MNIST_DIMS = (32, 32, 1)

def preprocess(x):
    shifted = tf.cast(x, tf.float32) / 127.5 - 1
    return tf.image.resize(tf.expand_dims(shifted, axis=-1), (32, 32))



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# We will use this to convert timestamps to time encodings


class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        return emb

# small helper modules


class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)


class Residual(Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x


def Upsample(dim):
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')


def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')


class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
        return x


class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        x = self.norm(x)
        return self.fn(x)


class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)

# building block modules


class Block(Layer):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = tfa.layers.GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)

        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta

        x = self.act(x)
        return x


class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        self.mlp = Sequential([
            SiLU(),
            nn.Dense(units=dim_out * 2)
        ]) if exists(time_emb_dim) else None

        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = nn.Conv2D(
            filters=dim_out, kernel_size=1, strides=1) if dim != dim_out else Identity()

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

        h = self.block1(x, gamma_beta=gamma_beta, training=training)
        h = self.block2(h, training=training)

        return h + self.res_conv(x)


class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.attend = nn.Softmax()
        self.to_qkv = nn.Conv2D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            LayerNorm(dim)
        ])

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)

        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)

        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out, training=training)

        return out


class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim_max = tf.stop_gradient(tf.expand_dims(
            tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, tf.float32)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        out = self.to_out(out, training=training)

        return out


class Unet(Model):
    def __init__(self,
                 dim=64,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 resnet_block_groups=8,
                 learned_variance=False,
                 sinusoidal_cond_mlp=True
                 ):
        super(Unet, self).__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2D(
            filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp

        self.time_mlp = Sequential([
            SinusoidalPosEmb(dim),
            nn.Dense(units=time_dim),
            GELU(),
            nn.Dense(units=time_dim)
        ], name="time embeddings")

        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity()
            ])

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()
            ])

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = Sequential([
            block_klass(dim * 2, dim),
            nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ], name="output")

    def call(self, x, time=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = tf.concat([x, h.pop()], axis=-1)
        x = self.final_conv(x)
        return x
    
    def set_key(self, key):
      np.random.seed(key)
    
    def forward_noise(self, key, x_0, t):
      self.set_key(key)
      noise = np.random.normal(size=x_0.shape)
      reshaped_sqrt_alpha_bar_t = np.reshape(np.take(noiser.SQRT_ALPHA_BAR, t), (-1, 1, 1, 1))
      reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(noiser.SQRT_ONE_MINUS_ALPHA_BAR, t), (-1, 1, 1, 1))
      noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
      return noisy_image, noise

    def generate_timestamp(self, key, num):
        self.set_key(key)
        return tf.random.uniform(shape=[num], minval=0, maxval=noiser.TIMESTEPS, dtype=tf.int32)
    
    def train_step(self, batch, opt):
      rng, tsrng = np.random.randint(0, 100000, size=(2,))
      timestep_values = self.generate_timestamp(tsrng, batch.shape[0])

      noised_image, noise = self.forward_noise(rng, batch, timestep_values)
      with tf.GradientTape() as tape:
          prediction = self(noised_image, timestep_values)
          
          loss_value = self.get_loss(noise, prediction)
      
      gradients = tape.gradient(loss_value, self.trainable_variables)
      opt.apply_gradients(zip(gradients, self.trainable_variables))

      return loss_value
    
    def train(self, dataset):
      batch_size = 64
      opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
      epochs = 2
      for e in range(1, epochs+1):
          # this is cool utility in Tensorflow that will create a nice looking progress bar
          bar = tf.keras.utils.Progbar(len(dataset) // batch_size)
          losses = []
          for batch_num in range (0, len(dataset), batch_size):
              batch = dataset[batch_num : batch_num + batch_size]
              # run the training loop
              loss = self.train_step(batch, opt)
              losses.append(loss)
              bar.update(batch_num // batch_size, values=[("loss", loss)])

          avg = np.mean(losses)
          print(f"Average loss for epoch {e}/{epochs}: {avg}")
    
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
        w, h, c = MNIST_DIMS
        denoised = tf.random.normal(shape=(1, w, h, c))

        for timestep in range(noiser.TIMESTEPS - 1, -1, -1):
          denoised = self.sample_timestep(denoised, timestep)
          samples.append(denoised)
      
      return samples
    
    def get_loss(self, actual, theoretical):
      return tf.reduce_mean(tf.square(actual - theoretical))
    def ddpm(self, x_t, pred_noise, t):
      alpha_t = np.take(noiser.ALPHAS, t)
      alpha_t_bar = np.take(noiser.ALPHA_BAR, t)

      eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
      mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

      var = np.take(noiser.BETAS, t)
      z = np.random.normal(size=x_t.shape)

      return mean + (var ** .5) * z
    
    # Save a GIF using logged images
    def save_gif(self, img_list, path=""):
        # Transform images from [-1,1] to [0, 255]
        imgs = []
        for im in img_list:
            im = np.array(im)
            im = (im + 1) * 127.5
            im = np.clip(im, 0, 255).astype(np.int32)
            im = Image.fromarray(im)
            imgs.append(im)
        
        imgs = iter(imgs)

        # Extract first image from iterator
        img = next(imgs)

        # Append the other images and save as GIF
        img.save(fp=path, format='GIF', append_images=imgs)


    def sample2(self):
      x = tf.random.normal((1,32,32,1))
      img_list = []
      img_list.append(np.squeeze(np.squeeze(x, 0),-1))

      for i in range(noiser.TIMESTEPS-1):
          t = np.expand_dims(np.array(noiser.TIMESTEPS-i-1, np.int32), 0)
          pred_noise = self(x, t)
          x = self.ddpm(x, pred_noise, t)
          img_list.append(np.squeeze(np.squeeze(x, 0),-1))

      self.save_gif(img_list + ([img_list[-1]] * 100), "ddpm.gif")

      return img_list