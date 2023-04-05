import tensorflow as tf
from matplotlib import pyplot as plt
import noiser
from Diffusion_test import Unet, timesteps, ddpm
import numpy as np
from tqdm import tqdm

print('instantiating model...')
unet = Unet(channels=1)# UNet(image_shape=(28, 28, 1))

# fully trained mnist model
print('loading weights...')
unet.load_weights('models/current_weights')
#model = keras.models.load_model('models/medium_mnist_weights')

"""
partials = []
noised_image = tf.random.normal(shape=(1, 28, 28, 1))
print('restoring image...')
for i in range(99, 0, -1):
  noised_image -= model(noised_image, tf.constant([i]))
  partials.append(noised_image)

partials = tf.squeeze(partials)
print('done!')

progressions, final = model.sample()
"""


print('generating samples...')
x = tf.random.normal((1,32,32,1))
img_list = []
img_list.append(np.squeeze(np.squeeze(x, 0),-1))

for i in tqdm(range(timesteps-1)):
    t = np.expand_dims(np.array(timesteps-i-1, np.int32), 0)
    pred_noise = unet(x, t)
    x = ddpm(x, pred_noise, t)
    img_list.append(np.squeeze(np.squeeze(x, 0),-1))


final = np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint8)
plt.show()
print('showing results...')
plt.ion()
for (i, sample) in enumerate(img_list):
  plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
  plt.imshow(tf.squeeze(sample), cmap='gray')
  plt.show()
  plt.pause(0.01)
  plt.close()
plt.ioff()

print('final image:')
plt.figure()
plt.suptitle('Final Image')
plt.imshow(tf.squeeze(final), cmap='gray')
plt.show()