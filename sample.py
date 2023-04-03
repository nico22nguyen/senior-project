from medium_code import Unet #UNet import UNet
import tensorflow as tf
from plotter import show_sample_process
from matplotlib import pyplot as plt
import noiser

print('instantiating model...')
model = Unet(channels=1)# UNet(image_shape=(28, 28, 1))

# fully trained mnist model
print('loading weights...')
model.load_weights('models/medium_mnist_weights')

"""
partials = []
noised_image = tf.random.normal(shape=(1, 28, 28, 1))
print('restoring image...')
for i in range(99, 0, -1):
  noised_image -= model(noised_image, tf.constant([i]))
  partials.append(noised_image)

partials = tf.squeeze(partials)
print('done!')
"""

print('generating samples...')
progressions, final = model.sample()

print('showing results...')
plt.ion()
for (i, sample) in enumerate(progressions):
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