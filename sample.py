from UNet import UNet
import tensorflow as tf
from plotter import show_sample_process
from matplotlib import pyplot as plt

model = UNet(image_shape=(28, 28, 1))

# fully trained mnist model
model.load_weights('models/mnist_weights')

# give the model random noise and see if it can produce the "original image"
partials = []
noised_image = tf.random.normal(shape=(1, 28, 28, 1))

print('restoring image...')
for i in range(99, 0, -1):
  noised_image -= model(noised_image, tf.constant([i]))
  partials.append(noised_image)

partials = tf.squeeze(partials)
print('done!')

# display partial restorations
print('showing results...')
show_sample_process(partials)
# out = model.sample((1, 28, 28, 1))
# plt.imshow(tf.squeeze(out), cmap='gray')
plt.show()