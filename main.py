import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
from UNet import UNet
import numpy as np

network_code = input('Which network would you like to train?\n\t1. MNIST\n\t2. Shoes\n\t3. Cats and Dogs\n\t4. Faces\n')

# load data
if network_code == '1':
  (x_train, _), (x_test, _) = mnist.load_data()
  data = np.concatenate((x_train, x_test))
elif network_code == '2':
  data = np.load('data/shoes.npy')
elif network_code == '3':
  data = np.load('data/cats_dogs.npy')
elif network_code == '4':
  data = np.load('data/faces.npy')

# normalize to [0, 1]
data = data / 255

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

model = UNet(image_shape=data[0].shape)
model.train(data, epochs=5, batch_size=32, show_samples=True, show_losses=True)

"""
denoised = tf.random.normal(shape=x_train[0].shape)
fig, axs = plt.subplots(10, 10)
for timestep in range(100 - 1, -1, -1):
  denoised = model.sample_timestep(denoised, timestep)[0]

  axs[timestep // 10][timestep % 10].imshow(denoised, cmap='gray')

plt.show()
"""