import keras.datasets.mnist as mnist
from medium_code import Unet, preprocess # UNet import UNet
import numpy as np

# ask user if they want to save the weights
def ask_to_save():
  saveYN = input('Do you want to save the weights? (y/n): ')
  if saveYN == 'y':
    model.save_weights('models/medium_mnist_weights')

network_code = '1'# input('Which network would you like to train?\n\t1. MNIST\n\t2. Shoes\n\t3. Cats and Dogs\n\t4. Faces\n')

# load data
if network_code == '1':
  (x_train, _), (x_test, _) = mnist.load_data()
  data = x_train # np.concatenate((x_train, x_test))
elif network_code == '2':
  data = np.load('data/shoes.npy')
elif network_code == '3':
  data = np.load('data/cats_dogs.npy')
elif network_code == '4':
  data = np.load('data/faces.npy')

# normalize to [-1, 1]
# data = 2 * (data / 255) - 1
data = preprocess(data)

# add channel dimension if necessary
if len(data.shape) != 4:
  data = np.expand_dims(data, axis=-1)

model = Unet(channels=1) #UNet(image_shape=data[0].shape, batch_size=64)
try:
  model.train(data)
except KeyboardInterrupt:
  ask_to_save()
  exit()

model.save_weights('models/medium_mnist_weights')# ask_to_save()

"""
denoised = tf.random.normal(shape=x_train[0].shape)
fig, axs = plt.subplots(10, 10)
for timestep in range(100 - 1, -1, -1):
  denoised = model.sample_timestep(denoised, timestep)[0]

  axs[timestep // 10][timestep % 10].imshow(denoised, cmap='gray')

plt.show()
"""