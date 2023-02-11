import math
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from diffusion import noise_images

NUM_SAMPLES = 3
MAX_TIME = 100
NUM_PROGRESSIONS = 25

STEP_SIZE = math.ceil(MAX_TIME / NUM_PROGRESSIONS)

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255

noised_image_progressions = []
for (i, image) in enumerate(x_train[:NUM_SAMPLES]):
  repeated_image = tf.repeat(tf.expand_dims(image, axis=0), repeats=NUM_PROGRESSIONS, axis=0)

  # noise the same image at different timesteps
  noise_progression = noise_images(repeated_image, range(0, MAX_TIME, STEP_SIZE))
  noised_image_progressions.append(noise_progression)

for sample_index, progression in enumerate(noised_image_progressions):
  square_size = math.sqrt(NUM_PROGRESSIONS)
  num_rows = math.floor(square_size)
  num_cols = math.ceil(square_size)
  fig, axs = plt.subplots(num_rows, num_cols)
  fig.suptitle(f'Sample {sample_index + 1}')
  for timestep, image in enumerate(progression):
    axs[timestep // num_cols][timestep % num_cols].imshow(image, cmap='gray')
plt.show()