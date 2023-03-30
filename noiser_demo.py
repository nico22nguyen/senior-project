import math
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from noiser import noise_images

NUM_SLIDES = 1
MAX_TIME = 100
NUM_PROGRESSIONS = 100

square_size = math.sqrt(NUM_PROGRESSIONS)
NUM_ROWS = math.floor(square_size)
NUM_COLS = math.ceil(square_size)

STEP_SIZE = math.ceil(MAX_TIME / NUM_PROGRESSIONS)

(data, _), (x_test, y_test) = mnist.load_data()
# only select the amount of samples as we need slides
data = data[:NUM_SLIDES]
# normalize to [-1, 1]
data = 2 * (data / 255) - 1

slides = []
for image in data:
  # add batch dimension
  batched_image = tf.expand_dims(image, axis=0)

  # repeat image as many times as we need progression steps
  repeated_image = tf.repeat(batched_image, repeats=NUM_PROGRESSIONS, axis=0)

  # add channel dimension if necessary
  if len(repeated_image.shape) != 4:
    repeated_image = tf.expand_dims(repeated_image, axis=-1)

  # noise the same image at different timesteps
  noise_progression = noise_images(repeated_image, range(0, MAX_TIME, STEP_SIZE))[0]
  slides.append(noise_progression)

# draw slides
for sample_index, progression in enumerate(slides):
  fig, axs = plt.subplots(NUM_ROWS, NUM_COLS)
  fig.suptitle(f'Sample {sample_index + 1}')
  for timestep, image in enumerate(progression):
    axs[timestep // NUM_COLS][timestep % NUM_COLS].imshow(image, cmap='gray')

plt.show()