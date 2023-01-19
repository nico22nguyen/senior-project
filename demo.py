import math
import keras.datasets.cifar100 as cifar100
import matplotlib.pyplot as plt
from diffusion import noise_image

NUM_SAMPLES = 10
MAX_TIME = 24
NUM_PROGRESSIONS = 12

STEP_SIZE = math.ceil(MAX_TIME / NUM_PROGRESSIONS)

(x_train, _), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255

noised_image_progressions = []
for (i, image) in enumerate(x_train[:NUM_SAMPLES]):
  noise_progression = [noise_image(image, timestep) for timestep in range(0, MAX_TIME, STEP_SIZE)]
  noised_image_progressions.append(noise_progression)

for sample_index, progression in enumerate(noised_image_progressions):
  square_size = math.sqrt(NUM_PROGRESSIONS)
  num_rows = math.floor(square_size)
  num_cols = math.ceil(square_size)
  fig, axs = plt.subplots(num_rows, num_cols)
  fig.suptitle(f'Sample {sample_index + 1}')
  for timestep, image in enumerate(progression):
    axs[timestep // num_cols][timestep % num_cols].imshow(image)
plt.show()