from UNet import UNet
from matplotlib import pyplot as plt
import plotter
import numpy as np
import noiser

SAVE_FILE = './models/mnist_50.pkl'
model = UNet(channels=1, image_shape=(32, 32, 1), dim_multipliers=(1, 2, 4, 8, 16))
model.load_weights(SAVE_FILE)

plt.ion()
while True:
  samples = model.sample()
  for i, sample in enumerate(samples):
    plt.suptitle(f'Timestep {noiser.TIMESTEPS - i}')
    plotter.imshow_rgb_safe(np.squeeze(sample))
    plt.show()
    plt.pause(0.01)
    plt.clf()
    
  final = np.array(np.clip((samples[-1][0] + 1) * 127.5, 0, 255), np.uint8)
  plt.suptitle('Final Image')
  plotter.imshow_rgb_safe(final)
  plt.show()
  plt.pause(0.1)
  plt.clf()