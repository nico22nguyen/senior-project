import matplotlib.pyplot as plt
import numpy as np

# Don't initialize figures in case the user doesn't want to see them.
# If it's initialized, the figures will stull pop up, they just won't update
sample_fig, sample_axs = (None, None)
loss_fig, loss_axs = (None, None)
plt.ion()

loss_xs = []
loss_ys = []

def update_losses(loss):
  global loss_fig, loss_axs
  # initialize loss figure/axis on first call
  if loss_fig is None:
    loss_fig, loss_axs = plt.subplots(1, 1)
    loss_fig.suptitle('Losses')

  # update running lists
  loss_xs.append(len(loss_xs))
  loss_ys.append(loss)

  # clear figure
  loss_axs.clear()
  
  # plot losses
  loss_axs.plot(loss_xs, loss_ys, label='raw losses')

  # plot line of best fit
  if len(loss_xs) > 1:
    b, a = np.polyfit(loss_xs, loss_ys, deg=1)
    loss_axs.plot(loss_xs, a + b * np.array(loss_xs), label='trendline')

def update_samples(batch_num, loss, starting_image, actual_noise, predicted_noise, timestep):
  global sample_fig, sample_axs
  # initialize sample figure/axes on first call
  if sample_fig is None:
    sample_fig, sample_axs = plt.subplots(3, 1)

  # title/configuration
  sample_fig.suptitle(f'batch: {batch_num} (avg. loss = {loss})')
  sample_fig.tight_layout()

  # show image from data set
  sample_axs[0].imshow(starting_image, cmap='gray')
  sample_axs[0].set_title('image from dataset')

  # show noised image from noiser function
  sample_axs[1].imshow(actual_noise, cmap='gray')
  sample_axs[1].set_title(f'noised image at timestep {timestep} according to noiser function')

  # show model output
  sample_axs[2].imshow(predicted_noise, cmap='gray')
  sample_axs[2].set_title(f'noised image at timestep {timestep} according to unet')

def draw_plots():
  plt.pause(0.05)
  plt.show(block=False)