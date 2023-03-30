import matplotlib.pyplot as plt
import numpy as np

# Don't initialize figures in case the user doesn't want to see them.
# If it's initialized, there will be a figure that pops up whether the user requests it or not.
sample_fig = None
loss_fig = None
sample_fig_id = 1
loss_fig_id = 2
sample_process_fig_id = 3

loss_xs = []
loss_ys = []

def activate_plots():
  plt.ion()

def update_losses(loss):
  global loss_fig
  # initialize loss figure/axis on first call
  if loss_fig is None:
    loss_fig = plt.figure(loss_fig_id)
    loss_fig.suptitle('Losses')

  # switch to sample figure
  plt.figure(loss_fig_id)

  # update running lists
  loss_xs.append(len(loss_xs))
  loss_ys.append(loss)

  # clear figure
  plt.clf()
  
  # plot losses
  plt.plot(loss_xs, loss_ys, label='raw losses')

  # plot line of best fit
  if len(loss_xs) > 1:
    b, a = np.polyfit(loss_xs, loss_ys, deg=1)
    plt.plot(loss_xs, a + b * np.array(loss_xs), label='trendline')

def update_samples(batch_num, epoch_num, loss, dataset_image, noised_at_t, noised_at_t_minus_1, unet_noise, timestep):
  global sample_fig
  # initialize sample figure/axes on first call
  if sample_fig is None:
    sample_fig = plt.figure(sample_fig_id)

  # switch to sample figure
  plt.figure(sample_fig_id)

  # title/configuration
  sample_fig.suptitle(f'batch: {batch_num}, epoch: {epoch_num} (avg. loss = {loss})')
  sample_fig.tight_layout(pad=0.1)

  # show image from data set
  plt.subplot(3, 1, 1)
  plt.imshow(dataset_image, cmap='gray')
  plt.title('image from dataset')

  # show noised image at t
  plt.subplot(3, 1, 2)
  plt.imshow(noised_at_t, cmap='gray')
  plt.title(f'image at timestep {timestep}')

  # show noised image at t-1
  plt.subplot(3, 3, 7)
  plt.imshow(noised_at_t_minus_1, cmap='gray')
  plt.title(f'image at timestep {timestep - 1}')

  # show unet generated noise
  plt.subplot(3, 3, 8)
  plt.imshow(unet_noise, cmap='gray')
  plt.title('noise gen by unet')

  # show unet generated noise
  plt.subplot(3, 3, 9)
  plt.imshow(unet_noise + noised_at_t_minus_1, cmap='gray')
  plt.title(f'unet prediction at timestep {timestep}')

def show_sample_process(img_noising_sequence):
  plt.figure(sample_process_fig_id)
  for i in range(10):
    for j in range(10):
      if i == 9 and j == 9:
        continue
      plt.subplot(10, 10, i * 10 + j + 1)
      plt.imshow(img_noising_sequence[i * 10 + j], cmap='gray')

  plt.show()

def draw_plots():
  plt.pause(0.05)
  plt.show(block=False)