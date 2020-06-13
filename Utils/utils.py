import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import seaborn as sns
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter



def evaluate(predictions, labels):              
	correct = 0
	for p, l in zip(predictions, labels):
		if p == l:
			correct += 1
       
	accuracy = correct / len(predictions)
	return accuracy



def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a tensor"""

  data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
  w, h = figure.canvas.get_width_height()
  data = data.reshape(h, w, 3)

  trans = transforms.ToPILImage()
  trans_tensor = transforms.ToTensor()
  return trans_tensor(trans(data))

def plot_grid(images, labels, predictions, M, N):
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(M,N))
  for i in range(M*N):
    # Start next subplot.
    plt.subplot(M, N, i + 1, title=predictions[i] + ' (' + labels[i] + ')')
    #plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
  plt.show()
  return figure

def display_pic(pic, position):
	pixels = pic.reshape(28, 28)
	plt.subplot(position)
	sns.heatmap(data=pixels)

def log_image_grid(images, labels, predictions, M, N, writer):
  plot = plot_grid(images, labels, predictions, M, N)
  to_show = plot_to_image(plot)
  writer.add_image('images', to_show, 0)



def parse_data(path, device):
	raw_data = pd.read_csv(path, sep=",")

	labels = raw_data['label']
	raw_data.drop('label', axis=1, inplace=True)

	data = raw_data.values
	labels = labels.values

	data = data.reshape(data.shape[0], 1, 28, 28)

	x = torch.cuda.FloatTensor(data).to(device)
	y = torch.cuda.LongTensor(labels).to(device)

	return x, y