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


def plot_grid(images, labels, predictions, m, n):
    
    figure = plt.figure(figsize=(m, n))
    for i in range(m*n):
        plt.subplot(m, n, i + 1, title=predictions[i] + ' (' + labels[i] + ')')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    return figure


def plot_adv_grid(reg_images, delta_images, adv_images, labels, reg_predictions, adv_predictions, m, n):

    figure = plt.figure(figsize=(m, 3*n))
    for i in range(m*n):
        plt.subplot(m, 3*n, 3*i + 1, title='#' + str(i+1) + ': ' + reg_predictions[i] + ' (' + labels[i] + ')')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(reg_images[i].reshape(28, 28), cmap=plt.cm.binary)

        plt.subplot(m, 3*n, 3*i + 2, title='#' + str(i+1) + ': ' + adv_predictions[i] + ' (' + labels[i] + ')')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(adv_images[i].reshape(28, 28), cmap=plt.cm.binary)

        plt.subplot(m, 3*n, 3*i + 3, title='#' + str(i+1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(delta_images[i].reshape(28, 28), cmap=plt.cm.binary)

    plt.show()
    return figure


def display_pic(pic, position):
	pixels = pic.reshape(28, 28)
	plt.subplot(position)
	sns.heatmap(data=pixels)


def log_image_grid(images, labels, predictions, m, n, writer):
    plot = plot_grid(images, labels, predictions, m, n)
    to_show = plot_to_image(plot)
    writer.add_image('images', to_show, 0)


def log_adv_image_grid(reg_images, delta_images, adv_images, labels, reg_predictions, adv_predictions, m, n, writer):
    plot = plot_adv_grid(reg_images, delta_images, adv_images, labels, reg_predictions, adv_predictions, m, n)
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