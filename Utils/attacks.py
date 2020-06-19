import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import seaborn as sns
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(x, requires_grad=True)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(model(x + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, x, y, epsilon, alpha = 1, num_iter = 100):
    delta = torch.zeros_like(x, requires_grad=True)
    loss_function = nn.CrossEntropyLoss()
    for t in range(num_iter):
        loss = loss_function(model(x + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()