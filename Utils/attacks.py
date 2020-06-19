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


def fgsm(model, x, y, epsilon = 4):
    delta = torch.zeros_like(x, requires_grad=True)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(model(x + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, x, y, epsilon = 4, alpha = 1, num_iter = 100):
    delta = torch.zeros_like(x, requires_grad=True)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(num_iter):
        loss = loss_function(model(x + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def norms(z):
    return z.view(z.shape[0], -1).norm(dim=1)[:, None, None, None]


def pgd_l2(model, x, y, epsilon = 40, alpha = 10, num_iter = 100):
    delta = torch.zeros_like(x, requires_grad=True)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(num_iter):
        loss = loss_function(model(x + delta), y)
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -x), 1-x)
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
    return delta.detach()