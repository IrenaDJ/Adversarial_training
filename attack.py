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

import Utils.utils as utils
import Utils.models as models

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, X, y, epsilon, alpha = 1e-2, num_iter = 40):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = -nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()



def attack(model_path, test_path, attack_type):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    model = models.MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    test_x, test_y = utils.parse_data(test_path, device)

    delta = attack_type(model, test_x, test_y, 0.1)
    predictions = model(test_x + delta)

    #test visualisation
    M, N = 2, 6
    #images = test_x[0:M*N].detach().cpu()

    letters = string.ascii_uppercase

    perm = torch.randperm(test_x.size(0))
    ids = perm[:M*N]

    images = (test_x+delta)[ids].detach().cpu()

    labels = [letters[test_y[i].detach().cpu()] for i in ids]
    predictions = [letters[torch.max(predictions.data, 1)[1][i].item()] for i in ids]

    utils.log_image_grid(images, labels, predictions, M, N, writer)

    writer.close()



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\nWrong command syntax.\n")
    else:
        model_path = sys.argv[1]
        test_path = sys.argv[2]
        
        attack_type = None
        if sys.argv[3] == "pgd_linf":
            attack_type = pgd_linf
        elif sys.argv[3] == "fgsm":
            attack_type = fgsm
        
        if attack_type == None:
            print("\nInvalid Attack Type selected.\n")
        else:
            attack(model_path, test_path, attack_type)