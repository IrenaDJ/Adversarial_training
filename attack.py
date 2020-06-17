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


def pgd_linf(model, X, y, epsilon, alpha = 1, num_iter = 100):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #print(loss)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()



def attack(model_path, test_path, batch_size, attack_type):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    letters = string.ascii_uppercase

    model = models.MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test_x, test_y = utils.parse_data(test_path, device)

    test_delta = torch.zeros_like(test_x)
    
    for i in range(0, test_x.shape[0], batch_size):
        test_x_mini = test_x[i:i + batch_size]
        test_y_mini = test_y[i:i + batch_size]

        delta = attack_type(model, test_x_mini, test_y_mini, 5)
        output = model(test_x_mini + delta)
        predictions = torch.max(output.data, 1)[1]
        
        if i % 1000 == 0:
            writer.add_scalar('Accuracy/Train', utils.evaluate(predictions, test_y_mini))		
        test_delta[i:i+ batch_size] = delta
    
    reg_output = model(test_x)
    adv_output = model(test_x + test_delta)

    #test visualisation
    M, N = 4, 3
    perm = torch.randperm(test_x.size(0))
    ids = perm[:M*N]

    reg_images = test_x[ids].detach().cpu()
    delta_images = test_delta[ids].detach().cpu()
    adv_images = (test_x+test_delta)[ids].detach().cpu()

    labels = [letters[test_y[i].detach().cpu()] for i in ids]
    reg_predictions = [letters[torch.max(reg_output.data, 1)[1][i].item()] for i in ids]
    adv_predictions = [letters[torch.max(adv_output.data, 1)[1][i].item()] for i in ids]

    utils.log_adv_image_grid(reg_images, delta_images, adv_images, labels, reg_predictions, adv_predictions, M, N, writer)

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
            attack(model_path, test_path, 100, attack_type)