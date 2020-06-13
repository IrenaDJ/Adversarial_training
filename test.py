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

def test(model_path, test_path):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter()

	model = models.MyNetwork()
	model.load_state_dict(torch.load(model_path))
	model = model.to(device)
	
	test_x, test_y = utils.parse_data(test_path, device)

	letters = string.ascii_uppercase
	predictions = model(test_x)
	model.eval()
	accuracy = utils.evaluate(torch.max(predictions.data, 1)[1], test_y)
	print("Accuracy: {}".format(accuracy))

	#test visualisation
	M, N = 2, 6
	#images = test_x[0:M*N].detach().cpu()
	
	perm = torch.randperm(test_x.size(0))
	ids = perm[:M*N]
	
	images = test_x[ids].detach().cpu()

	labels = [letters[test_y[i].detach().cpu()] for i in ids]
	predictions = [letters[torch.max(predictions.data, 1)[1][i].item()] for i in ids]

	utils.log_image_grid(images, labels, predictions, M, N, writer)

	writer.close()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("\nWrong command syntax.\n")
	else:
		model_path = sys.argv[1]
		test_path = sys.argv[2]
		test(model_path, test_path)