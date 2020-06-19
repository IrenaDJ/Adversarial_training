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


def test(model_path, test_path, display_rows, display_columns):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter()
	letters = string.ascii_uppercase

	model = models.MyNetwork()
	model.load_state_dict(torch.load(model_path))
	model = model.to(device)
	model.eval()

	test_x, test_y = utils.parse_data(test_path, device)
	output = model(test_x)
	predictions = torch.max(output.data, 1)[1]

	accuracy = utils.evaluate(predictions, test_y)
	print("Accuracy: {}".format(accuracy))

	perm = torch.randperm(test_x.size(0))
	ids = perm[:display_rows*display_columns]

	images = test_x[ids].detach().cpu()
	labels = [letters[test_y[i].detach().cpu()] for i in ids]
	predictions = [letters[predictions[i].item()] for i in ids]

	utils.log_image_grid(images, labels, predictions, display_rows, display_columns, writer)
	writer.close()


# python3 test.py <model_path> <test_path> <display_rows> <display_columns>
if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("\nWrong command syntax.\n")
	else:
		good = True
		try:
			model_path = sys.argv[1]
			test_path = sys.argv[2]
			display_rows = int(sys.argv[3])
			display_columns = int(sys.argv[4])
		except ValueError:
			print("\nInvalid parameters.\n")
			good = False
		if good:
			test(model_path, test_path, display_rows, display_columns)