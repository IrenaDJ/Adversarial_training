import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
import Utils.attacks as attacks


def train_adversarial(model_path, train_path, attack_type, num_epochs, batch_size):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter()

	model = models.MyNetwork().to(device)
	train_x, train_y = utils.parse_data(train_path)

	optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.7)
	loss_function = nn.CrossEntropyLoss()

	iteration = 0
	for e in range(num_epochs):
		for i in range(0, train_x.shape[0], batch_size):
			train_x_mini = train_x[i:i + batch_size].to(device)
			train_y_mini = train_y[i:i + batch_size].to(device)
        
			delta_mini = attack_type(model, train_x_mini, train_y_mini)
			output = model(train_x_mini + delta_mini)

			optimizer.zero_grad()
			loss = loss_function(output, train_y_mini)
			loss.backward()
			optimizer.step()

			predictions = torch.max(output.data, 1)[1]
			if i % 1000 == 0:
				writer.add_scalar('Loss/Train Adversarial', loss.item(), iteration)
				writer.add_scalar('Accuracy/Train Adversarial', utils.evaluate(predictions, train_y_mini))
			iteration += 1
        
		print('Epoch: {}, Loss: {:.6f}'.format(e + 1, loss.item()))

	writer.close()
	torch.save(model.state_dict(), model_path)
	summary(model, (1, 28, 28))


# python3 train_adversarial.py <model_path> <train_path> <attack_type> <num_epochs> <batch_size>
if __name__ == "__main__":
	if len(sys.argv) != 6:
		print("\nWrong command syntax.\n")
	else:
		good = True
		try:
			model_path = sys.argv[1]
			train_path = sys.argv[2]
			num_epochs = int(sys.argv[4])
			batch_size = int(sys.argv[5])
		except ValueError:
			print("\nInvalid parameters.\n")
			good = False

		if good:
			attack_type = None
			if sys.argv[3] == "pgd_linf":
				attack_type = attacks.pgd_linf
			elif sys.argv[3] == "pgd_l2":
				attack_type = attacks.pgd_l2
			elif sys.argv[3] == "fgsm":
				attack_type = attacks.fgsm
			
			if attack_type == None:
				print("\nInvalid attack type specified.\n")
			else:
				train_adversarial(model_path, train_path, attack_type, num_epochs, batch_size)
