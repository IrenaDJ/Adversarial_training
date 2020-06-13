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

def train_regular(model_path, train_path, num_epochs, batch_size):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter()

	model = models.MyNetwork().to(device)
	train_x, train_y = utils.parse_data(train_path, device)

	optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.7)
	loss_function = nn.CrossEntropyLoss()

	iteration = 0
	for e in range(num_epochs):
		for i in range(0, train_x.shape[0], batch_size):
			train_x_mini = train_x[i:i + batch_size] 
			train_y_mini = train_y[i:i + batch_size] 
        
			optimizer.zero_grad()
			output = model(train_x_mini)
			loss = loss_function(output, train_y_mini)
			loss.backward()
			optimizer.step()
			predictions = torch.max(output.data, 1)[1]
			
			if i % 1000 == 0:
				writer.add_scalar('Loss/Train', loss.item(), iteration)
				writer.add_scalar('Accuracy/Train', utils.evaluate(predictions, train_y_mini))
			iteration += 1
        
		print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))

	writer.close()
	torch.save(model.state_dict(), model_path)
	summary(model, (1, 28, 28))


# python3 train_regular.py <model_path> <train_path> <num_epochs> <batch_size>
if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("\nWrong command syntax.\n")
	else:
		good = True
		try:
			model_path = sys.argv[1]
			train_path = sys.argv[2]
			num_epochs = int(sys.argv[3])
			batch_size = int(sys.argv[4])
		except ValueError:
			print("\nInvalid parameters.\n")
			good = False
		if good:
			train_regular(model_path, train_path, num_epochs, batch_size)
