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


class MyNetwork(nn.Module): 


	def __init__(self):
		super(MyNetwork, self).__init__()
		
		self.conv1 = nn.Conv2d(1, 10, 3)
		self.pool1 = nn.MaxPool2d(2)
        
		self.conv2 = nn.Conv2d(10, 20, 3)
		self.pool2 = nn.MaxPool2d(2)
        
		self.conv3 = nn.Conv2d(20, 30, 3) 
		self.dropout = nn.Dropout2d()
        
		self.fc1 = nn.Linear(30 * 3 * 3, 270) 
		self.fc2 = nn.Linear(270, 26)
        
		self.softmax = nn.LogSoftmax(dim=1)
    

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.pool1(x)
        
		x = self.conv2(x)
		x = F.relu(x)
		x = self.pool2(x)
        
		x = self.conv3(x)
		x = F.relu(x)
		x = self.dropout(x)
                
		x = x.view(-1, 30 * 3 * 3)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.softmax(x)

		return x
        


def evaluate(predictions, labels):
               
	correct = 0
	for p, l in zip(predictions, labels):
		if p == l:
			correct += 1
       
	accuracy = correct / len(predictions)
	return accuracy


def display_pic(pic, position):
	pixels = pic.reshape(28, 28)
	plt.subplot(position)
	sns.heatmap(data=pixels)

def log_image_grid(images, writer):
	grid = torchvision.utils.make_grid(images)
	writer.add_image('images', grid, 0)


def log_image(image, prediction, writer):
	#display_image = image.transpose(2, 0, 1)
	#print(image)
	writer.add_image(prediction, image, 0)
	

def image_grid(images, labels):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(25):
    # Start next subplot.
    plt.subplot(5, 5, i + 1, title=labels[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)


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


def train_model(model_path, train_path, num_epochs, batch_size, device, writer):

	model = MyNetwork().to(device)
	train_x, train_y = parse_data(train_path, device)

	optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.7)
	loss_function = nn.CrossEntropyLoss()

	iteration = 0
	for e in range(num_epochs):
		for i in range(0, train_x.shape[0], batch_size):
			train_x_mini = train_x[i:i + batch_size] 
			train_y_mini = train_y[i:i + batch_size] 
        
			optimizer.zero_grad()
			output = model(Variable(train_x_mini))
			loss = loss_function(output, Variable(train_y_mini))
			loss.backward()
			optimizer.step()
        
			if i % 1000 == 0:
				writer.add_scalar('Loss/Train', loss.item(), iteration)
			iteration += 1
        
		print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))

	torch.save(model.state_dict(), model_path)
	return model


def main(pretrained, model_path, test_path, train_path, num_epochs, batch_size):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter()

	if not pretrained:
		model = train_model(model_path, train_path, num_epochs, batch_size, device, writer)
	else:
		model = MyNetwork()
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)

	summary(model, (1, 28, 28))
	test_x, test_y = parse_data(test_path, device)

	#log_image_grid(test_x, writer)
	letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
	predictions = model(Variable(test_x))
	model.eval()
	accuracy = evaluate(torch.max(predictions.data, 1)[1], test_y)
	print("Accuracy: {}".format(accuracy))


	print(test_x[0].size())
	image = test_x[0].detach().cpu()
	prediction = torch.max(predictions.data, 1)[1][0]
	print(prediction)
	prediction = letters[prediction.item()]
	log_image(image, prediction, writer)
	writer.close()

	
# python3 small_cnn.py 0 <model_path> <test_path> <train_path> <num_epochs> <batch_size>
# python3 small_cnn.py 1 <model_path> <test_path>
if __name__ == "__main__":
	good = True
	try:
		pretrained = bool(int(sys.argv[1]))
		model_path = sys.argv[2]
		test_path = sys.argv[3]
		if not pretrained:
			train_path = sys.argv[4]
			num_epochs = int(sys.argv[5])
			batch_size = int(sys.argv[6])
		else:
			train_path = None
			num_epochs = None
			batch_size = None
	except ValueError:
		print("\nInvalid parameters.\n")
		good = False
	if good:
		main(pretrained, model_path, test_path, train_path, num_epochs, batch_size)
