
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


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
        