# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random


path = pathlib.Path(__file__).parent

#Load Datasets
data_images = np.load(f"{path}/train_images.npy")
data_labels = np.load(f"{path}/train_labels.npy")

#Preview data images
f, axarr = plt.subplots(9, 9)
for i in range(9):
  for j in range(9):
    n = random.randint(0, data_images.shape[0])
    axarr[i,j].imshow(data_images[n], cmap="gray")
    axarr[i,j].set_xticks([])
    axarr[i,j].set_yticks([])
plt.savefig(path / "preview.jpg")
plt.show()


#Split dataset to train_data and test_data
train_x, test_x, train_y, test_y = train_test_split(data_images,
                                                    data_labels,
                                                    shuffle=True,
                                                    test_size=0.2)


#Convert numpy arrays to tensor
train_x = np.expand_dims(train_x, 1)
test_x = np.expand_dims(test_x, 1)

train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = torch.from_numpy(train_y).to(torch.long)

test_x = torch.from_numpy(test_x).to(torch.float32)
test_y = torch.from_numpy(test_y).to(torch.long)

#Convolutional neural network

class Activated2D(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding): 
    super(Activated2D,self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, input):
    output = self.conv(input)
    output = self.bn(output)
    output = self.relu(output)
    return output

class network(torch.nn.Module):
  def __init__(self):
    super(network, self).__init__()

    self.bn_input = nn.BatchNorm2d(1)
      
    #Create conv layers with max pooling in between
    self.subnetwork1 = Activated2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    
    self.subnetwork2 = Activated2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.subnetwork3 = Activated2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.subnetwork4 = Activated2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.subnetwork5 = Activated2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    #Add all the conv subnetworks into the Sequential layer in exact order  
    self.net = nn.Sequential(self.subnetwork1, self.pool1,
                             self.subnetwork2, self.pool2,
                             self.subnetwork3, self.pool3,
                             self.subnetwork4, self.pool4,
                             self.subnetwork5, self.pool5
                            )

    #Create dense layers
    self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)
    self.drop = nn.Dropout2d(0.25)
    self.fc2 = nn.Linear(in_features=1024, out_features=100)
    self.fc3 = nn.Linear(in_features=100, out_features=10)

    #Add all the dense layers into the Sequential layer in exact order  
    self.lin = nn.Sequential(self.fc1, self.drop, self.fc2, self.fc3)

        
  def forward(self, input):
      output = self.net(input)
      output = output.view(output.size(0), -1)
      output = self.lin(output)
      return output
model = network()

#Train part
net = NeuralNetClassifier(
    module=model,
    max_epochs=10,
    lr=0.001,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    device = "cuda:0"
    )

net.initialize()
net.fit(train_x, train_y)

#Plot model residuals
history = net.history
losses = history[:, ('train_loss', 'valid_loss')]

train_losses = [loss[0] for loss in losses]
valid_losses = [loss[1] for loss in losses]

plt.plot(range(len(losses)), train_losses)
plt.plot(range(len(losses)), valid_losses)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

# Get network score on training dataset and testing datset
print(f"Training Accuracy : {100 * net.score(train_x, train_y):.2f}%")
print(f"Testing Accuracy : {100 * net.score(test_x, test_y):.2f}%")



