import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import torch.optim as optim

class CNNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class dataset_loader():
    def __init__(self, batch=128, num_workers=2):
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5, ))])
        self.batch = batch
        self.num_workers = num_workers

    def LoadMnist(self):
        trainset = MNIST(root='./mnist',
                        train=True,
                        download=True,
                        transform=self.transform)
        
        testset = MNIST(root='./mnist',
                        train=False,
                        download=True,
                        transform=self.transform)
        
        trainloader = DataLoader(dataset=trainset,
                                batch_size=self.batch,
                                shuffle=True,
                                num_workers=self.num_workers)
        testloader = DataLoader(dataset=testset,
                                batch_size=self.batch,
                                shuffle=False,
                                num_workers=self.num_workers)
        
        return trainloader, testloader


    def LoadFashionMnist(self):
        trainset = FashionMNIST(root='./fashion-mnist',
                                download=True,
                                train=True,
                                transform=self.transform)
        
        testset = FashionMNIST(root='./fashion-mnist',
                               download=True,
                               train=False,
                               transform=self.transform)
        
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=self.batch,
                                 shuffle=True,
                                 num_workers=self.num_workers)

        testloader = DataLoader(dataset=testset,
                                batch_size=self.batch,
                                shuffle=True,
                                num_workers=self.num_workers)
        
        return trainloader, testloader

def train_function():
    dataload = dataset_loader()
    train, test = dataload.LoadFashionMnist()
    net = Net()

def main():
    train_function()

    print(train)

if __name__ == "__main__":
    main()