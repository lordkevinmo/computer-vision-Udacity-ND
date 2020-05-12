## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        #self.conv4 = nn.Conv2d(128, 256, 5)
        #self.conv5 = nn.Conv2d(512, 1024, 3)
        #self.conv6 = nn.Conv2d(1024, 2048, 3)
        #self.conv7 = nn.Conv2d(2048, 4096, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24*24*128, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        #self.fc4 = nn.Linear(1024, 512)
        #self.fc5 = nn.Linear(512, 136)
        #self.dropout1 = nn.Dropout(p=0.1)
        #self.dropout2 = nn.Dropout(p=0.2)
        #self.dropout3 = nn.Dropout(p=0.3)
        self.dropout = nn.Dropout(p=0.4)
        #self.dropout2 = nn.Dropout(p=0.5)
        #self.dropout6 = nn.Dropout(p=0.6)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))  # 224x224x1 => 220x220x32 => 110x110x32
        x = self.pool(F.relu(self.conv2(x)))  # 110x110x32 => 106x106x63 => 53x53x64
        x = self.pool(F.relu(self.conv3(x)))  # 53x53x64 => 49x49x128 => 24x24x128
        #x = self.pool(F.relu(self.conv4(x)))  # 24x24x128 => 20x20x128 => 10x10x256
        x = self.dropout(x)
        x = x.view(x.size(0), -1) #Flattened the output
        x = F.relu(self.fc1(x)) # Fed in fc layer with relu activation
        x = self.dropout(x) #dropout of probability 0.6
        x = F.relu(self.fc2(x)) # fed in second fc layer with relu activation
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
