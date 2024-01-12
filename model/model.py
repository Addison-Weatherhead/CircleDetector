import torch
import torch.nn as nn
import torchvision

class CircleDetector(nn.Module):
    def __init__(self, width: int):
        super(CircleDetector, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=0)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0)
        self.relu3 = nn.ReLU()

        conv_size = (width - 4) // 2 - 2
        self.linear1 = nn.Linear(conv_size * conv_size * 64, 3)

    def forward(self, x: torch.Tensor):
        # Apply convolutional layers and activations
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # Apply pooling
        x = self.pool1(x)

        # Apply third convolutional layer and activation
        x = self.conv3(x)
        x = self.relu3(x)

        # reshape to (bs, -1)
        x = x.reshape(x.size(0), -1) 

        # output is (bs, 3)
        x = self.linear1(x)

        return x
