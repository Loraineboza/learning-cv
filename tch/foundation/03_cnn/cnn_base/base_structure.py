import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__() #если height = 14, width = 14, то =>
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)  # out.shape = (10, 32, 14, 14)
        self.conv2 = nn.Conv2d(10, 64, kernel_size=3, padding=1) # out.shape = (10, 64, 14, 14)
        self.pool = nn.MaxPool2d(2, 2)  # H/2, W/2 => out.shape = (10, 64, 7, 7)
        
        self.flatten = nn.Flatten(1, -1) # out.shape = (10, 64 * 7 * 7)
        self.fc = nn.Linear(64 * 7 * 7, 10)  # out.shape = (10, 10)
        
    def forward(self, x):
        # x размером (batch, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))  # => (batch, 32, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))  # => (batch, 64, 7, 7)
        x = self.flatten(x) # => (batch, 64 * 7 * 7)
        x = self.fc(x)  # → (batch, 10)
        return x
