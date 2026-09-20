import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 канал → 32 фильтра
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # уменьшаем в 2 раза
        self.fc = nn.Linear(64 * 7 * 7, 10)  # после пулинга: 28→14→7

    def forward(self, x):
        # x размером (batch, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))  # → (batch, 32, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))  # → (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # разворачиваем → (batch, 64*7*7)
        x = self.fc(x)  # → (batch, 10)
        return x