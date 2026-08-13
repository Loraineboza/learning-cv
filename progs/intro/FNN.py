'''
Полносвязная сеть (FNN) на синтетических данных с их шумом
'''

from dataset01 import set_
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + .5
y = y + .1 * torch.randn_like(y)
mse = nn.MSELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

dataset = set_(X, y)
iter_load = DataLoader(dataset, shuffle=True, batch_size=10)

e=50
model.train()
for i in range(e):
    loss_mean, mean_count = (0, 0)
    for signal, label in iter_load:
        optim.zero_grad()
        loss = mse(model(signal), label)
        loss.backward()
        optim.step()
        mean_count += 1
        loss_mean = 1/mean_count * loss.item() + (1 - 1/mean_count) * loss_mean
    print(f"iter=[{i+1}]: loss_mean: {loss_mean:.5f}")