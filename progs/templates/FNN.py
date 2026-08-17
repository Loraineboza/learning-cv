'''
Полносвязная сеть (FNN) на синтетических данных с их шумом
'''

from model import dataset, EPOCHS
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from tqdm import tqdm
device = ("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
).to(device=device)
X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + .5
y = y + .1 * torch.randn_like(y)
mse = nn.MSELoss().to(device=device)
optim = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True, foreach=True)
dataset_ = dataset(X, y)
iter_load = DataLoader(dataset_, shuffle=True, batch_size=10, prefetch_factor=2, pin_memory=False, num_workers=4)

model.train()
for i in range(EPOCHS):
    loss_mean, mean_count = (0, 0)
    for inputs, labels in iter_load:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        loss = mse(model(inputs), labels)
        loss.backward()
        optim.step()
        mean_count += 1
        loss_mean = 1/mean_count * loss.item() + (1 - 1/mean_count) * loss_mean
    print(f"iter=[{i+1}]: loss_mean: {loss_mean:.5f}")