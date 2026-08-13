import torch
from dataset01 import set_
from torch.utils.data import DataLoader
import tqdm

X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + .5
y = y + .1 * torch.randn_like(y)


dataset = set_(X, y)
iter_loader = DataLoader(dataset, batch_size=10, shuffle=True)

e = 1

for i in range(e):
    iter_tqdm = tqdm.tqdm(iter_loader, leave=True)
    for  signal, label in iter_tqdm:
        iter_tqdm.set_description(f"-e [{i+1}/{e}]")
