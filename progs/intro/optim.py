from dataset01 import set_
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn as nn

X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
print(X.data)
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + .5
y = y + .1 * torch.randn_like(y)

dataset = set_(X, y)
