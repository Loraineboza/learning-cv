import torch
from model import dataset
from torch.utils.data import DataLoader
import tqdm

X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
print(X.data)
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + 0.5
y = y + 0.1 * torch.randn_like(y)

device = ("cuda" if torch.cuda.is_available() else "cpu")
dataset_ = dataset(X, y)
iter_loader = DataLoader(dataset_, batch_size=10, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=False)

e = 1
for i in range(e):
    iter_tqdm = tqdm.tqdm(iter_loader, leave=True)
    for  inputs, labels in iter_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        iter_tqdm.set_description(f"-e [{i+1}/{e}]")
