import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

class set_(Dataset):
    def __init__(this, X, y):
        this.x_i = X; this.y_i = y
    def __len__(this):
        return len(this.x_i)
    def __getitem__(this, i):
        return this.x_i[i], this.y_i[i]

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
