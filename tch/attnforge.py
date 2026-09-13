import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, x, true):
        super().__init__()
        self.x = x
        self.true = true

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.true[index]


x = torch.randn(65536, 3, 2, device=device)
true = torch.randn(65536, 3, 3, device=device)

dataset = MyDataset(x, true)
dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)

print(len(dataset))
print(dataset[1])
print(x.shape)
print(true.shape)
