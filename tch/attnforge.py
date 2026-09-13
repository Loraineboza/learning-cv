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

class ConvStem(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.drop(F.gelu(self.bn1(self.conv1(x))))
        x = self.drop(F.gelu(self.bn2(self.conv2(x))))
        return x.transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))

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
