import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms.v2 as tfs

import os
import json
from PIL import Image
from tqdm import tqdm

class SunDataset(data.Dataset):
    def __init__(this, path, train=True, transform=None):
        this.path = os.path.join(path, "train" if train else "test")
        this.transform = transform

        with open(os.path.join(this.path, "format.json"), "r") as fd: 
            this.format = json.load(fd)

        this.length = len(this.format)
        this.files = tuple(this.format.keys())
        this.targets = tuple(this.format.values())

    def __getitem__(this, index):
        path_file = os.path.join(this.path, this.files[index])
        img = Image.open(path_file).convert('RGB')

        if this.transform:
            img = this.transform(img)

        return img, torch.tensor(this.targets[index], dtype=torch.float32)

    def __len__(this):
        return this.length

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train = SunDataset("dataset_reg", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model = nn.Sequential(
    #1. свертка + активация + пулинг
    nn.Conv2d(3, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), # (batch, 32, 128, 128)
    
    #2. свертка + активация + пулинг
    nn.Conv2d(32, 8, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), # (batch, 8, 64, 64)

    #3. свертка + активация + пулинг
    nn.Conv2d(8, 4, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),  # (batch, 4, 32, 32)

    #4. разворот и полносвязные слои
    nn.Flatten(), # (batch, 4096)
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)