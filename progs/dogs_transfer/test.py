import os
import json

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class DogDataset(data.Dataset):
    def __init__(this, path, train=True, transform=None):
        this.path = os.path.join(path, "train" if train else "test")
        this.transform = transform

        with open(os.path.join(this.path, "format.json"), "r") as fp:
            this.format = json.load(fp)

        this.length = 0
        this.files = []
        this.targets = torch.eye(10)

        for _dir, _target in this.format.items():
            path = os.path.join(this.path, _dir)
            list_files = os.listdir(path)
            this.length += len(list_files)
            this.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(this, item):
        path_file, target = this.files[item]
        t = this.targets[target]
        img = Image.open(path_file)

        if this.transform:
            img = this.transform(img)
        return img, t

    def __len__(this):
        return this.length

resnet_weights = models.ResNet50_Weights.DEFAULT
transforms = resnet_weights.transforms()

model = models.resnet50(weights=resnet_weights)
model.requires_grad_(False)

model.fc = nn.Linear(512 * 4, 10)
model.fc.requires_grad_(True)

d_train = DogDataset(r"dataset_dogs", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# learning 
optimizer = optim.Adam(params=model.fc.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 10
model.train()

print(model)
exit(0)
for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")


st = model.state_dict()
torch.save(st, 'model_transfer_resnet.tar')

# test
d_test = DogDataset(r"dataset_dogs", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=32, shuffle=False)

Q = 0
P = 0
count = 0
model.eval()


test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        p2 = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        P += torch.sum(p2 == y).item()
        Q += loss_function(p, y_test).item()
        count += 1   

Q /= count
P /= len(d_test)
print(Q)
print(P)