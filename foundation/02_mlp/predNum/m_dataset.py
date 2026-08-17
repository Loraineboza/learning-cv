import os
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim

class DigitDataset(data.Dataset):
    def __init__(this, path, train=True, transform=None):
        this.path = os.path.join(path, "train" if train else "test") 
        this.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            this.format = json.load(fp)

        print("Сохранили \"папки + метка\" в format:") #debug
        print(json.dumps(this.format, indent=4, ensure_ascii=False)) #debug

        this.length = 0
        this.files = []
        this.targets = torch.eye(10)

        for _dir, _target in this.format.items():
            print(f"this.path=\"{this.path}\"\n_dir=\"{_dir}\"") #debug
            path = os.path.join(this.path, _dir)
            print(f"path: {path}\n") #debug

            list_files = os.listdir(path)
            this.length += len(list_files)
            this.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))
    def __getitem__(this, item):
        path_file, target = this.files[item]
        t = this.targets[target]
        img = Image.open(path_file)

        if this.transform:
            img = this.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(this):
        return this.length

class DigitNN(nn.Module):
    def __init__(this, input_dim, num_hidden, output_dim):
        super().__init__()
        this.layer1 = nn.Linear(input_dim, num_hidden)
        this.layer2 = nn.Linear(num_hidden, output_dim)
 
    def forward(this, x):
        x = this.layer1(x)
        x = nn.functional.relu(x)
        x = this.layer2(x)
        return x
    
if __name__== "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Нет GPU'}")
    
    model = DigitNN(28 * 28, 32, 10)

    to_tensor = tfs.ToImage()
    d_train = DigitDataset("./dataset", transform=to_tensor)
    train_data = data.DataLoader(d_train, batch_size=32, shuffle=True )

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    epochs = 2
    model.train()

    for _e in range(epochs):
        loss_mean = 0 # среднее значение функции потерь (по эпохе)
        lm_count = 0 # текущее кол-во слагаемых

        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_function(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + 1 * (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    d_test = DigitDataset("dataset", train=False, transform=to_tensor)
    test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
    Q = 0

    model.eval()

    for x_test, y_test in test_data:
        with torch.no_grad():
            p = model(x_test)
            p = torch.argmax(p, dim=1)
            y = torch.argmax(y_test, dim=1)
            Q += torch.sum(p == y).item()

    Q /= len(d_test)
    print(Q)