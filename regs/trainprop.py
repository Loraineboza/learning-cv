''' 
    Валидационный тренинг с выборками в пропорции 80/20 с нормальным распределением (среднее 0, std=1) и шумом radn_like
'''
from model import dataset, model, EPOCHS
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore, Style
X = torch.normal(mean=0.0, std=1.0, size=(100, 5))
y = 3 * X[:, 0:1] - 2 * X[:, 1:2] + .5
y = y + .1 * torch.randn_like(y)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
).to(device=device)
print(f"device = {device}\n")
dataset_ = dataset(X, y)
train_size = int(0.8 * len(dataset_))
val_size = len(dataset_) - train_size
train_dataset, val_dataset = random_split(dataset_, [train_size, val_size])
iter_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False, prefetch_factor=2)
iter_val = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False, prefetch_factor=2)
optimizer = torch.optim.SGD(model.parameters(), 0.01, 0.9, foreach=True, nesterov=True, weight_decay=1e-4)
mse = torch.nn.MSELoss().to(device=device)
for epoch in range(EPOCHS):
    loss_m, count_m = (0,0)
    model.train()
    train_pbar = tqdm(iter_train, desc=f'{Fore.BLUE}epoch {epoch+1}/{EPOCHS}{Style.BRIGHT}', ncols=100)

    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = mse(outputs, labels)
        loss.backward()
        optimizer.step()
        count_m+=1
        loss_m = 1/count_m * loss.item() + (1-1/count_m) * loss_m
        train_pbar.set_postfix({
            'Loss now': f'{loss.item():.4f}',
            'Loss avg': f'{loss_m:.4f}'
        })
    print(f"{Fore.CYAN}epoch={epoch+1}: loss_mean ={loss_m:.10f}")
