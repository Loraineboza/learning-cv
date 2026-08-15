'''
    валидационный трейнинг с выборками в пропорции 80/20 с нормальным распределением (среднее 0, std=1) и шумом randn_like
    с сэйвом checkpoint'ов модель и выбором с лучшим показателем посредством ранней остановки, если валидация превзошла треннинг
'''
from model import dataset, model, EPOCHS
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore, Style
import os
def calculate_accuracy(model, iterloader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in iterloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples
def train_model(model, iter_train: DataLoader, iter_val: DataLoader, criterion, optimizer, epochs=EPOCHS, patience=10):
    os.makedirs('model_of_regr_train', exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = 'model_of_regr_train/best_model.pth'
    stop_total = 0      
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        running_loss, avg_loss = (0.,0.)
        train_pbar = tqdm(iter_train, desc=f'{Fore.BLUE}Epoch {epoch+1}/{EPOCHS}{Style.BRIGHT}', ncols=100)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            avg_loss = loss.item() if batch_idx == 0 else 0.9 * avg_loss + 0.1 * loss.item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss:.4f}'
            })
        avg_train_loss = running_loss / len(iter_train)
        train_losses.append(avg_train_loss)
        #mse validation
        val_mse = calculate_accuracy(model, iter_val, device)
        val_losses.append(val_mse)
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            stop_total = 0  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_mse
            }, best_model_path)
            print(f"{Fore.GREEN}checkpoint of model; Val MSE: {val_mse:.6f}{Style.RESET_ALL}")
        else: 
            stop_total += 1  
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val MSE = {val_mse:.6f}")
        if stop_total >= patience:
            print(f"{Fore.YELLOW}выполнена ранняя остановка на эпохе {epoch+1} (нет улучшений {patience} эпох){Style.RESET_ALL}")
            break #break training of model
    #download the best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n{Fore.GREEN}training is end{Style.RESET_ALL}")
    print(f"best model: {best_model_path}")
    print(f"best Val MSE: {best_val_loss:.6f} на эпохе {checkpoint['epoch']+1}")
    
    return train_losses, val_losses

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
print(f"using device = {device}\n")
dataset_ = dataset(X, y)
train_size = int(0.8 * len(dataset_))
val_size = len(dataset_) - train_size
train_dataset, val_dataset = random_split(dataset_, [train_size, val_size])
iter_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False, prefetch_factor=2)
iter_val = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=2)
optimizer = torch.optim.SGD(model.parameters(), 0.01, 0.9, foreach=True, nesterov=True, weight_decay=1e-4)
criterion = torch.nn.MSELoss().to(device=device)

# start train
train_losses, val_losses = train_model(model, iter_train, iter_val, criterion, optimizer, EPOCHS, patience=10)