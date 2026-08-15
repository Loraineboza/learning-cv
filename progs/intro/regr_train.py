'''
    валидационный тренинг с выборками в пропорции 80/20 с нормальным распределением (среднее 0, std=1) и шумом randn_like
    с сэйвом checkpoint'ов модель и выбором с лучшим показателем посредством ранней остановки, если валидация превзошла треннинг
'''
from model import dataset, model, EPOCHS
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore, Style
import os
def calculate_validation_mse(model, iterloader: DataLoader, device: str) -> float:
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

def train_model(model, iter_train, iter_val, criterion, optimizer, epochs=EPOCHS, patience=10):
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
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n{Fore.GREEN}Обучение завершено!{Style.RESET_ALL}")
    print(f"Лучшая модель: {best_model_path}")
    print(f"Лучший Val MSE: {best_val_loss:.6f} на эпохе {checkpoint['epoch']+1}")
    
    return train_losses, val_losses