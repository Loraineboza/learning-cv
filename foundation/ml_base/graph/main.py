'''
Здесь показано, как добавить график к любой pytorch программе, чтобы наглядно понять,
почему нейросеть плохо обучается 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import matplotlib.pyplot as plt  # ← import 

class Xor(nn.Module):
    def __init__(this, i, io, o):
        super().__init__()
        this.layer1 = nn.Linear(i, io)
        this.layer2 = nn.Linear(io, o)
    def forward(this, x):
        x = F.relu(this.layer1(x))
        x = torch.sigmoid(this.layer2(x))
        return x

def main():
    model = Xor(2, 4, 1)
    
    x_train = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
    y_train = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    length_x = len(x_train)
    
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=15.)
    error_f = torch.nn.MSELoss()
    
    # добавь список потерь
    losses = []
    
    model.train()
    for i in range(5000):
        index = randint(0, length_x-1)
        ret = model(x_train[index])
        loss = error_f(ret, y_train[index].unsqueeze(0))    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # сохраняй все потери
        losses.append(loss.item())
    
    # создай график
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='blue', linewidth=1)
    plt.title('График потерь (все итерации)')
    plt.xlabel('Итерация')
    plt.ylabel('Потеря (Loss)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses[-100:], color='red', linewidth=2)
    plt.title('Последние 100 итераций')
    plt.xlabel('Итерация')
    plt.ylabel('Потеря (Loss)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()  # и отобрази его
    
    model.eval()
    with torch.no_grad():
        print("result:")
        for xi, yi in zip(x_train, y_train):
            ret = model(xi)
            print(f"{xi.tolist()} → {ret.item():.4f} (true: {yi:.0f})")

if __name__ == '__main__':
    main()