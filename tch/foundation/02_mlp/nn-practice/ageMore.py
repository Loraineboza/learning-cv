import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class AgeDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randint(0, 120, (size, 2), dtype=torch.float32)
        print(f"data:\n{self.data.tolist()}")
        
        self.labels = torch.tensor([max(d1, d2) for d1, d2 in self.data], dtype=torch.float32)
        print(f"labels:\n{self.labels}")

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

class AgeModel(nn.Module):
    def __init__(self, inp, hidden1, hidden2, out):
        super().__init__()
        self.layer1 = nn.Linear(inp, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, out)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def main():
    dataset = AgeDataset(1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = AgeModel(2, 8, 4, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    
    epochs = 100
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for x_batch, y_batch in dataloader:
            y_batch = y_batch.unsqueeze(1)  
            
            pred = model(x_batch)  
            loss = loss_func(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
    
    model.eval()
    print("\nОбучение завершено")

    testset = AgeDataset(200)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    print("\nРезультаты тестирования:")
    
    total_error = 0
    count = 0
    
    with torch.no_grad():
        for x_batch, y_batch in testloader:
            pred = model(x_batch)
            for i in range(len(x_batch)):
                error = abs(pred[i].item() - y_batch[i].item())
                total_error += error
                count += 1
                print(f"Родители: {x_batch[i].tolist()} -> Предсказано: {pred[i].item():.1f}, Правильно: {y_batch[i].item():.0f}, Ошибка: {error:.1f}")
    
    avg_error = total_error / count
    accuracy = max(0, 100 - avg_error * 1.5) 
    
    print(f"\nВердикт:")
    print(f"Средняя ошибка: {avg_error:.2f} лет")
    print(f"Примерная точность: {accuracy:.1f}%")
    print(f"Всего проверено: {count} примеров")
if __name__ == '__main__':
    main()