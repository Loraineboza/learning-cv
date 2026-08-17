import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class OddEventDataset(Dataset):
    def __init__(this, size=1000):
        this.data = torch.randint(0, 100, (size,), dtype=torch.float32)
        this.labels = this.data % 2

    def __getitem__(this, idx):
        return this.data[idx], this.labels[idx]

    def __len__(this):
        return len(this.data)

class OddEvenNN(nn.Module):
    def __init__(this):
        super().__init__()
        this.fc1 = nn.Linear(1, 16)
        this.fc2 = nn.Linear(16, 8)
        this.fc3 = nn.Linear(8, 1)

    def forward(this, x):
        x = torch.relu(this.fc1(x))
        x = torch.relu(this.fc2(x))
        x = torch.sigmoid(this.fc3(x))
        return x


dataset = OddEventDataset(size=500)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = OddEvenNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for e in range(50):
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.unsqueeze(1)
        y_batch = y_batch.unsqueeze(1).float()

        pred = model(X_batch) 
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if e % 10 == 0:
        print(f"Epoch {e}, Loss: {loss.item():.4f}")


test_dataset = OddEventDataset(size=100)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

correct = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.unsqueeze(1)
        pred = model(X_batch)
        predicted = (pred > 0.5).float()
        correct += (predicted.squeeze() == y_batch).sum().item()

accuracy = correct / len(test_dataset)
print(f"\nТочность на тесте: {accuracy * 100:.2f}%")