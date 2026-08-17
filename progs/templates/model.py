from torch.utils.data import Dataset 
import torch.nn as nn
class dataset(Dataset):
    def __init__(this, X, y):
        this.x_i = X; this.y_i = y
    def __len__(this):
        return len(this.x_i)
    def __getitem__(this, i):
        return this.x_i[i], this.y_i[i]

model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
EPOCHS=50