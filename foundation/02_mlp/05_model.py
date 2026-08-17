import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import randint

class NetGirl(nn.Module):
    def __init__(this, input_dim, num_hidden, output_dim): 
        super().__init__()
        this.layer1 = nn.Linear(input_dim, num_hidden)
        this.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(this, x):
        x = this.layer1(x)
        x = F.tanh(x)
        x = this.layer2(x)
        x = F.tanh(x)
        return x

def main():
    model = NetGirl(3, 2, 1)
    print(model, end="\n\n")

    x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                                (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
    y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
    total = len(y_train)

    optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    
    model.train()

    for i in range(1000):
        k = randint(0, total-1)
        y = model(x_train[k])
        
        loss = loss_func(y, y_train[k].view(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    for x, d in zip(x_train, y_train):
        with torch.no_grad():
            y = model(x)
            print(f"Выходное значение НС: {y.data} => {d}")

if __name__ == '__main__':
    main()