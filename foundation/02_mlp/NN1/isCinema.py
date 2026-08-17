
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import randint

class IsCinema(nn.Module):
    def __init__(self, inp, inp_out, out):
        super().__init__()
        self.layer1 = nn.Linear(inp, inp_out)
        self.layer2 = nn.Linear(inp_out, out)
    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x
    
def main():
    model = IsCinema(3, 2, 1)

    x_train = torch.FloatTensor([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)])
    
    y_train = torch.FloatTensor([1, 1, -1, -1, -1, -1, -1, -1])
    total = len(x_train)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1) # => 0.8729
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1) # => 0.9890
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.1) # => 0.9949 (точнее всех)
    # loss_f = nn.MSELoss # 0.9949
    loss_f = nn.L1Loss() # 1.0

    '''
    Вердикт: лучший результат показали функции оптимизатора и потери RMSprop и L1Loss,
    соответственно
    '''
    model.train()

    for _ in range(1000):
        k = randint(0, total-1)
        y = model(x_train[k])
        loss = loss_f(y, y_train[k].view(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    for x, y in zip(x_train, y_train):
        with torch.no_grad():
            ret = model(x)
            loss = loss_f(ret, y.view(1))
            print(f"loss = {loss}")
            print(f"Выходное значение НС: {ret.data} => {y}\n")
    
    
if __name__ == "__main__":
    main()