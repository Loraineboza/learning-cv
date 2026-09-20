import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(input, l1: nn.Linear, l2: nn.Linear):
    u1 = l1.forward(input)
    s1 = F.tanh(u1)

    u2 = l2.forward(s1)
    s2 = F.tanh(u2)

    return s2

def main():
    layer1 = nn.Linear(3, 2)
    layer2 = nn.Linear(2, 1)

    layer1.weight.data = torch.tensor([[0.7402, 0.6008, -1.3340], [0.2098, 0.4537, -0.7692]])
    layer1.bias.data = torch.tensor([0.5505, 0.3719])

    layer2.weight.data = torch.tensor([[-2.0719, -0.9485]])
    layer2.bias.data = torch.tensor([-0.1461])

    x = torch.FloatTensor([1, -1, 1]) # (1) квартира, (2) не любит тяжелый урок (3) красив
    y = forward(x, layer1, layer2)
    print(y.data)

if __name__ == '__main__':
    main()