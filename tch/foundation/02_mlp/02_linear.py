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
    
    print(layer1.weight)
    print(layer1.bias)

if __name__ == '__main__':
    main()