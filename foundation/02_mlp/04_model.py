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
    
    
if __name__ == '__main__':
    main()