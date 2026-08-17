import torch
import torch.nn as nn
import torch.nn.functional as F

layer1 = nn.Linear(3, 2)

x = torch.tensor([1.0, 2.0, 3.0])

output = layer1(x)

print("Входные числа:", x)
print("Веса слоя:\n", layer1.weight)
print("Смещения слоя:", layer1.bias)
print("Выходные числа:", output)