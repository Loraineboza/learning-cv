import torch
import math as m

t = torch.FloatTensor([-3.5435])

print(f"Вычисление tanh ч/з встроенную функцию: torch.tanh({t.item():.4f})")
print(f"результат = {torch.tanh(t)}")

print(f"\nВычисление tanh ч/з математическую формулу: (e^{t.item():.4f} - e^(-{t.item():.4f})) / (e^{t.item():.4f} + e^(-{t.item():.4f}))")
print(f"результат = {(torch.exp(t) - torch.exp(-t)) / (torch.exp(t) + torch.exp(-t))}")