import torch
import math as m

t = torch.FloatTensor([-3.5435])

print(f"Вычисление Softplus ч/з встроенную функцию: F.softplus({t.item():.4f})")
print(f"результат = {torch.nn.functional.softplus(t)}")

print(f"\nВычисление Softplus ч/з математическую формулу: ln(1 + e^{t.item():.4f})")
softplus_manual = torch.log(1 + torch.exp(t))
print(f"результат = {softplus_manual}")