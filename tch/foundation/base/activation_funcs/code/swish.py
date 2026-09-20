import torch
import math as m

t = torch.FloatTensor([-3.5435])

print(f"Вычисление Swish ч/з встроенную функцию: F.silu({t.item():.4f})")
print(f"результат = {torch.nn.functional.silu(t)}")

print(f"\nВычисление Swish ч/з математическую формулу: {t.item():.4f} * sigmoid({t.item():.4f})")
swish_manual = t * torch.sigmoid(t)
print(f"результат = {swish_manual}")