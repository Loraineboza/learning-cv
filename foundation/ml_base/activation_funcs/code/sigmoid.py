import torch
import torch.nn.functional as F
import math as m

t = torch.FloatTensor([-3.5435])

print(f"Вычисление сигмоиды ч/з встроенную функцию: torch.sigmoid({t.item():.4f})")
print(f"результат = {torch.sigmoid(t)}")

print(f"\nВычисление сигмоиды ч/з математическую формулу: 1/(1+(m.e ** {t.item():.4f}))")
print(f"результат = { 1/(1+(m.e ** -t)) }")



