import torch
import math as m

t = torch.FloatTensor([-3.5435])

print(f"Вычисление GELU ч/з встроенную функцию: F.gelu({t.item():.4f})")
print(f"результат = {torch.nn.functional.gelu(t)}")

print(f"\nВычисление GELU ч/з математическую формулу: 0.5 * {t.item():.4f} * (1 + erf({t.item():.4f}/sqrt(2)))")
# Приближенная формула: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
gelu_manual = 0.5 * t * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/m.pi)) * (t + 0.044715 * t**3)))
print(f"результат (приближенный) = {gelu_manual}")