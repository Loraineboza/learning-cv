import torch
import math as m

t = torch.FloatTensor([-3.5435])
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946

print(f"Вычисление SELU ч/з встроенную функцию: F.selu({t.item():.4f})")
print(f"результат = {torch.nn.functional.selu(t)}")

print(f"\nВычисление SELU ч/з математическую формулу: scale * (alpha * e^{t.item():.4f} - alpha) если t < 0, иначе scale * t")
if t.item() < 0:
    selu_manual = scale * (alpha * torch.exp(t) - alpha)
else:
    selu_manual = scale * t
print(f"результат = {selu_manual}")