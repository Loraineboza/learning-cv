import torch
import math as m

t = torch.FloatTensor([-3.5435])
alpha = 1.0

print(f"Вычисление ELU ч/з встроенную функцию: F.elu({t.item():.4f}, alpha={alpha})")
print(f"результат = {torch.nn.functional.elu(t, alpha)}")

print(f"\nВычисление ELU ч/з математическую формулу: alpha * (e^{t.item():.4f} - 1) если t < 0, иначе t")
elu_manual = alpha * (torch.exp(t) - 1) if t.item() < 0 else t
print(f"результат = {elu_manual}")