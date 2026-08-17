import torch

t = torch.FloatTensor([-3.5435])
alpha = 0.01

print(f"Вычисление Leaky ReLU ч/з встроенную функцию: F.leaky_relu({t.item():.4f}, alpha={alpha})")
print(f"результат = {torch.nn.functional.leaky_relu(t, alpha)}")

print(f"\nВычисление Leaky ReLU ч/з математическую формулу: max({alpha}*{t.item():.4f}, {t.item():.4f})")
print(f"результат = {max(alpha * t.item(), t.item())}")