import torch

t = torch.FloatTensor([-3.5435])

print(f"Вычисление ReLU ч/з встроенную функцию: torch.relu({t.item():.4f})")
print(f"результат = {torch.relu(t)}")

print(f"\nВычисление ReLU ч/з математическую формулу: max(0, {t.item():.4f})")
print(f"результат = {max(0, t.item())}")