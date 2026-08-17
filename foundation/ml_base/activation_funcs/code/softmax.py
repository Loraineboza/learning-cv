import torch
import math as m

t = torch.FloatTensor([-3.5435, 0.5, 2.1, -1.2])

print(f"Вычисление Softmax ч/з встроенную функцию: F.softmax({t.numpy()}, dim=0)")
print(f"результат = {torch.nn.functional.softmax(t, dim=0)}")

print(f"\nВычисление Softmax ч/з математическую формулу: e^{t_i} / sum(e^{t_j})")
exp_t = torch.exp(t)
softmax_manual = exp_t / exp_t.sum()
print(f"результат = {softmax_manual}")