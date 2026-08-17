'''
линейный слой и функция потерь вручную
'''
import torch
import torch.nn as nn   
X = torch.rand((5, 3), requires_grad=True, dtype=torch.float32) * (1 - (-1)) + (-1)
y_true = 2 * X [:, 0:1] + 0.5
layer = nn.Linear(3, 1, bias=False)
mse = nn.MSELoss()
y_pred = layer(X)
loss = mse(y_pred, y_true)
print("loss:", loss.item()) # до update w
loss.backward()
layer.weight.data -= 0.01 * layer.weight.grad
layer.weight.grad.zero_()

y_pred = layer(X)
print(f"loss: {mse(y_pred, y_true).item():.10}") # after


