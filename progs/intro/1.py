import torch

x = torch.randn((3, 4), dtype=torch.float32, requires_grad=True)
w = torch.randn((4, 2), requires_grad=True)
b = torch.zeros((2,), dtype=torch.float32, requires_grad=True)   
y = x @ w + b
loss = y.sum()
loss.backward()

print("x =", x.shape)
print("loss = ", loss)
print("w.grad = ", w.grad)
print("b.grad =", b.grad)

w.grad.zero_()
b.grad.zero_()

print(f"w.grad = {w.grad}; b.grad = {b.grad}")