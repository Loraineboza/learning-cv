from autograd import Tensor
from F import MSE 
import numpy as np


X = Tensor(np.array([[1.0, 2.0],
                     [2.0, 1.0],
                     [3.0, 0.5]]), name="X")
W = Tensor(np.random.randn(2, 1), name="W")
b = Tensor(np.random.randn(1), name="b")

y_pred = (X @ W + b).sum(axis=1, keepdims=True)
y_true = Tensor(np.array([[1.0], [0.0], [1.0]]), name="y_true")

loss = MSE(y_pred, y_true)
loss.backward()

print("Loss:", loss.data)
print("W.grad:", W.grad)
print("b.grad:", b.grad)
