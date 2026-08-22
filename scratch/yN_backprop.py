#!/home/paul/venv/bin/python3

import numpy as np

def forward(X, W, b):
    return X @ W + b

def mse(pred, target):
    return np.mean((pred - target) ** 2)

def mse_grad(pred, target):
    return 2 * (pred - target) / len(pred)

X = np.array([
    [2.0, 3.0], [1.0, 4.0], [5.0, 2.0]
])

W = np.array([
    [10.0], [20.0]
])

b = np.array([5.0])

target = np.array([
    [10.0], [10.0], [10.0]
])

pred = forward(X, W, b)
loss = mse(pred, target)
print("loss", loss)

#dl/dW[0,0] = ?

# y1
dl_dy1 = mse_grad(pred, target)[0, 0]
dy1_dW00 = X[0, 0]

# y2
dl_dy2 = mse_grad(pred, target)[1, 0]
dy2_dW00 = X[1, 0]

# y3
dl_dy3 = mse_grad(pred, target)[2, 0]
dy3_dW00 = X[2, 0]

dl_dW00 = (
    dl_dy1 * dy1_dW00 +
    dl_dy2 * dy2_dW00 +
    dl_dy3 * dy3_dW00
)
print("dl_dW[0,0]", dl_dW00)

#dl/dW[1,0]=?

# y1
dl_dy1 = mse_grad(pred, target)[0, 0]
dy1_dW10 = X[0, 1]

# y2
dl_dy2 = mse_grad(pred, target)[1, 0]
dy2_dW10 = X[1, 1]

# y3
dl_dy3 = mse_grad(pred, target)[2, 0]
dy3_dW10 = X[2, 1]

dl_dW10 = (
    dl_dy1 * dy1_dW10 +
    dl_dy2 * dy2_dW10 +
    dl_dy3 * dy3_dW10
)
print("dl_dW[1,0]", dl_dW10)

# dl/db=?

dl_dy = mse_grad(pred, target)
dl_db = dl_dy[0, 0] + dl_dy[1, 0] + dl_dy[2, 0]
print("dl_db", dl_db)

lr = 0.01

W_new = W - lr * np.array([
    [dl_dW00],
    [dl_dW10]
])

b_new = b - lr * dl_db

print("\nW до:\n", W)
print("W после:\n", W_new)
print("b до:", b)
print("b после:", b_new)
