import numpy as np
from view import (
    Conv2D, BatchNorm2D, LeakyReLU, MaxPool2D,
    Flatten, Linear, CrossEntropyLoss, SGD,
)

x = np.random.randn(8, 3, 16, 16).astype(np.float32)
y = np.random.randint(0, 10, size=8)

conv = Conv2D(3, 8, k=3, stride=1, pad=1)
bn = BatchNorm2D(8)
act = LeakyReLU(0.1)
pool = MaxPool2D(2, 2)
flat = Flatten()
fc = Linear(8 * 8 * 8, 10)
loss_fn = CrossEntropyLoss()
opt = SGD([conv, bn, fc], lr=1e-3)

for step in range(20):
    h = conv.forward(x)
    h = bn.forward(h)
    h = act.forward(h)
    h = pool.forward(h)
    h = flat.forward(h)
    logits = fc.forward(h)
    loss = loss_fn.forward(logits, y)

    d = loss_fn.backward()
    d = fc.backward(d)
    d = flat.backward(d)
    d = pool.backward(d)
    d = act.backward(d)
    d = bn.backward(d)
    conv.backward(d)
    opt.step()

    print(step, loss)
