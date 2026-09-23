import numpy as np
from im2col import (
    Conv2D, BatchNorm2D, LeakyReLU, MaxPool2D,
    Flatten, Linear, CrossEntropyLoss, SGD,
)

x = np.random.randn(8, 3, 16, 16).astype(np.float32) # .png 16x16, RGB; 8 batch
y = np.random.randint(0, 10, size=8) # true "logits"

conv = Conv2D(3, 8, k=3, stride=1, pad=1)
bn = BatchNorm2D(8)
act = LeakyReLU(0.1)
pool = MaxPool2D(2, 2)
flat = Flatten()
fc = Linear(8 * 8 * 8, 10)
loss_fn = CrossEntropyLoss()
opt = SGD([conv, bn, fc], lr=1e-3)

for step in range(20)
    h = conv.forward(x) # h = conv(x)
    h = bn.forward(h) # h = bn(h)
    h = act.forward(h) # h = act(h)
    h = pool.forward(h) # h = pool(h)
    h = flat.forward(h)   # h = flat(h) = flatten(h)
    logits = fc.forward(h) # logits = fc(h)
    loss = loss_fn.forward(logits, y)  # loss = L(logits, y)

    # Backward pass
    d = loss_fn.backward()# dL/dlogits
    d = fc.backward(d) # dL/dh (перед fc) = dL/dlogits * dlogits/dh
    d = flat.backward(d) # dL/dh (перед flat) = dL/dh_after_flat * dh_after_flat/dh_before_flat
    d = pool.backward(d) # dL/dh (перед pool) = dL/dh_after_pool * dh_after_pool/dh_before_pool
    d = act.backward(d) # dL/dh (перед act) = dL/dh_after_act * dact/dh_before_act
    d = bn.backward(d) # dL/dh (перед bn) = dL/dh_after_bn * dbn/dh_before_bn
    conv.backward(d) # dL/dx (и dL/dW_conv, dL/db_conv внутри)
    opt.step() # W -= lr * dL/dW

    print(step, loss)
