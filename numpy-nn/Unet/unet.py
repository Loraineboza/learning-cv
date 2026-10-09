import numpy as np
from init import init_conv, init_bn, init_dense, init_model, he_normal


def conv2d(x, w, b, stride=1, pad=1):
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((n, f, oh, ow))
    for i in range(oh):
        for j in range(ow):
            p = xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(p, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def batchnorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return g.reshape(1, -1, 1, 1) * (x - mu) / np.sqrt(var + eps) + b.reshape(1, -1, 1, 1)


def relu(x):
    return np.maximum(x, 0.0)


def avgpool(x):
    return x.mean(axis=(2, 3), keepdims=True)


def dense(x, w, b):
    return x @ w + b

def basic_block(x, p, stride=1):
    w1, b1, g1, be1, w2, b2, g2, be2 = p
    out = relu(batchnorm(conv2d(x, w1, b1, stride=stride), g1, be1))
    out = batchnorm(conv2d(out, w2, b2), g2, be2)

    if stride != 1 or x.shape[1] != out.shape[1]:
        s = x[:, :, ::stride, ::stride].mean(axis=(2, 3), keepdims=True)
        s = np.broadcast_to(s, (x.shape[0], s.shape[1], out.shape[2], out.shape[3]))
        shortcut = np.zeros_like(out)
        c = min(s.shape[1], out.shape[1])
        shortcut[:, :c] = s[:, :c]
    else:
        shortcut = x

    return relu(out + shortcut)


def forward(x, m):
    w, b, g, be = m["stem"]
    x = relu(batchnorm(conv2d(x, w, b, stride=2, pad=3), g, be))

    for key, stride in [("l1", 1), ("l2", 2), ("l3", 2), ("l4", 2)]:
        for i, p in enumerate(m[key]):
            x = basic_block(x, p, stride=stride if i == 0 else 1)

    x = avgpool(x).reshape(x.shape[0], -1)
    wf, bf = m["fc"]
    return dense(x, wf, bf)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    m = init_model(10, rng)
    x = rng.normal(size=(2, 3, 32, 32))
    logits = forward(x, m)
    probs = softmax(logits)
    loss = cross_entropy(probs, np.array([3, 7]))
    print(logits.shape, probs.shape, round(float(loss), 4))
