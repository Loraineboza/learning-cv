import numpy as np


def im2col(x, kh, kw, stride, pad):
    """x: (N,C,H,W) -> (N*OH*OW, C*kh*kw)"""
    N, C, H, W = x.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, OH, OW), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * OH
        for j in range(kw):
            j_max = j + stride * OW
            cols[:, :, i, j, :, :] = x_pad[:, :, i:i_max:stride, j:j_max:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1), OH, OW


class Conv2D:
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=None):
        pad = k // 2 if pad is None else pad
        self.stride, self.pad = stride, pad
        # He init
        self.W = np.random.randn(out_ch, in_ch * k * k) * np.sqrt(2 / (in_ch * k * k))
        self.b = np.zeros(out_ch)
        self.k, self.in_ch, self.out_ch = k, in_ch, out_ch
        # для backprop
        self._cols = self._shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        cols, OH, OW = im2col(x, self.k, self.k, self.stride, self.pad)
        self._cols, self._shape = cols, (N, C, H, W, OH, OW)
        out = cols @ self.W.T + self.b          # (N*OH*OW, out_ch)
        return out.reshape(N, OH, OW, self.out_ch).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, C, H, W, OH, OW = self._shape
        # dout: (N, out_ch, OH, OW) -> (N*OH*OW, out_ch)
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        self.dW = (dout_r.T @ self._cols).reshape(self.W.shape)
        self.db = dout_r.sum(axis=0)
        dcols = dout_r @ self.W                       # (N*OH*OW, C*k*k)
        # col2im
        dx = np.zeros((N, C, H + 2 * self.pad, W + 2 * self.pad), dtype=dout.dtype)
        dcols = dcols.reshape(N, OH, OW, C, self.k, self.k).transpose(0, 3, 4, 5, 1, 2)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + self.stride * OH:self.stride,
                       j:j + self.stride * OW:self.stride] += dcols[:, :, i, j]
        # аккуратный срез до исходного H, W (безопасно и при pad=0, и при нечётных размерах)
        return dx[:, :, self.pad:self.pad + H, self.pad:self.pad + W]


class BatchNorm2D:
    def __init__(self, ch, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(ch)
        self.beta = np.zeros(ch)
        self.run_mean = np.zeros(ch)
        self.run_var = np.ones(ch)
        self.momentum, self.eps = momentum, eps
        self.training = True

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.run_mean = self.momentum * self.run_mean + (1 - self.momentum) * mean
            self.run_var = self.momentum * self.run_var + (1 - self.momentum) * var
        else:
            mean, var = self.run_mean, self.run_var

        self._x = x
        self._mean, self._var = mean, var
        self._std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean.reshape(1, -1, 1, 1)) * self._std_inv.reshape(1, -1, 1, 1)
        self._x_hat = x_hat
        return self.gamma.reshape(1, -1, 1, 1) * x_hat + self.beta.reshape(1, -1, 1, 1)

    def backward(self, dout):
        N = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.dgamma = (dout * self._x_hat).sum(axis=(0, 2, 3))
        self.dbeta = dout.sum(axis=(0, 2, 3))
        dx_hat = dout * self.gamma.reshape(1, -1, 1, 1)
        return self._std_inv.reshape(1, -1, 1, 1) / N * (
            N * dx_hat
            - dx_hat.sum(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            - self._x_hat * (dx_hat * self._x_hat).sum(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        )


class MaxPool2D:
    def __init__(self, k=2, stride=2):
        self.k, self.stride = k, stride

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H - self.k) // self.stride + 1
        OW = (W - self.k) // self.stride + 1
        self._x_shape = x.shape
        self._argmax = np.zeros((N, C, OH, OW), dtype=np.int64)
        out = np.zeros((N, C, OH, OW), dtype=x.dtype)
        for i in range(OH):
            for j in range(OW):
                window = x[:, :, i * self.stride:i * self.stride + self.k,
                                 j * self.stride:j * self.stride + self.k]
                flat = window.reshape(N, C, -1)
                idx = flat.argmax(-1)
                self._argmax[:, :, i, j] = idx
                out[:, :, i, j] = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
        return out

    def backward(self, dout):
        N, C, H, W = self._x_shape
        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        _, _, OH, OW = dout.shape
        for i in range(OH):
            for j in range(OW):
                idx = self._argmax[:, :, i, j]
                di = idx // self.k
                dj = idx % self.k
                n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
                dx[n_idx, c_idx, i * self.stride + di, j * self.stride + dj] += dout[:, :, i, j]
        return dx


class LeakyReLU:
    def __init__(self, slope=0.1):
        self.slope = slope

    def forward(self, x):
        self._x = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, dout):
        return dout * np.where(self._x > 0, 1.0, self.slope)


class Upsample:
    """Nearest-neighbor ×2."""

    def __init__(self, scale=2):
        self.scale = scale

    def forward(self, x):
        return np.repeat(np.repeat(x, self.scale, axis=2), self.scale, axis=3)

    def backward(self, dout):
        s = self.scale
        N, C, H, W = dout.shape
        assert H % s == 0 and W % s == 0, "H и W должны делиться на scale"
        return dout.reshape(N, C, H // s, s, W // s, s).sum(axis=(3, 5))


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class CrossEntropyLoss:
    """logits: (N, C), targets: (N,) — индексы классов."""

    def __init__(self):
        self._probs = None
        self._targets = None

    def forward(self, logits, targets):
        probs = softmax(logits, axis=-1)
        N = logits.shape[0]
        eps = 1e-12
        loss = -np.log(probs[np.arange(N), targets] + eps).mean()
        self._probs = probs
        self._targets = targets
        return loss

    def backward(self):
        N = self._probs.shape[0]
        dlogits = self._probs.copy()
        dlogits[np.arange(N), self._targets] -= 1.0
        return dlogits / N


class SGD:
    """SGD с momentum и weight decay. params — список слоёв с .W/.b/.gamma/.beta."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self._v = {}

    def step(self):
        for i, layer in enumerate(self.params):
            for name in ("W", "b", "gamma", "beta"):
                if not hasattr(layer, name):
                    continue
                p = getattr(layer, name)
                g = getattr(layer, "d" + name)
                if self.wd:
                    g = g + self.wd * p
                key = (i, name)
                if key not in self._v:
                    self._v[key] = np.zeros_like(p)
                self._v[key] = self.momentum * self._v[key] - self.lr * g
                setattr(layer, name, p + self._v[key])

    def zero_grad(self):
        self._v = {}
