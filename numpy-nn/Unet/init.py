import numpy as np


def he_normal(shape, rng):
    return rng.normal(scale=np.sqrt(2.0 / np.prod(shape[1:])), size=shape)

def init_conv(cin, cout, k, rng):
    return he_normal((cout, cin, k, k), rng), np.zeros(cout)


def init_bn(c):
    return np.ones(c), np.zeros(c)


def init_dense(fin, fout, rng):
    return he_normal((fin, fout), rng), np.zeros(fout)

def init_block(cin, cout, rng):
    w1, b1 = init_conv(cin, cout, 3, rng)
    g1, be1 = init_bn(cout)
    w2, b2 = init_conv(cout, cout, 3, rng)
    g2, be2 = init_bn(cout)
    return (w1, b1, g1, be1, w2, b2, g2, be2)


def init_model(num_classes=10, rng=None):
    rng = rng or np.random.default_rng(0)
    w, b = init_conv(3, 64, 7, rng)
    g, be = init_bn(64)
    m = {"stem": (w, b, g, be)}

    m["l1"] = [init_block(64, 64, rng), init_block(64, 64, rng)]
    m["l2"] = [init_block(64, 128, rng), init_block(128, 128, rng)]
    m["l3"] = [init_block(128, 256, rng), init_block(256, 256, rng)]
    m["l4"] = [init_block(256, 512, rng), init_block(512, 512, rng)]

    m["fc"] = init_dense(512, num_classes, rng)
    return m

