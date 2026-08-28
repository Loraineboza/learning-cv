import numpy as np

class Binary_cross_entropy:
    def __init__(self):
        pass

    def bce(pred, true):
        pred = np.array(pred)
        true = np.array(true)
        eps = 1e-15 #эпсилион -число с 15 нулями впереди
        pred = np.clip(pred, eps, 1-eps) #Если pred < eps: pred=eps; иначе если pred > 1-eps: pred=1-eps
        loss = -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))
        return loss 

    def backward(pred, true):
        pred = np.array(pred)
        true = np.array(true)
        pred = np.clip(pred, 1e-15, 1- 1e-15)

        #найти произвольные по матрешке: если встречается ф-я, найти производную самой ф-ии и потом для ее операндов. Производные перемножить, если они от одной ф-ии
        #dlog(x) = 1/x

        prod1 = true * 1/pred
        prod2 = (1-true) * (1/1-pred) * (-1)
        dl_dz = -(prod1 + prod2)
        
        return dl_dz / len(true)


class Linear:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.randn(in_channels, out_channels)
        self.bias = np.random.randn(in_channels, out_channels)
    
    def forward(self, X):
        self.X = X
        z = X @ self.W + self.bias
        return z
    
    def backward(self) 

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

true = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])
