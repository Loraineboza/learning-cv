import numpy as np

class Linear:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.randn(in_channels, out_channels)
        self.bias = np.random.randn(in_channels, out_channels)
    
    def forward(self, X):
        self.X = X
        z = X @ self.W + self.bias
        return z
    
    def backward(self) 

class Binary_cross_entropy:
    def bce(pred, target):
        #подготовка данных для потребности в векторизации, если будет передан список
        pred = np.array(pred)
        target = np.array(target)

        eps = 1e-15 #эпсилион число с 15 нулями впереди. Если pred < eps: pred=eps; иначе если pred > 1-eps: pred=1-eps
                    # необходимо, чтобы избежать log(0) или log(1), потому что комп не умеет считать логарифм от нуля (log(0) равен ∞
        target = np.clip(pred, eps, 1-eps)
        
        #считаю ошибку
        loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return loss 

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

target = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])
