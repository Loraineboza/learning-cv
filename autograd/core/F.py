import numpy as np

class Binary_cross_entropy:
    def __init__(self):
        pass

    def bce(self, pred, true):
        pred = np.array(pred)
        true = np.array(true)
        eps = 1e-15 #эпсилион -число с 15 нулями впереди
        pred = np.clip(pred, eps, 1-eps) #Если pred < eps: pred=eps; иначе если pred > 1-eps: pred=1-eps
        loss = -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))

        self.pred, self.true = (pred, true)
        return loss

    def backward(self):
        prod1 = self.true * 1/self.pred
        prod2 = (1-self.true) * (1/1-self.pred) * (-1)
        dl_dz = -(prod1 + prod2)


        return dl_dz / len(self.true)

class MSE:
    def __init__(self):
        pass
    def mse(self, pred, true):
        self.pred, self.true = (pred, true)
        return np.mean((pred - true) ** 2)

    def mse_grad(self):
        dl_dz = 2 * (self.pred - self.true) 
        return dl_dz / len(self.true)
