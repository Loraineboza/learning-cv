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

class Sigmoid:
    def __init__(self):
        pass
    def sigmoid(self, x):
        #save input x, чтобы потом использовать в backward
        self.x = x
        
        #вычисляю сигмоиду по формуле s = 1 / (1 + exp(-x))
        s = 1 / (1 + np.exp(-x))
        return s

    def backward(self, x):
        #раскладываю сигмоиду на правилу цепочке вложенных функций:
        #s = 1/u, где u = 1+v, где v = exp(w), где w = -x
        
        #prod1: производная самой внешней функции s = 1/u = u^(-1)
        #по правилу (u^n)' = n * u^(n-1), где n = -1
        #получаем: -1 * u^(-2) = -1 / u^2
        #подставляем u = 1 + exp(-x)
        #P.S штрих в (u^n)' означает производная от того, что в скобках
        prod1 = -1 / (1 + np.exp(-x)) ** 2
        
        #prod2: производная функции u = 1+v по v
        #производная от 1 равна 0, производная от v по v равна 1
        #итого: 0 + 1 = 1
        prod2 = 1
        
        #prod3: производная функции v = exp(w) по w
        #экспонента равна своей производной: (exp(w))' = exp(w)
        #подставляем w = -x, получаем exp(-x)
        prod3 = np.exp(-x)
        
        #prod4: производная функции w = -x по x
        #производная от -x равна -1, так как d(-x)/dx = -1
        prod4 = -1
        
        #применяем цепное правило: перемножаем все производные
        #ds/dx = (ds/du) * (du/dv) * (dv/dw) * (dw/dx)
        return prod1 * prod2 * prod3 * prod4turn prod1 * prod2 * prod3 * prod

class Linear:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.randn(in_channels, out_channels)
        self.bias = np.random.randn(in_channels, out_channels)
    
    def forward(self, X):
        self.X = X
        z = X @ self.W + self.bias
        return z
    
    def backward(self, dl_dz): 
        w_grad = self.X.T @ dl_dz #dl_dw = ? - найти производную для каждого w[i,j]. Делаю ч/з векторизацию транспонирования 
                                  #После чего умножить ее на каждый элемент dl_dz[i,j] 
        b_grad = dl_dz
        return w_grad, b_grad

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
