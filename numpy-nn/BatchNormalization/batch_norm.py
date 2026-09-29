import numpy as np

class BN:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        #train parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        #dl/dg 
        self.dgamma = None
        #dl/db
        self.dbeta = None
        
        #буферы скользящего среднего для инференса
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            x_minus_mean = x - mean
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_norm = x_minus_mean * inv_std
            
            #обновление истории для инференса
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            self.cache = (x_norm, x_minus_mean, inv_std, var)
        else:
            #в инференсе используются накопленные данные
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        #dout: градиент от следующего слоя, форма [batch_size, num_features]
        x_norm, x_minus_mean, inv_std, var = self.cache
        N = dout.shape[0]  # Размер батча
        
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * x_norm, axis=0)
        
        #градиент по нормализованному входу (dx_norm)
        dx_norm = dout * self.gamma
        
        #градиент по дисперсии (dvar)
        dvar = np.sum(dx_norm * x_minus_mean * -0.5 * (inv_std ** 3), axis=0)
        
        #градиент по среднему арифметическому (dmean)
        dmean = np.sum(dx_norm * -inv_std, axis=0) + dvar * np.mean(-2.0 * x_minus_mean, axis=0)
        
        #градиент по входу x (dx), который пойдет в предыдущий слой
        dx = dx_norm * inv_std + dvar * 2.0 * x_minus_mean / N + dmean / N
        
        return dx 

#как происходит обновление обуч. параметров геммы и беты:
#при lr = 3e-1
#bn_layer.gamma -= lr * bn_layer.dgamma
#bn_layer.beta -= lr * bn_layer.dbeta

