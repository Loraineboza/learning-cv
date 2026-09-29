import numpy as np 

class BN:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps 
        self.momentum = momentum

        #train parameters "gamma" and "beta"
        self.gamma = np.zeros(num_features)
        self.beta = np.ones(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):

        if x.ndim == 2:
            keepdims = False 
            axis = 0
        else x.ndim == 4:
            keepdims = True 
            axis = (0, 2, 3)

            #для корректного умножения матриц нужно 
            #превратить gamma и beta из формы (c,) в (1, c, 1, 1)
            gamma = self.gamma[:, None, None]
            beta = self.beta[:, None, None]
            running_mean = self.running_mean[:, None, None]
            running_var = self.running_var[:, None, None]
        else:
            raise valueerror("поддерживается только 2d / 4d")

        if training:
            #model.train()
            mean = np.mean(x, axis=axis, keepdims=keepdims)
            var = np.var(x, axis=axis, keepdims=keepdims)

            x_norm = (x - mean) / np.sqrt(var + self.eps)

            mean_flat = mean.squeeze() if keepdims else mean 
            var_flat = var.squeeze() if keepdims else var 

            self.running_mean = (1 - self.momentum) * self.running_mean 
                + self.momentum * mean_flat 
            self.running_var = (1- self.momentum) * self.running_var + self. 
                + self.momentum * var_flat
            
            g = gamma if x.ndim == 4 else self.gamma 
            b = beta if x.ndim == 4 else self.beta 
        else:
            #инференс
            m = running_mean if x.ndim == 4 else self.running_mean 
            v = running_var if x.ndim == 4 else self.running_var 
            g = gamma if x.ndim == 4 else self.gamma 
            b = beta if x.ndim == 4 else self.beta 
            
            x_norm = (x - m) / np.sqrt(v + self.eps)
        
        out = g * x_norm + b 
        return out 


