import numpy as np

class Sequential:
    def __init__(self, *argv):
        self.layers = argv

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, dl_dy):
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy) 
        return dl_dy
    
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

class Linear:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.randn(in_channels, out_channels)
        self.b = np.random.randn(out_channels)

    def __call__(self, X):
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dl_dy):
        self.dW = self.X.T @ dl_dy
        self.dX = dl_dy @ self.W.T
        self.db = np.sum(dl_dy, axis=0)

        return self.dX

    def update(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db 

class MSE:
    def __init__(self):
        pass
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        return 2 * (self.pred - self.target) / len(self.pred)

class ReLU:
    def __init__(self):
        pass

    def __call__(self, X): 
        self.X = X
        return X * (X > 0)
    
    def backward(self, dL_dY):
        return dL_dY * (self.X > 0) 
 

X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

target = np.array([
    [5.0],
    [8.0],
    [11.0],
    [14.0]
])

model = Sequential(
    Linear(2, 3),
    ReLU(),
    Linear(3, 1)
)

mse = MSE()
lr = 0.01
epochs = 1000

for e in range(epochs):
    pred = model(X)
    loss = mse(pred, target)
    dL_dpred = mse.backward()
    model.backward(dL_dpred)
    model.update(lr)

    if e % 100 == 0:
        print(f"loss = {loss:.10f}")

print(f"\npred = {pred.tolist()}")
