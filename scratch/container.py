import numpy as np

class Sequential:
    def __init__(self, *argv):
        self.layers = argv

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, dl_dy):
        # прохожу слои в обратном порядке и передаю градиент от предыдущего слоя к следующему
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy) 
        return dl_dy
    
    def update(self, lr):
        # обновляю веса только у тех слоев, у которых есть метод update
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
        # dl_dy - это градиент, который пришел с предыдущего слоя (или с функции потерь)
        # В forward я делал X @ W + b. Мне нужно найти производные по W, b и X.
        
        # Каждый вес W[i,j] участвует во всех yN (для всех строк X). Поэтому его градиент 
        # считается как сумма по всем путям: dL/dW[i,j] = сумма по всем N (dL/dyN * dyN/dW[i,j]).
        # Векторизованным способом я делаю: X.T @ dl_dy - это делает то же самое за один раз.
        # X.T имеет размер (in_channels, batch_size), dl_dy имеет (batch_size, out_channels).
        # При умножении суммируются все строки (все объекты), и я получаю (in_channels, out_channels)
        self.dW = self.X.T @ dl_dy
        
        # dL/dX - градиент, который пойдет дальше к предыдущему слою.
        # В forward X участвовал во всех yN, а каждый yN влиял на Loss.
        # Формула: dL/dX = dL/dY @ W.T. Я умножаю на транспонированный W, потому что
        # в forward я умножал X на W, а при обратном проходе мне нужно "отменить" это умножение.
        self.dX = dl_dy @ self.W.T
        
        # dL/db - градиент по смещению. Каждый bias b[j] участвует во всех yN для всех объектов.
        # Суммирую по оси 0 (по всем объектам), чтобы получить градиент для каждого нейрона.
        self.db = np.sum(dl_dy, axis=0)

        return self.dX

    def update(self, lr):
        # градиентный спуск: обновляю веса и смещения в сторону уменьшения Loss
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
        # производная MSE по предсказаниям: dL/dpred = 2 * (pred - target) / n
        # Делю на len(pred) потому что в forward я брал среднее (mean)
        return 2 * (self.pred - self.target) / len(self.pred)

class ReLU:
    def __init__(self):
        pass

    def __call__(self, X): 
        self.X = X
        return X * (X > 0)
    
    def backward(self, dL_dY):
        # Производная ReLU: если X > 0, то градиент проходит дальше (умножается на 1),
        # если X <= 0, то градиент обнуляется (умножается на 0).
        # Использую self.X > 0 как маску: True = 1, False = 0
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
    
    # Я передаю dL/dpred в model.backward().
    # Внутри Sequential.backward() этот градиент последовательно проходит через все слои в обратном порядке.
    # Каждый слой вычисляет свой dL/dW, dL/db и возвращает dL/dX для предыдущего слоя.
    # В итоге все градиенты накапливаются в каждом слое (в self.dW, self.db).
    model.backward(dL_dpred)
    model.update(lr)

    if e % 100 == 0:
        print(f"loss = {loss:.10f}")

print(f"\npred = {pred.tolist()}")
