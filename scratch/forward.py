import numpy as np

X = np.array([3.0, 1.0, 4.0, 2.0]) # signals
Y_target = np.array([10.0, 4.0, 13.0, 7.0])  # labels

w = 2.0
b = 1.0
lr = 0.01
epochs = 100

def forward(x, w, b):
    z = x * w + b
    y = z
    return y

def loss(y, y_target):
    return np.mean((y - y_target) ** 2)

def backward(x, y, y_target, w, b):
    dL_dy = 2 * (y - y_target) / len(x) # dl/dy = 2*(y-target)*1, однако используеся MSE, поэтому квадрат / len(x)
    # в противном случае обычно используется ф-я SSE, где производная равна сумме квадратов, без среднеквадратичности 
    dy_dz = 1.0 #y=z
    dz_dw = x #dW = nW = n
    dz_db = 1.0 #... +/- b = +1/-1

    dL_dw = np.sum(dL_dy * dy_dz * dz_dw) #перемножить производные одного направления и просуммировать все эти направления 
    dL_db = np.sum(dL_dy * dy_dz * dz_db) #аналогично
    return dL_dw, dL_db #частные производные для w, bias = dl/dw, dl/db - соответственно; 

print("Старт обучения")
print(f"Начальный w = {w}, b = {b}\n")

for epoch in range(epochs):
    y_pred = forward(X, w, b)
    L = loss(y_pred, Y_target)

    dL_dw, dL_db = backward(X, y_pred, Y_target, w, b) #дифференцирование для w и bias. найти dl/dw и dl/db после чего оптимизировать w и bias на соответствующие производные 

    w -= lr * dL_dw # вес оптимизируется ч/з частную производную. Здесь нет градиента, потому что w является скаляром, где shape=(1,) 
    #А из скаляра невозможно построить вектор производных (др. словами градиент), потому что производная только одна 
    b -= lr * dL_db# аналогично, bias.grad не является градиентом

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss = {L:.6f} | w = {w:.6f} | b = {b:.6f}")

print("\nПосле обучения:")
y_final = forward(X, w, b)
L_final = loss(y_final, Y_target)

print(f"w = {w}")
print(f"b = {b}")
print(f"Предсказания: {y_final}")
print(f"Целевые:      {Y_target}")
print(f"Итоговый Loss = {L_final}")