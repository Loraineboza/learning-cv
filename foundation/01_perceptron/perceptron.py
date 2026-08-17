import torch

X = torch.tensor([
    [10/50, 80/100, 0.0],   
    [5/50,  70/100, 1.0],   
    [50/50, 10/100, 1.0],   
    [0/50,  0/100,  1.0],   
    [0/50,  0/100,  0.0],   
], dtype=torch.float32)

w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print("Начальные веса (random):")
print(f"  Вес расстояния от авто: {w[0].item():.4f}")
print(f"  Вес скорости авто:   {w[1].item():.4f}")
print(f"  Вес зел. света (вкл/выкл):    {w[2].item():.4f}")

y_true = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0], 
    [0.0],  
], dtype=torch.float32)

def mse_loss(pred, true):
    return ((pred - true) ** 2).mean()

lr = 5.0
for i in range(0, 200):
    y_pred = torch.sigmoid(torch.mm(X, w) + b)
    loss = mse_loss(y_pred, y_true)
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b-= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

print("debug:")
print("обучение нейрона завершено. Итоговые веса:")
print(f"  Вес расстояния от авто: {w[0].item():.4f}")
print(f"  Вес скорости авто:   {w[1].item():.4f}")
print(f"  Вес зел. света (вкл/выкл):    {w[2].item():.4f}")

distance = 29
speed = 10
green_light = 1
new_X = torch.tensor([distance/50, speed/100, green_light], dtype=torch.float32)
# pred = torch.mm(new_X, w)
pred = torch.sigmoid(new_X @ w)
print(f"\nМашина едет (расстояние от нее {distance} м., ее скорость {speed} км/ч, зел. свет {"горит" if green_light else "не горит"}):")
print(f"{pred.item():.2f} [~1.0 - безопасно переходить/ ~0.0 - опасно переходить]")
