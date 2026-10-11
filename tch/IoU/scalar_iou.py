#вычисления происходит по скалярно: max[0, y2-y1], где y2/y1 является скаляром  
#промежуточное решение

import numpy as np

def area(box):
    x1, y1, x2, y2 = box
    h = max(0, y2 - y1)
    w = max(0, x2 - x1)
    
    err = False
    if h == 0 or w == 0:
        err = True
    return (h * w), (h, w), (err)
    
def iou(a, b):
    a = np.asarray(a, dtype=np.float64) 
    b = np.asarray(b, dtype=np.float64) 
    
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("Прямоугольник не соответствует формату значений => (x1, y1, x2, y3)")

    S_a, (h, w), err = area(a)
    if err:
       raise ValueError(f"Размеры прямоугольника должны быть > 0. Получено: h={h}, w={w}") 

    S_b, (h, w), err = area(b)
    if err:
       raise ValueError(f"Размеры прямоугольника должны быть > 0. Получено: h={h}, w={w}") 

    x1_a, y1_a, x2_a, y2_a = a
    x1_b, y1_b, x2_b, y2_b = b

    I = [max(x1_a, x1_b), max(y1_a, y1_b), min(x2_a, x2_b), min(y2_a, y2_b)]
    
    S_i, (h, w), err = area(I)
    if err:
        return 0.0   

    iou = S_i / ((S_a + S_b) - S_i)  
    return iou

def print_iou(i):
    if i != 0.0:
        print("IoU =", i)
    else:
        print(f"IoU = {i} (пересечения нет)")


A = (20, 20, 100, 80)
B = (60, 40, 120, 100)
print_iou((iou(A, B)))

A = (10, 10, 50, 50)
B = (10, 10, 50, 50)
print_iou((iou(A, B)))

A = (10, 10, 30, 30)
B = (40, 15, 60, 35)
print_iou((iou(A, B)))

A = (10, 10, 30, 30)
B = (30, 10, 50, 30)
i = iou(A, B)
print_iou((iou(A, B)))

A = (10, 10, 100, 100)
B = (30, 30, 50, 50)
i = iou(A, B)
print_iou((iou(A, B)))
