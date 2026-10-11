#iou обрабатывается на неограниченое кол-во maxtrix[..., 4] 

import numpy as np

def area(box):
    x1, y1, x2, y2 = box
    h = max(0, y2 - y1)
    w = max(0, x2 - x1)
    return h * w


def iou(a, b_s) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b_s = np.asarray(b_s, np.float64)
    eps = 1.e-7
    
    if a.shape != (4,) or b_s.shape[-1] != 4:
        raise ValueError("Ошибка размерностей прямоугольников")

    x_left = np.maximum(a[0], b_s[:, 0])
    y_top = np.maximum(a[1], b_s[:, 1])
    x_right = np.minimum(a[2], b_s[:, 2])
    y_bottom = np.minimum(a[3], b_s[:, 3])
    
    w_i = np.maximum(0, x_right - x_left)
    h_i = np.maximum(0, y_bottom - y_top)
    
    s_i = w_i * h_i
    
    s_a = area(a)
    
    x1 = b_s[:, 0]
    y1 = b_s[:, 1]
    x2 = b_s[:, 2]
    y2 = b_s[:, 3]
    
    w_b = np.maximum(0, x2 - x1)
    h_b = np.maximum(0, y2 - y1)

    s_b = w_b * h_b
    
    
    _ious = s_i / (s_a + s_b - s_i + eps)
    
    return _ious

if __name__ == "__main__":
    box_a = np.array([10, 10, 50, 50])

    boxes_b = np.array([
        [20, 20, 60, 60],
        [10, 10, 50, 50],
        [60, 60, 90, 90],
        [ 0,  0, 30, 30],
        [10, 40, 20, 15],
        [40, 30, 20, 5],
    ])

    ious_array = iou(box_a, boxes_b)
    print(ious_array)
