from init import nms
import numpy as np

if __name__ == "__main__":
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],
        [60, 60, 90, 90],
        [65, 65, 95, 95],
        [0, 0, 30, 30],
    ])

    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    keep = nms(boxes, scores, iou_threshold=0.5)
    print(keep)   
