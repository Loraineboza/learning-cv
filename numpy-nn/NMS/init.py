import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if boxes.shape != (scores.shape[0], 4):
        raise ValueError(f"Shape mismatch: boxes {boxes.shape}, scores {scores.shape}")

    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area_i = np.maximum(0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0, boxes[i, 3] - boxes[i, 1])
        area_j = np.maximum(0, boxes[order[1:], 2] - boxes[order[1:], 0]) * np.maximum(0, boxes[order[1:], 3] - boxes[order[1:], 1])

        iou = inter / (area_i + area_j - inter + 1e-7)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=np.int64)   
