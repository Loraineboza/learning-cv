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

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        area_i = np.clip(boxes[i, 2] - boxes[i, 0], 0, None) * np.clip(boxes[i, 3] - boxes[i, 1], 0, None)
        area_j = np.clip(boxes[order[1:], 2] - boxes[order[1:], 0], 0, None) * np.clip(boxes[order[1:], 3] - boxes[order[1:], 1], 0, None)

        iou = inter / (area_i + area_j - inter + 1e-7)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=np.int64)   
