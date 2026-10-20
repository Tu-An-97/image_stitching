"""Section 3 - component detection on the stitched image (template matching + NMS)."""
import cv2
import numpy as np


def detect_components(img_bgr, template_bgr, thr=0.8, rotations=(0, 90), nms_iou=0.3):
    """Normalised cross-correlation template matching; 0201 parts can be rotated 90 deg."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tpl = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    boxes, scores = [], []
    for rot in rotations:
        t = np.rot90(tpl, rot // 90).copy()
        th, tw = t.shape
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= thr)
        boxes += [[int(x), int(y), tw, th] for x, y in zip(xs, ys)]
        scores += res[ys, xs].tolist()
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, thr, nms_iou)
    return [(*boxes[i], scores[i]) for i in np.array(keep).flatten()]


def precision(tp, fp):
    return tp / (tp + fp) if tp + fp else 0.0


def draw_boxes(img, boxes, color=(0, 0, 255)):
    out = img.copy()
    for x, y, w, h, _ in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out
