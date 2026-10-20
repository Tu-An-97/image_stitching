"""Section 2.2 - component-density heatmap and dense-region selection."""
import cv2
import numpy as np


def build_heatmap(img_bgr, block_size=51, c=5, small_area=50 * 50, min_area=16,
                  median_ksize=5, gauss_sigma=25):
    """Adaptive threshold -> contours -> weighted fill -> denoise + Gaussian smoothing.

    Contours with area < 50x50 px get weight 1, larger ones 0.5 (favours small parts).
    Returns (heatmap float32 normalised to [0, 1], binary image, contours).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heat = np.zeros(gray.shape, np.float32)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:  # pure noise
            continue
        weight = 1.0 if area < small_area else 0.5
        x, y, w, h = cv2.boundingRect(cnt)
        roi = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi, [cnt - [x, y]], -1, 1, thickness=-1)
        heat[y:y + h, x:x + w] += roi * weight

    heat = cv2.medianBlur(heat, median_ksize)
    heat = cv2.GaussianBlur(heat, (0, 0), gauss_sigma)
    if heat.max() > 0:
        heat /= heat.max()
    return heat, binary, contours


def dense_regions(heat, grid=100, k_sigma=0.5):
    """Score every grid x grid cell; cells with score > Th = mean(h) + 0.5*sigma are dense (eq. 1),
    where mean(h) and sigma are taken over the whole heatmap h, as in the paper.

    Adjacent dense cells are merged; returns (boxes [(x, y, w, h)], cell_scores, Th).
    """
    H, W = heat.shape
    gh, gw = int(np.ceil(H / grid)), int(np.ceil(W / grid))
    scores = np.zeros((gh, gw), np.float32)
    for r in range(gh):
        for col in range(gw):
            scores[r, col] = heat[r * grid:(r + 1) * grid, col * grid:(col + 1) * grid].mean()

    th = heat.mean() + k_sigma * heat.std()
    dense = (scores > th).astype(np.uint8)
    n, labels = cv2.connectedComponents(dense, connectivity=8)
    boxes = []
    for lbl in range(1, n):
        rs, cs = np.where(labels == lbl)
        x0, y0 = cs.min() * grid, rs.min() * grid
        x1, y1 = min((cs.max() + 1) * grid, W), min((rs.max() + 1) * grid, H)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes, scores, th


def colorize(img_bgr, heat, alpha=0.5):
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)
