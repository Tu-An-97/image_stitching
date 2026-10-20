"""Transform estimation + warping/blending and match/cluster visualisation."""
import cv2
import numpy as np


def homography(k0, k1, ransac_thr=3.0, model="homography"):
    """3x3 transform mapping image B (k1) onto image A (k0).

    model: "homography" (8 DoF), "affine" (6 DoF) or "similarity" (rotation + scale + shift, 4 DoF).
    Fewer DoF extrapolate much better when the matches cover only part of the overlap
    (e.g. one K-means cluster), and fit a camera looking straight down at a flat PCB.
    """
    if len(k0) < 4:
        return None, None
    if model == "homography":
        return cv2.findHomography(k1, k0, cv2.USAC_MAGSAC, ransac_thr, maxIters=10000, confidence=0.999)
    estimate = cv2.estimateAffine2D if model == "affine" else cv2.estimateAffinePartial2D
    A, inl = estimate(k1, k0, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr, maxIters=10000, confidence=0.999)
    if A is None:
        return None, None
    return np.vstack([A, [0, 0, 1]]), inl


def warp_and_blend(img_a, img_b, H):
    """Warp B into A's frame on a canvas large enough for both, feather-blend the overlap."""
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    corners_b = cv2.perspectiveTransform(np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]])[None], H)[0]
    all_pts = np.vstack([corners_b, [[0, 0], [wa, 0], [wa, ha], [0, ha]]])
    x0, y0 = np.floor(all_pts.min(0)).astype(int)
    x1, y1 = np.ceil(all_pts.max(0)).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (x1 - x0, y1 - y0)

    wa_img = cv2.warpPerspective(img_a, T, size)
    wb_img = cv2.warpPerspective(img_b, T @ H, size)
    ma = cv2.warpPerspective(np.ones((ha, wa), np.float32), T, size)
    mb = cv2.warpPerspective(np.ones((hb, wb), np.float32), T @ H, size)
    # distance-to-border weights give a seamless feathered seam
    da = cv2.distanceTransform((ma > 0).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((mb > 0).astype(np.uint8), cv2.DIST_L2, 5)
    w = (da / np.maximum(da + db, 1e-6))[..., None]
    out = wa_img.astype(np.float32) * w + wb_img.astype(np.float32) * (1 - w)
    return out.clip(0, 255).astype(np.uint8)


def draw_matches(img_a, img_b, k0, k1, inliers=None, max_lines=300):
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((h, img_a.shape[1] + img_b.shape[1], 3), np.uint8)
    canvas[:img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[:img_b.shape[0], img_a.shape[1]:] = img_b
    idx = np.random.default_rng(0).permutation(len(k0))[:max_lines]
    for i in idx:
        ok = inliers is None or inliers[i]
        p, q = tuple(map(int, k0[i])), (int(k1[i][0] + img_a.shape[1]), int(k1[i][1]))
        cv2.line(canvas, p, q, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def draw_clusters(img, pts, labels, chosen, scale):
    palette = [(255, 80, 80), (80, 200, 80), (80, 80, 255), (0, 200, 255), (255, 0, 255), (255, 255, 0)]
    out = img.copy()
    for p, l in zip(pts * scale, labels):
        r = 6 if l == chosen else 3
        cv2.circle(out, tuple(map(int, p)), r, palette[l % len(palette)], -1, cv2.LINE_AA)
    return out
