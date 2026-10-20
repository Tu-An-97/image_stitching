"""Split an image into an overlapping grid of tiles to use as stitching test input.

Each tile can optionally get the distortions used in the paper's robustness test
(small rotation + blur) and a brightness change. Ground truth is saved to
meta.json: for every tile, the 3x3 homography H mapping tile pixels -> original
image pixels, so stitching results can be checked against it.

Examples:
  # 1x2 split, 30% overlap (the A/B pair used by `main.py stitch`)
  python -m sffa_tk.split_overlap --img sffa_tk/data/origin_img/source_20260625_134850.png --cols 2 --overlap 0.3
  # 2x2 split, 25% overlap, B..D rotated up to 1 deg and blurred as in Fig. 6
  python -m sffa_tk.split_overlap --img ... --rows 2 --cols 2 --overlap 0.25 --rotate 1 --blur 3
"""
import argparse
import json
import os

import cv2
import numpy as np


def tile_starts(length, n, overlap):
    """Start offsets of n tiles of equal size covering `length` with the given overlap ratio."""
    if n == 1:
        return [0], length
    size = int(round(length / (n - (n - 1) * overlap)))
    step = (length - size) / (n - 1)
    return [int(round(i * step)) for i in range(n)], size


def distort(tile, angle, blur, gain, rng):
    """Rotate about the tile centre, blur, change brightness. Returns (tile, 3x3 M tile->distorted)."""
    h, w = tile.shape[:2]
    M = np.eye(3)
    if angle:
        a = rng.uniform(-angle, angle)
        M[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        tile = cv2.warpAffine(tile, M[:2], (w, h), borderMode=cv2.BORDER_REFLECT)
    if blur:
        k = blur | 1  # kernel must be odd
        tile = cv2.GaussianBlur(tile, (k, k), 0)
    if gain:
        g = rng.uniform(1 - gain, 1 + gain)
        tile = cv2.convertScaleAbs(tile, alpha=g, beta=0)
    return tile, M


def split(img, rows, cols, overlap, rotate=0.0, blur=0, gain=0.0, keep_first=True, seed=0):
    H, W = img.shape[:2]
    ys, th = tile_starts(H, rows, overlap)
    xs, tw = tile_starts(W, cols, overlap)
    rng = np.random.default_rng(seed)
    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tile = img[y:y + th, x:x + tw].copy()
            M = np.eye(3)
            if not (keep_first and r == 0 and c == 0):  # tile 0 stays clean as the reference
                tile, M = distort(tile, rotate, blur, gain, rng)
            # distorted tile px -> clean tile px -> original px
            H_gt = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64) @ np.linalg.inv(M)
            tiles.append({"row": r, "col": c, "x": x, "y": y, "w": tw, "h": th,
                          "H_tile_to_orig": H_gt.tolist(), "image": tile})
    return tiles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img", required=True)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--overlap", type=float, default=0.3, help="overlap ratio between neighbours (0-0.9)")
    p.add_argument("--rotate", type=float, default=0.0, help="max random rotation in degrees")
    p.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (0 = off)")
    p.add_argument("--gain", type=float, default=0.0, help="max brightness change, e.g. 0.1 = +-10%%")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: sffa_tk/data/split/<image name>")
    a = p.parse_args()

    if not 0 <= a.overlap < 0.9:
        raise SystemExit("--overlap must be in [0, 0.9)")
    img = cv2.imread(a.img)
    if img is None:
        raise SystemExit(f"cannot read {a.img}")

    name = os.path.splitext(os.path.basename(a.img))[0]
    out = a.out or os.path.join(os.path.dirname(__file__), "data", "split", name)
    os.makedirs(out, exist_ok=True)

    tiles = split(img, a.rows, a.cols, a.overlap, a.rotate, a.blur, a.gain, seed=a.seed)
    preview = img.copy()
    meta = {"source": os.path.abspath(a.img), "size": [img.shape[1], img.shape[0]],
            "rows": a.rows, "cols": a.cols, "overlap": a.overlap,
            "rotate": a.rotate, "blur": a.blur, "gain": a.gain, "seed": a.seed, "tiles": []}
    for t in tiles:
        fname = f"tile_r{t['row']}_c{t['col']}.png"
        cv2.imwrite(os.path.join(out, fname), t.pop("image"))
        meta["tiles"].append({"file": fname, **t})
        cv2.rectangle(preview, (t["x"], t["y"]), (t["x"] + t["w"] - 1, t["y"] + t["h"] - 1),
                      (0, 255, 255), 4)
        cv2.putText(preview, f"r{t['row']}c{t['col']}", (t["x"] + 15, t["y"] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(out, "preview_grid.jpg"), preview)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "layout.txt"), "w", encoding="utf-8") as f:  # input for grid_panorama --order
        for r in range(a.rows):
            f.write(" ".join(f"tile_r{r}_c{c}.png" for c in range(a.cols)) + "\n")

    t0 = meta["tiles"][0]
    print(f"{len(tiles)} tiles of {t0['w']}x{t0['h']} px -> {out}")
    for t in meta["tiles"]:
        print(f"  {t['file']}: x={t['x']} y={t['y']}")


if __name__ == "__main__":
    main()
