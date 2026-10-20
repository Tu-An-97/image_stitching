"""Stitch N overlapping images into a single panorama with SFFA-TK.

1. Match every pair with SFFA-TK and estimate a transform per pair.
2. Build a graph weighted by inlier count; pick the best-connected image as reference
   and keep a maximum spanning tree, so every image is chained to the reference
   through its strongest overlaps.
3. Warp all images onto one canvas and feather-blend them.
4. If the folder has meta.json (from split_overlap.py), report the error against
   the ground-truth tile positions.

Examples:
  python -m sffa_tk.panorama --dir sffa_tk/data/split/grid2x2
  python -m sffa_tk.panorama --dir sffa_tk/data/split/grid2x2 --no-kmeans
  python -m sffa_tk.panorama --imgs a.png b.png c.png --out pano_out
"""
import argparse
import glob
import itertools
import json
import os
import time

import cv2
import numpy as np

from . import stitch


def list_images(folder):
    meta_path = os.path.join(folder, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return [os.path.join(folder, t["file"]) for t in meta["tiles"]], meta
    files = sorted(p for ext in ("png", "jpg", "jpeg", "bmp", "tif")
                   for p in glob.glob(os.path.join(folder, f"*.{ext}")) if "preview" not in os.path.basename(p))
    return files, None


def make_matcher(args):
    from .matcher import SFFATK  # torch import is slow, only load when needed
    m = SFFATK(pretrained=args.weights, top_k=args.top_k, n_clusters=args.k, use_kmeans=not args.no_kmeans)
    return lambda g0, g1: m.match(g0, g1, args.max_side)[:2]


def pairwise(grays, match, min_inliers, model, ransac_thr=3.0, max_side=840):
    """Returns {(i, j): (H_j->i, n_inliers)} for every pair that overlaps reliably.

    ransac_thr is in pixels at the matcher resolution, so it is scaled up to original pixels.
    """
    edges = {}
    for i, j in itertools.combinations(range(len(grays)), 2):
        t = time.perf_counter()
        k0, k1 = match(grays[i], grays[j])
        scale = max(1.0, max(grays[i].shape) / max_side)
        H, inl = stitch.homography(k0, k1, ransac_thr=ransac_thr * scale, model=model)
        n = 0 if inl is None else int(inl.sum())
        ok = H is not None and n >= min_inliers and plausible(H, grays[j].shape)
        print(f"  pair ({i},{j}): matches={len(k0):4d} inliers={n:4d} "
              f"{'OK ' if ok else 'skip'} {time.perf_counter() - t:.1f}s")
        if ok:
            edges[(i, j)] = (H, n)
    return edges


def plausible(H, shape, max_scale=4.0):
    """Reject degenerate homographies (flipped, collapsed or exploded warps)."""
    h, w = shape[:2]
    c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    area = cv2.contourArea(c)
    return cv2.isContourConvex(c.astype(np.float32)) and 1 / max_scale < area / (w * h) < max_scale


def spanning_tree(n, edges, ref=None):
    """Maximum spanning tree (Prim) from the reference. Returns (ref, {node: H_node->ref})."""
    weight = np.zeros((n, n))
    for (i, j), (_, w) in edges.items():
        weight[i, j] = weight[j, i] = w
    if ref is None:
        ref = int(weight.sum(1).argmax())
    H_to_ref = {ref: np.eye(3)}
    while True:
        best = None
        for u in H_to_ref:
            for v in range(n):
                if v not in H_to_ref and weight[u, v] > 0 and (best is None or weight[u, v] > best[2]):
                    best = (u, v, weight[u, v])
        if best is None:
            break
        u, v, _ = best
        H_vu = edges[(u, v)][0] if (u, v) in edges else np.linalg.inv(edges[(v, u)][0])
        H_to_ref[v] = H_to_ref[u] @ H_vu
        H_to_ref[v] /= H_to_ref[v][2, 2]
    return ref, H_to_ref


def compose(images, H_to_ref):
    """Warp every image onto a shared canvas and feather-blend by distance to the image border."""
    corners = []
    for i, H in H_to_ref.items():
        h, w = images[i].shape[:2]
        corners.append(cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0])
    corners = np.vstack(corners)
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (x1 - x0, y1 - y0)

    acc = np.zeros((size[1], size[0], 3), np.float32)
    wsum = np.zeros((size[1], size[0]), np.float32)
    for i, H in H_to_ref.items():
        img = images[i]
        h, w = img.shape[:2]
        # weight is highest in the tile centre and fades to 0 at its border
        border = np.ones((h, w), np.uint8)
        border[[0, -1], :] = border[:, [0, -1]] = 0
        weight = cv2.distanceTransform(border, cv2.DIST_L2, 5)
        warped_w = cv2.warpPerspective(weight, T @ H, size)
        acc += cv2.warpPerspective(img, T @ H, size).astype(np.float32) * warped_w[..., None]
        wsum += warped_w
    pano = acc / np.maximum(wsum, 1e-6)[..., None]
    return pano.clip(0, 255).astype(np.uint8), T


def gt_error(meta, ref, H_to_ref, step=50):
    """Mean / max reprojection error (px) of each tile vs. the ground truth from split_overlap.py."""
    G = [np.array(t["H_tile_to_orig"]) for t in meta["tiles"]]
    report = {}
    for i, H in H_to_ref.items():
        w, h = meta["tiles"][i]["w"], meta["tiles"][i]["h"]
        xs, ys = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        pts = np.float32(np.stack([xs.ravel(), ys.ravel()], 1))[None]
        est = cv2.perspectiveTransform(pts, H)[0]
        gt = cv2.perspectiveTransform(pts, np.linalg.inv(G[ref]) @ G[i])[0]
        d = np.linalg.norm(est - gt, axis=1)
        report[i] = (float(d.mean()), float(d.max()))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="folder with the images (uses meta.json order if present)")
    src.add_argument("--imgs", nargs="+")
    p.add_argument("--ref", type=int, default=None, help="reference image index (default: best connected)")
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--ransac-thr", type=float, default=3.0, help="RANSAC threshold in px at --max-side resolution")
    p.add_argument("--model", default="similarity", choices=["similarity", "affine", "homography"],
                   help="pairwise transform; similarity suits a camera looking straight down at a PCB")
    p.add_argument("--weights", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--no-kmeans", action="store_true")
    p.add_argument("--max-side", type=int, default=840)
    p.add_argument("--out", default=None, help="default: <dir>/panorama_<model>")
    a = p.parse_args()

    files, meta = list_images(a.dir) if a.dir else (a.imgs, None)
    if len(files) < 2:
        raise SystemExit("need at least 2 images")
    out = a.out or os.path.join(a.dir or ".", f"panorama_{a.model}")
    os.makedirs(out, exist_ok=True)

    images = [cv2.imread(f) for f in files]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)} {images[i].shape[1]}x{images[i].shape[0]}")

    match = make_matcher(a)
    t = time.perf_counter()
    edges = pairwise(grays, match, a.min_inliers, a.model, a.ransac_thr, a.max_side)
    ref, H_to_ref = spanning_tree(len(images), edges, a.ref)
    missing = sorted(set(range(len(images))) - set(H_to_ref))
    if missing:
        print(f"WARNING: images {missing} have no reliable overlap with the rest and are left out")
    pano, _ = compose(images, H_to_ref)
    print(f"reference = [{ref}], panorama {pano.shape[1]}x{pano.shape[0]}, total {time.perf_counter() - t:.1f}s")

    cv2.imwrite(os.path.join(out, "panorama.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if meta:
        report = gt_error(meta, ref, H_to_ref)
        print("error vs ground truth (px):")
        for i, (mean, mx) in sorted(report.items()):
            print(f"  [{i}] mean={mean:6.2f}  max={mx:6.2f}")
        with open(os.path.join(out, "gt_error.json"), "w", encoding="utf-8") as f:
            json.dump({"reference": ref, "error_px": {files[i]: v for i, v in report.items()}}, f, indent=2)
    print(f"saved -> {os.path.join(out, 'panorama.jpg')}")


if __name__ == "__main__":
    main()
