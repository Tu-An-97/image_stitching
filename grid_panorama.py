"""Stitch frames whose grid layout (R x C, row-major order) is known, with SFFA-TK.

Compared to panorama.py (which matches every pair and guesses the layout):
1. Only grid neighbours are matched (right and down), on the facing strips of the two frames,
   so non-overlapping frames can never be linked and LoFTR sees the overlap at higher resolution.
2. Each pairwise transform must point in the expected direction (right / down), which rejects
   false matches on repetitive parts (DIMM slots, pin rows, ...).
3. The scan step is (nearly) constant: pairs far from the median right/down step - typically a
   match on the next identical DRAM chip - are re-fitted with only the matches that agree with
   the step, or dropped (--step-tol). Frames left without any pair are placed from the step.
4. All frames are placed at once by least squares over every accepted pair (translation for one
   camera on a scan stage - default; similarity for several cameras), so the error is spread over
   the grid instead of accumulating along a chain; pairs that still
   disagree with the joint solution (--max-residual) are dropped and the solve is repeated.

Layout input, either:
  --order layout.txt   one grid row per line, file names separated by spaces/commas, "-" = empty cell
                       (relative names are resolved against --dir, else the file's folder)
  --dir/--imgs + --grid RxC   files in row-major order (sorted names for --dir); --snake if every
                       other row was captured right-to-left

Examples:
  python -m sffa_tk.grid_panorama --dir frames --grid 5x5
  python -m sffa_tk.grid_panorama --order layout.txt --dir frames --out pano_out
"""
import argparse
import glob
import json
import os
import re
import time

import cv2
import numpy as np

from . import stitch
from .panorama import compose, plausible, spanning_tree

IMG_EXT = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


def load_grid(a):
    """Returns a list of rows, each a list of file paths (None = empty cell)."""
    if a.order:
        base = a.dir or os.path.dirname(os.path.abspath(a.order))
        grid = []
        with open(a.order, encoding="utf-8-sig") as f:  # -sig: Notepad/PowerShell write a BOM
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names = [t for t in re.split(r"[,\s]+", line) if t]
                grid.append([None if t == "-" else os.path.join(base, t) for t in names])
        return grid

    files = a.imgs or sorted(p for ext in IMG_EXT for p in glob.glob(os.path.join(a.dir, f"*.{ext}"))
                             if "preview" not in os.path.basename(p))
    if not a.grid:
        raise SystemExit("--grid RxC is required with --dir/--imgs")
    R, C = map(int, a.grid.lower().split("x"))
    if len(files) != R * C:
        raise SystemExit(f"--grid {R}x{C} needs {R * C} images, got {len(files)}")
    grid = [files[r * C:(r + 1) * C] for r in range(R)]
    return [row[::-1] if a.snake and r % 2 else row for r, row in enumerate(grid)]


def facing_strip(img, side, frac):
    """Crop the part of `img` on `side` (left/right/top/bottom); returns (crop, (x_off, y_off))."""
    h, w = img.shape[:2]
    if side == "right":
        x0 = int(w * (1 - frac))
        return img[:, x0:], (x0, 0)
    if side == "left":
        return img[:, :int(round(w * frac))], (0, 0)
    if side == "bottom":
        y0 = int(h * (1 - frac))
        return img[y0:], (0, y0)
    return img[:int(round(h * frac))], (0, 0)


def direction_ok(H, shape_i, shape_j, direction, max_offaxis):
    """Frame j must sit to the right of (or below) frame i, roughly on the same row (column)."""
    hi, wi = shape_i[:2]
    hj, wj = shape_j[:2]
    cj = cv2.perspectiveTransform(np.float32([[[wj / 2, hj / 2]]]), H)[0, 0]
    dx, dy = cj[0] - wi / 2, cj[1] - hi / 2
    if direction == "right":
        return dx > 0.2 * wi and abs(dy) < max_offaxis * hi
    return dy > 0.2 * hi and abs(dx) < max_offaxis * wi


def pair_model(model):
    """Pairwise RANSAC model; a pure translation is fitted as a similarity (it still rejects outliers)."""
    return "similarity" if model == "translation" else model


def match_neighbours(cells, grays, match, a):
    """cells: {(r, c): idx}. Returns {(i, j): dict(H, n, p_i, p_j)} for accepted neighbour pairs."""
    edges = {}
    for (r, c), i in sorted(cells.items()):
        for direction, (rr, cc), (side_i, side_j) in (("right", (r, c + 1), ("right", "left")),
                                                      ("down", (r + 1, c), ("bottom", "top"))):
            j = cells.get((rr, cc))
            if j is None:
                continue
            t = time.perf_counter()
            si, oi = facing_strip(grays[i], side_i, a.strip)
            sj, oj = facing_strip(grays[j], side_j, a.strip)
            # same pixel budget as a full frame at --max-side: a thinner strip gets a finer resolution
            side = int(a.max_side / np.sqrt(a.strip)) // 8 * 8
            k0, k1 = match(si, sj, side, f"{r}_{c}_{direction}")
            k0, k1 = k0 + np.float32(oi), k1 + np.float32(oj)
            scale = max(1.0, max(si.shape) / side)
            H, inl = stitch.homography(k0, k1, ransac_thr=a.ransac_thr * scale, model=pair_model(a.model))
            n = 0 if inl is None else int(inl.sum())
            why = ""
            if H is None or n < a.min_inliers:
                why = "few inliers"
            elif not plausible(H, grays[j].shape):
                why = "degenerate"
            elif not direction_ok(H, grays[i].shape, grays[j].shape, direction, a.max_offaxis):
                why = "wrong direction"
            print(f"  ({r},{c})->{direction:5s} ({rr},{cc}): matches={len(k0):4d} inliers={n:4d} "
                  f"{'OK' if not why else 'skip: ' + why}  {time.perf_counter() - t:.1f}s")
            # raw matches are kept so a pair can be re-fitted later around the grid step
            edges[(i, j)] = {"dir": direction, "k0": k0, "k1": k1, "ok": not why, "scale": scale}
            if not why:
                keep = inl.ravel().astype(bool)
                edges[(i, j)].update(H=H, n=n, p_i=k0[keep], p_j=k1[keep])
    return edges


def offset(H, shape_j):
    """Where the centre of frame j lands in frame i, relative to j's own centre (= the grid step)."""
    h, w = shape_j[:2]
    c = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
    return c - np.float32([w / 2, h / 2])


def enforce_grid_step(edges, grays, a):
    """Frames of a scan are taken at a (nearly) constant step, so every right pair should have about
    the same offset, and so should every down pair. A pair far from the median step usually matched a
    repeated part (the next DRAM chip, pin row, ...): it is re-fitted using only the matches that agree
    with the median step, or dropped. Pairs rejected earlier get the same second chance."""
    steps = {}
    for d in ("right", "down"):
        good = [offset(e["H"], grays[j].shape) for (i, j), e in edges.items() if e["ok"] and e["dir"] == d]
        if len(good) >= 3:  # too few pairs for a reliable median -> no check in this direction
            steps[d] = np.median(np.array(good), axis=0)
            print(f"  grid step {d:5s}: dx={steps[d][0]:8.1f} dy={steps[d][1]:8.1f}  ({len(good)} pairs)")
    for (i, j), e in edges.items():
        if e["dir"] not in steps:
            continue
        step = steps[e["dir"]]
        tol = a.step_tol * max(grays[j].shape)
        if e["ok"] and np.linalg.norm(offset(e["H"], grays[j].shape) - step) <= tol:
            continue
        # matches whose displacement agrees with the median step (rotation between frames is small)
        d = e["k0"] - e["k1"]
        sel = np.linalg.norm(d - step, axis=1) <= tol
        H, inl = (None, None)
        if sel.sum() >= a.min_inliers:
            H, inl = stitch.homography(e["k0"][sel], e["k1"][sel], ransac_thr=a.ransac_thr * e["scale"],
                                       model=pair_model(a.model))
        n = 0 if inl is None else int(inl.sum())
        was = "re-fitted" if e["ok"] else "rescued"
        if H is not None and n >= a.min_inliers and plausible(H, grays[j].shape) and \
                np.linalg.norm(offset(H, grays[j].shape) - step) <= tol:
            keep = inl.ravel().astype(bool)
            e.update(ok=True, H=H, n=n, p_i=e["k0"][sel][keep], p_j=e["k1"][sel][keep], note=was)
        else:
            e.update(ok=False, note="off grid step")
        print(f"  pair {i}-{j} ({e['dir']}): {e['note']} ({n} inliers)")
    return steps


def place_by_step(H_to_ref, cells, steps):
    """Frames with no usable pair (blank background, ...) are placed from a placed neighbour + grid step."""
    pos = {i: rc for rc, i in cells.items()}
    guessed, grew = [], True
    while grew:
        grew = False
        for i in set(pos) - set(H_to_ref):
            r, c = pos[i]
            for (dr, dc), d, sign in (((0, -1), "right", 1), ((0, 1), "right", -1),
                                      ((-1, 0), "down", 1), ((1, 0), "down", -1)):
                n = cells.get((r + dr, c + dc))
                if n in H_to_ref and d in steps:
                    T = np.eye(3)
                    T[:2, 2] = sign * steps[d]
                    H_to_ref[i] = H_to_ref[n] @ T
                    guessed.append(pos[i])
                    grew = True
                    break
    return sorted(guessed)


def global_similarity(n_img, edges, ref, reachable, max_pts=300, translation=False):
    """Least-squares similarity per image (x' = a x - b y + tx, y' = b x + a y + ty) into ref's frame.
    translation=True fixes a=1, b=0 (same camera moved by a stage): scale/rotation cannot drift
    across the grid, which a similarity per frame does on a 5x5 scan."""
    nodes = sorted(reachable)
    npar = 2 if translation else 4
    col = {v: npar * k for k, v in enumerate(nodes)}
    rows_A, rows_b = [], []
    rng = np.random.default_rng(0)
    for (i, j), e in edges.items():
        if i not in col or j not in col:
            continue
        sel = rng.permutation(len(e["p_i"]))[:max_pts]
        w = 1.0 / np.sqrt(len(sel))  # every pair counts about the same, whatever its match count
        for (xi, yi), (xj, yj) in zip(e["p_i"][sel], e["p_j"][sel]):
            if translation:  # xi + txi = xj + txj
                rows = (([1, 0], [1, 0], xj - xi), ([0, 1], [0, 1], yj - yi))
            else:
                rows = (([xi, -yi, 1, 0], [xj, -yj, 1, 0], 0.0), ([yi, xi, 0, 1], [yj, xj, 0, 1], 0.0))
            for row_i, row_j, rhs in rows:
                r = np.zeros(npar * len(nodes))
                r[col[i]:col[i] + npar] = row_i
                r[col[j]:col[j] + npar] -= np.array(row_j, float)
                rows_A.append(r * w)
                rows_b.append(rhs * w)
    pin = [0.0, 0.0] if translation else [1.0, 0.0, 0.0, 0.0]
    for p, v in enumerate(pin):  # pin the reference frame to identity
        r = np.zeros(npar * len(nodes))
        r[col[ref] + p] = 1e3
        rows_A.append(r)
        rows_b.append(1e3 * v)
    x = np.linalg.lstsq(np.array(rows_A), np.array(rows_b), rcond=None)[0]
    H_to_ref = {}
    for v in nodes:
        a_, b_, tx, ty = (1.0, 0.0, *x[col[v]:col[v] + 2]) if translation else x[col[v]:col[v] + 4]
        H_to_ref[v] = np.array([[a_, -b_, tx], [b_, a_, ty], [0, 0, 1]])
    return H_to_ref


def residuals(edges, H_to_ref):
    out = {}
    for (i, j), e in edges.items():
        if i in H_to_ref and j in H_to_ref:
            pi = cv2.perspectiveTransform(e["p_i"][None].astype(np.float64), H_to_ref[i])[0]
            pj = cv2.perspectiveTransform(e["p_j"][None].astype(np.float64), H_to_ref[j])[0]
            out[(i, j)] = float(np.linalg.norm(pi - pj, axis=1).mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order", help="layout file (one grid row per line)")
    p.add_argument("--dir")
    p.add_argument("--imgs", nargs="+")
    p.add_argument("--grid", help="RxC, e.g. 5x5 (with --dir/--imgs)")
    p.add_argument("--snake", action="store_true", help="odd rows were captured right-to-left")
    p.add_argument("--strip", type=float, default=0.5, help="fraction of each frame matched on the facing side")
    p.add_argument("--max-offaxis", type=float, default=0.3, help="max sideways drift, fraction of frame size")
    p.add_argument("--step-tol", type=float, default=0.05,
                   help="max deviation of a pair from the median grid step, fraction of frame size (0 = off)")
    p.add_argument("--max-residual", type=float, default=10.0,
                   help="pairs above this residual (px, and 3x the median) after placement are dropped")
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--ransac-thr", type=float, default=3.0, help="RANSAC threshold in px at --max-side resolution")
    p.add_argument("--model", default="translation", choices=["translation", "similarity", "affine", "homography"],
                   help="translation (one camera on a scan stage) / similarity (several cameras) -> global least "
                        "squares; affine/homography -> chained spanning tree")
    p.add_argument("--no-cache", action="store_true", help="ignore <out>/matches_cache.npz and match again")
    p.add_argument("--ref", default=None, help="reference cell r,c (default: grid centre)")
    p.add_argument("--weights", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--no-kmeans", action="store_true")
    p.add_argument("--max-side", type=int, default=840)
    p.add_argument("--out", default=None, help="default: <dir>/grid_panorama")
    a = p.parse_args()
    if not (a.order or a.dir or a.imgs):
        raise SystemExit("give --order, --dir or --imgs")

    grid = load_grid(a)
    cells, files = {}, []
    for r, row in enumerate(grid):
        for c, f in enumerate(row):
            if f is not None:
                cells[(r, c)] = len(files)
                files.append(f)
    out = a.out or os.path.join(a.dir or os.path.dirname(files[0]), "grid_panorama")
    os.makedirs(out, exist_ok=True)

    images = [cv2.imread(f) for f in files]
    for f, im in zip(files, images):
        if im is None:
            raise SystemExit(f"cannot read {f}")
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    print(f"grid {len(grid)}x{max(len(r) for r in grid)}, {len(files)} frames")
    for (r, c), i in sorted(cells.items()):
        print(f"  ({r},{c}) {os.path.basename(files[i])} {images[i].shape[1]}x{images[i].shape[0]}")

    if a.ref:
        ref_cell = tuple(map(int, a.ref.split(",")))
    else:
        R, C = len(grid), max(len(r) for r in grid)
        ref_cell = min(cells, key=lambda rc: (rc[0] - (R - 1) / 2) ** 2 + (rc[1] - (C - 1) / 2) ** 2)
    ref = cells[ref_cell]

    # LoFTR is the slow part: raw matches are cached so RANSAC / placement options can be re-tuned
    # without matching again. The cache is only valid for the same frames and matcher settings.
    tag = json.dumps([files, a.strip, a.max_side, a.weights, a.top_k, a.k, a.no_kmeans])
    cache_path = os.path.join(out, "matches_cache.npz")
    cache = {}
    if os.path.exists(cache_path) and not a.no_cache:
        z = np.load(cache_path)
        if str(z["tag"]) == tag:
            cache = {k: z[k] for k in z.files if k != "tag"}
            print(f"using cached matches from {cache_path}")
    m = None

    def match(g0, g1, side, key):
        nonlocal m
        if key + "_0" not in cache:
            if m is None:
                from .matcher import SFFATK  # torch import is slow, only load when needed
                m = SFFATK(pretrained=a.weights, top_k=a.top_k, n_clusters=a.k, use_kmeans=not a.no_kmeans)
            cache[key + "_0"], cache[key + "_1"] = m.match(g0, g1, side)[:2]
        return cache[key + "_0"], cache[key + "_1"]

    t = time.perf_counter()
    all_edges = match_neighbours(cells, grays, match, a)
    if m is not None:
        np.savez(cache_path, tag=tag, **cache)
    steps = enforce_grid_step(all_edges, grays, a) if a.step_tol > 0 else {}
    edges = {k: e for k, e in all_edges.items() if e["ok"]}
    pos = {i: rc for rc, i in cells.items()}

    while True:
        # frames connected to the reference through accepted pairs (also the fallback placement)
        tree_edges = {k: (e["H"], e["n"]) for k, e in edges.items()}
        _, H_tree = spanning_tree(len(files), tree_edges, ref)
        H_to_ref = global_similarity(len(files), edges, ref, H_tree, translation=a.model == "translation") \
            if a.model in ("translation", "similarity") else H_tree
        # a pair that still disagrees strongly with the joint solution is a false match: drop and re-solve
        res = residuals(edges, H_to_ref)
        worst = max(res, key=res.get, default=None)
        if worst is None or res[worst] <= max(a.max_residual, 3 * np.median(list(res.values()))):
            break
        print(f"  drop pair {pos[worst[0]]}-{pos[worst[1]]}: residual {res[worst]:.1f}px")
        all_edges[worst].update(ok=False, note=f"residual {res[worst]:.1f}px")
        del edges[worst]

    guessed = place_by_step(H_to_ref, cells, steps)
    if guessed:
        print(f"WARNING: cells {guessed} have no reliable match, placed from the grid step only")
    missing = sorted(pos[i] for i in set(range(len(files))) - set(H_to_ref))
    if missing:
        print(f"WARNING: cells {missing} have no reliable overlap with the reference and are left out")
    pano, _ = compose(images, H_to_ref)
    print(f"reference = {ref_cell}, panorama {pano.shape[1]}x{pano.shape[0]}, total {time.perf_counter() - t:.1f}s")

    res = residuals(edges, H_to_ref)
    print("mean residual per pair after placement (px):")
    for (i, j), v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {pos[i]} - {pos[j]}: {v:6.2f}")

    cv2.imwrite(os.path.join(out, "panorama.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": ref_cell, "missing": missing, "placed_by_step": guessed,
                   "grid_step": {d: s.tolist() for d, s in steps.items()},
                   "H_to_ref": {f"{pos[i][0]},{pos[i][1]}": H.tolist() for i, H in H_to_ref.items()},
                   "pairs": [{"a": pos[i], "b": pos[j], "used": e["ok"], "note": e.get("note", ""),
                              "inliers": e.get("n", 0), "residual_px": res.get((i, j))}
                             for (i, j), e in all_edges.items()]}, f, indent=2)
    print(f"saved -> {os.path.join(out, 'panorama.jpg')}")


if __name__ == "__main__":
    main()
