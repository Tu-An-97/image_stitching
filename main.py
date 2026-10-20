"""SFFA-TK pipeline (Wu et al., JETTA 2024, 40:669-678).

Examples:
  # 1) heatmap + dense regions of the full-board image
  python -m sffa_tk.main heatmap --board board.jpg --out out
  # 2) stitch two HD shots of a dense region (Table 1: compare all / auto / class i)
  python -m sffa_tk.main stitch --a A.jpg --b B.jpg --out out
  python -m sffa_tk.main stitch --a A.jpg --b B.jpg --no-kmeans --out out
  python -m sffa_tk.main stitch --a A.jpg --b B.jpg --cluster 2 --out out
  # 3) stitch a whole folder of tiles into one image: see panorama.py
  # 4) count components on the stitched image (Table 2)
  python -m sffa_tk.main detect --img out/stitched.jpg --template r0201.png --out out
"""
import argparse
import os
import time

import cv2

from . import detect, heatmap, stitch


def cmd_heatmap(a):
    img = cv2.imread(a.board)
    heat, binary, _ = heatmap.build_heatmap(img)
    boxes, _, th = heatmap.dense_regions(heat, grid=a.grid)
    vis = heatmap.colorize(img, heat)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 3)
    cv2.imwrite(os.path.join(a.out, "binary.jpg"), binary)
    cv2.imwrite(os.path.join(a.out, "heatmap.jpg"), vis)
    print(f"Th = {th:.4f}; dense regions (x, y, w, h): {boxes}")


def cmd_stitch(a):
    from .matcher import SFFATK  # torch import is slow, only load when needed

    img_a, img_b = cv2.imread(a.a), cv2.imread(a.b)
    g_a, g_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    m = SFFATK(pretrained=a.weights, top_k=a.top_k, n_clusters=a.k,
               cluster="auto" if a.cluster is None else a.cluster, use_kmeans=not a.no_kmeans)
    m.match(g_a, g_b, a.max_side)  # warm-up so timing excludes model init

    t = time.perf_counter()
    k0, k1, _ = m.match(g_a, g_b, a.max_side)
    scale = max(1.0, max(g_a.shape) / a.max_side)  # threshold is given at matcher resolution
    H, inl = stitch.homography(k0, k1, ransac_thr=a.ransac_thr * scale, model=a.model)
    if H is None:
        raise SystemExit(f"only {len(k0)} matches - cannot estimate homography")
    pano = stitch.warp_and_blend(img_a, img_b, H)
    dt = time.perf_counter() - t

    last = m.filter.last
    n_kept = len(k0)
    print(f"SFFA-TK: candidates={len(last['pts0'])} kept={n_kept} "
          f"inliers={int(inl.sum())} chosen_cluster={last['chosen']} time={dt:.2f}s")
    if last["labels"] is not None:
        for c in range(a.k):
            print(f"  class {c}: {(last['labels'] == c).sum()} points")
        cv2.imwrite(os.path.join(a.out, "clusters.jpg"),
                    stitch.draw_clusters(img_a, last["pts0"], last["labels"], last["chosen"], m.scale0))
    cv2.imwrite(os.path.join(a.out, "matches.jpg"), stitch.draw_matches(img_a, img_b, k0, k1, inl.ravel()))
    cv2.imwrite(os.path.join(a.out, "stitched.jpg"), pano)


def cmd_detect(a):
    img, tpl = cv2.imread(a.img), cv2.imread(a.template)
    boxes = detect.detect_components(img, tpl, thr=a.thr)
    cv2.imwrite(os.path.join(a.out, "detections.jpg"), detect.draw_boxes(img, boxes))
    print(f"{len(boxes)} detection boxes (count TP/FP manually, then precision = TP / (TP + FP))")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("heatmap")
    h.add_argument("--board", required=True)
    h.add_argument("--grid", type=int, default=100)

    s = sub.add_parser("stitch")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--weights", default="outdoor", choices=["outdoor", "indoor", "indoor_new"])
    s.add_argument("--top-k", type=int, default=200)
    s.add_argument("--k", type=int, default=4, help="number of K-means clusters")
    s.add_argument("--cluster", type=int, default=None, help="force a cluster id (default: auto)")
    s.add_argument("--no-kmeans", action="store_true", help="use all top-K points (Table 1 baseline)")
    s.add_argument("--max-side", type=int, default=840)
    s.add_argument("--model", default="similarity", choices=["similarity", "affine", "homography"])
    s.add_argument("--ransac-thr", type=float, default=3.0, help="RANSAC threshold in px at --max-side resolution")

    d = sub.add_parser("detect")
    d.add_argument("--img", required=True)
    d.add_argument("--template", required=True)
    d.add_argument("--thr", type=float, default=0.8)

    for sp in (h, s, d):
        sp.add_argument("--out", default="sffa_out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"heatmap": cmd_heatmap, "stitch": cmd_stitch, "detect": cmd_detect}[a.cmd](a)


if __name__ == "__main__":
    main()
