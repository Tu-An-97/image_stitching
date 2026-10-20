"""Convert Cognex DS1000 .cdb image databases (deskewed range images) to PNG.

The format is undocumented; this reader was reverse-engineered from the Cognex Designer
"PCB_Image_Stitching" sample (DS1..DS4_DeskewedImage.cdb - the .png there is a .cdb too):
  - magic 00 00 00 20 88 44 22 11, raw uncompressed pixels, VisionPro XML/.NET metadata at the end
  - every row is `stride` bytes = 6*W (+ padding to a multiple of 4) holding 4 planes of W pixels:
      range     uint16 LE  (height map)
      intensity uint16 LE  (grey value in the high byte)
      mask      uint8 0/255 (valid pixels), stored twice
The stride is found by row autocorrelation; the row start (header length) is found by requiring
the two mask planes to agree with each other and with the non-zero pixels of the range plane.

Outputs per file: <name>_range.png (16-bit), <name>_range_vis.png (8-bit, contrast-stretched,
best for stitching), <name>_intensity.png (8-bit, laser intensity, speckled), <name>_mask.png.

Example:
  python -m sffa_tk.cdb2png --dir "path/to/Images" --out sffa_tk/data/cognex
"""
import argparse
import glob
import os

import cv2
import numpy as np

MAGIC = bytes.fromhex("0000002088442211")


def find_stride(buf, lo=1000, hi=20000):
    nz = np.flatnonzero(buf)
    mid = (nz[0] + nz[-1]) // 2
    chunk = buf[max(0, mid - 4_000_000): mid + 4_000_000].astype(np.int16)
    return min(range(lo, hi), key=lambda w: np.abs(chunk[:-w:7] - chunk[w::7]).mean())


def planes(rows, k, W):
    """Split rows (already starting at byte k of the row) into range/intensity/mask planes."""
    idx = (k + np.arange(6 * W)) % rows.shape[1]
    r = rows[:, idx]
    rng = np.ascontiguousarray(r[:, :2 * W]).view("<u2")
    inten = np.ascontiguousarray(r[:, 2 * W:4 * W]).view("<u2")
    return rng, inten, r[:, 4 * W:5 * W], r[:, 5 * W:6 * W]


def find_row_start(rows, W):
    """Byte offset of the range plane inside a row read from file offset 0."""
    stride = rows.shape[1]
    sample = rows[np.flatnonzero((rows != 0).any(1))[::8]]
    # 1) coarse: the two mask planes are W bytes apart and identical where non-zero
    eq = ((sample == np.roll(sample, -W, axis=1)) & (sample != 0)).sum(0).astype(np.float64)
    win = np.convolve(np.concatenate([eq, eq[:W]]), np.ones(W), "valid")[:stride]
    cands = (np.flatnonzero(win >= 0.98 * win.max()) - 4 * W) % stride

    # 2) fine: valid range pixels must coincide with the mask
    def score(k):
        rng, _, m1, m2 = planes(sample, k, W)
        return ((rng != 0) == (m1 != 0)).mean() + ((m1 != 0) == (m2 != 0)).mean()

    return int(max(cands, key=score))


def read_cdb(path, stride=None):
    raw = np.fromfile(path, np.uint8)
    if raw[:8].tobytes() != MAGIC:
        raise ValueError(f"{path}: not a Cognex .cdb (magic {raw[:8].tobytes().hex()})")
    tail = raw[-4_000_000:].tobytes().find(b"<?xml")
    end = len(raw) - 4_000_000 + tail if tail >= 0 else len(raw)
    buf = raw[:end]
    stride = stride or find_stride(buf)
    W = stride // 6

    rows = buf[: len(buf) // stride * stride].reshape(-1, stride)
    k = find_row_start(rows, W)
    rows = buf[k: k + (len(buf) - k) // stride * stride].reshape(-1, stride)
    rng, inten, mask, _ = planes(rows, 0, W)

    ys, xs = np.nonzero(mask)
    crop = np.s_[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return {"range": rng[crop], "intensity": (inten[crop] >> 8).astype(np.uint8), "mask": mask[crop],
            "stride": stride, "offset": k}


def stretch(range16, mask):
    valid = (mask > 0) & (range16 > 0)
    lo, hi = np.percentile(range16[valid], [1, 99])
    vis = np.clip((range16.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    vis[~valid] = 0
    return vis


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="folder with Cognex files (any extension, detected by magic)")
    src.add_argument("--files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=None, help="row length in bytes (default: auto)")
    a = p.parse_args()

    files = a.files or sorted(glob.glob(os.path.join(a.dir, "*")))
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        if not os.path.isfile(f):
            continue
        with open(f, "rb") as fh:
            if fh.read(8) != MAGIC:
                continue
        img = read_cdb(f, a.stride)
        name = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, f"{name}_range.png"), img["range"])
        cv2.imwrite(os.path.join(a.out, f"{name}_range_vis.png"), stretch(img["range"], img["mask"]))
        cv2.imwrite(os.path.join(a.out, f"{name}_intensity.png"), img["intensity"])
        cv2.imwrite(os.path.join(a.out, f"{name}_mask.png"), img["mask"])
        h, w = img["range"].shape
        print(f"{os.path.basename(f)}: stride={img['stride']} row_offset={img['offset']} -> {w}x{h}")


if __name__ == "__main__":
    main()
