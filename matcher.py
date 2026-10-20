"""Section 2.3 - SFFA-TK: FPN + Transformer (LoFTR) with K-means filtering of coarse matches.

The paper's FPN / 4x(self+cross) attention / dual-softmax + MNN / fine refinement is
exactly LoFTR [28], so kornia's pretrained LoFTR is reused. The K-means step is inserted
between coarse matching and fine refinement, as in the paper, so only the selected
cluster is refined (this is where the speed-up comes from).
"""
import numpy as np
import torch
import torch.nn as nn
import kornia.feature as KF
from sklearn.cluster import KMeans

_COARSE_KEYS = ("b_ids", "i_ids", "j_ids", "gt_mask", "m_bids", "mkpts0_c", "mkpts1_c", "mconf")


def select_cluster(points, n_clusters=4, seed=0):
    """K-means on keypoint coordinates; pick the cluster with "the highest number of points
    and the most concentrated distribution" (Sec. 2.3-4).

    Both criteria count equally: clusters are ranked by size (desc) and by spread = mean
    distance to centroid (asc), the smallest rank sum wins, ties go to the larger cluster.
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(points)
    counts = np.array([(km.labels_ == k).sum() for k in range(n_clusters)])
    spreads = np.array([np.linalg.norm(points[km.labels_ == k] - km.cluster_centers_[k], axis=1).mean()
                        if counts[k] else np.inf for k in range(n_clusters)])
    rank_sum = np.argsort(np.argsort(-counts)) + np.argsort(np.argsort(spreads))
    best = min(range(n_clusters), key=lambda k: (rank_sum[k], -counts[k]))
    return km.labels_, best


class KMeansCoarseFilter(nn.Module):
    """Wraps LoFTR's CoarseMatching: keep top-K by confidence, K-means, keep one cluster."""

    def __init__(self, coarse_matching, top_k=200, n_clusters=4, cluster="auto", enabled=True):
        super().__init__()
        self.inner = coarse_matching
        self.top_k, self.n_clusters, self.cluster, self.enabled = top_k, n_clusters, cluster, enabled
        self.last = {}

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        self.inner(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        conf = data["mconf"]
        keep = torch.argsort(conf, descending=True)[: self.top_k]
        pts0 = data["mkpts0_c"][keep].detach().cpu().numpy()
        pts1 = data["mkpts1_c"][keep].detach().cpu().numpy()
        self.last = {"pts0": pts0, "pts1": pts1, "labels": None, "chosen": None}

        if self.enabled and len(keep) >= self.n_clusters:
            labels, best = select_cluster(pts0, self.n_clusters)
            chosen = best if self.cluster == "auto" else int(self.cluster)
            keep = keep[torch.from_numpy(labels == chosen).to(keep.device)]
            self.last.update(labels=labels, chosen=chosen)

        for k in _COARSE_KEYS:
            if k in data:
                data[k] = data[k][keep]


class SFFATK:
    def __init__(self, pretrained="outdoor", top_k=200, n_clusters=4, cluster="auto",
                 use_kmeans=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KF.LoFTR(pretrained=pretrained).to(self.device).eval()
        self.filter = KMeansCoarseFilter(self.model.coarse_matching, top_k, n_clusters, cluster, use_kmeans)
        self.model.coarse_matching = self.filter

    @torch.inference_mode()
    def match(self, gray0, gray1, max_side=840):
        """gray0/gray1: uint8 HxW. Returns (kpts0, kpts1, conf) in original pixel coords."""
        t0, s0 = self._prep(gray0, max_side)
        t1, s1 = self._prep(gray1, max_side)
        self.scale0 = s0  # to map self.filter.last["pts0"] back to original pixels
        out = self.model({"image0": t0, "image1": t1})
        k0 = out["keypoints0"].cpu().numpy() * s0
        k1 = out["keypoints1"].cpu().numpy() * s1
        return k0, k1, out["confidence"].cpu().numpy()

    def _prep(self, gray, max_side):
        h, w = gray.shape
        scale = min(1.0, max_side / max(h, w))
        nh, nw = int(h * scale) // 8 * 8, int(w * scale) // 8 * 8  # LoFTR needs /8
        t = torch.from_numpy(gray).float()[None, None] / 255.0
        t = torch.nn.functional.interpolate(t, size=(nh, nw), mode="area").to(self.device)
        return t, np.array([w / nw, h / nh], np.float32)
