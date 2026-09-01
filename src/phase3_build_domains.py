"""
dataset_ab.py  (Phase 3)
------------------------
Build the two domains for CycleGAN, following the paper's data-processing recipe.

  target signal  T_i = Blood_i - mean_healthy_template(head_i)     (the pure bleed)
  background     H_k = an individual healthy run
  mixed / clutter  X = T_i + H_k        (superimposed: bleed on a head background)
  clean target     Y = T_j              (the bleed alone)

By pairing ANY target with ANY background we get ~ (#targets x #backgrounds)
distinct cluttered examples -> plenty of data. Training is UNPAIRED: __getitem__
returns an independently sampled A (mixed) and B (clean).

Normalisation: domain A by a_scale, domain B by b_scale (they live at very
different amplitudes), each mapped to about [-1, 1] for the tanh generators.
"""
import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from common import read_index, CACHE, DIR_3


def _load(cache_key):
    return np.load(os.path.join(CACHE, cache_key))


def downsample_freq(arr, f_out):
    """[2,256,256] -> [2,256,f_out] by average-pooling the frequency axis.
    Antenna axis (256 = 16x16) is kept intact so Phase-6 physics can reshape it."""
    F = arr.shape[-1]
    if f_out is None or f_out == F:
        return arr
    k = F // f_out
    return arr[:, :, :k * f_out].reshape(arr.shape[0], arr.shape[1], f_out, k).mean(-1)


def build_bank(split="train"):
    """Return targets, backgrounds, and per-head info for a split."""
    rows = [r for r in read_index() if r["split"] == split]
    blood = [r for r in rows if r["label"] == "blood"]
    healthy = [r for r in rows if r["label"] == "healthy"]

    # per-head healthy runs -> template = mean of runs
    healthy_by_head = {}
    for r in healthy:
        healthy_by_head.setdefault(int(r["head"]), []).append(_load(r["cache"]))
    template = {h: np.mean(v, axis=0) for h, v in healthy_by_head.items()}

    # targets = blood - template(head)
    targets = []
    for r in blood:
        h = int(r["head"])
        T = _load(r["cache"]) - template[h]
        targets.append({"arr": T.astype(np.float32), "head": h, "file": r["file"]})

    # backgrounds = individual healthy runs (all of them)
    backgrounds = [v.astype(np.float32) for runs in healthy_by_head.values() for v in runs]
    return targets, backgrounds, healthy_by_head, template


def compute_scales(targets, backgrounds, n=400):
    """Robust (99.9pct) scales for domain A (mixed) and B (clean target)."""
    rng = np.random.default_rng(0)
    b_vals = np.concatenate([np.abs(t["arr"]).reshape(-1) for t in targets])
    b_scale = float(np.percentile(b_vals, 99.9))
    a_vals = []
    for _ in range(n):
        t = targets[rng.integers(len(targets))]["arr"]
        bg = backgrounds[rng.integers(len(backgrounds))]
        a_vals.append(np.abs(t + bg).reshape(-1))
    a_scale = float(np.percentile(np.concatenate(a_vals), 99.9))
    return a_scale, b_scale


class ABDataset(Dataset):
    """Unpaired A (mixed/cluttered) and B (clean target) sampler."""
    def __init__(self, split="train", length=4000, scales=None, target_jitter=True,
                 seed=0, f_out=None, p_nobleed=0.35, asym_mode=False, mirror_m=8):
        self.targets, self.backgrounds, _, _ = build_bank(split)
        self.asym_mode = asym_mode
        if asym_mode:
            # work in the left-right symmetry-difference space: clutter is cancelled
            # and the one-sided bleed is exposed, so the CycleGAN cannot wash it away.
            from common import mirror_asym
            for t in self.targets:
                t["arr"] = mirror_asym(t["arr"], mirror_m)
            self.backgrounds = [mirror_asym(b, mirror_m) for b in self.backgrounds]
        if scales is None:
            scales = compute_scales(self.targets, self.backgrounds)
        self.a_scale, self.b_scale = scales
        self.length = length
        self.target_jitter = target_jitter
        self.f_out = f_out
        self.p_nobleed = p_nobleed          # fraction of samples with NO bleed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def _rand_target(self):
        return self.targets[self.rng.integers(len(self.targets))]["arr"]

    def _rand_bg(self):
        return self.backgrounds[self.rng.integers(len(self.backgrounds))]

    def __getitem__(self, idx):
        # --- domain A + its content target ---
        if self.rng.random() < self.p_nobleed:
            # NO-bleed case: A is a pure head background; the clean answer is ZERO.
            # This teaches G that "no bleed in -> no bleed out" (crucial for Phase 5).
            A = self._rand_bg() / self.a_scale
            A_target = np.zeros_like(A)
        else:
            T = self._rand_target()
            if self.target_jitter:
                T = T * float(self.rng.uniform(0.6, 1.4))     # vary bleed strength
            A = (T + self._rand_bg()) / self.a_scale
            A_target = T / self.b_scale
        # --- domain B: independently sampled clean signal (UNPAIRED with A) ---
        if self.rng.random() < self.p_nobleed:
            B = np.zeros_like(A)                               # "clean" can be no-bleed
        else:
            B = self._rand_target() / self.b_scale
        A = downsample_freq(np.clip(A, -1, 1), self.f_out)
        B = downsample_freq(np.clip(B, -1, 1), self.f_out)
        A_target = downsample_freq(np.clip(A_target, -1, 1), self.f_out)
        return {"A": torch.tensor(A, dtype=torch.float32),
                "B": torch.tensor(B, dtype=torch.float32),
                "A_target": torch.tensor(A_target, dtype=torch.float32)}


def real_blood_samples(split, a_scale):
    """Yield (normalised real Blood scan [2,256,256], label=1, head). For Phase 5."""
    rows = [r for r in read_index() if r["split"] == split and r["label"] == "blood"]
    for r in rows:
        arr = _load(r["cache"]) / a_scale
        yield np.clip(arr, -1, 1).astype(np.float32), 1, int(r["head"])


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    targets, backgrounds, hbh, template = build_bank("train")
    a_scale, b_scale = compute_scales(targets, backgrounds)
    print(f"#targets={len(targets)}  #backgrounds={len(backgrounds)}  "
          f"=> up to {len(targets)*len(backgrounds)} distinct mixed examples")
    print(f"a_scale={a_scale:.4f}  b_scale={b_scale:.4f}")
    json.dump({"a_scale": a_scale, "b_scale": b_scale},
              open(os.path.join(DIR_3, "ab_scales.json"), "w"), indent=2)

    # visualise one constructed (mixed, clean) pair
    t = targets[0]["arr"]; bg = backgrounds[0]
    mixed = t + bg
    def mag(x): return np.sqrt(x[0]**2 + x[1]**2)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for a, d, ti in zip(ax, [mag(bg), mag(mixed), mag(t)],
                        ["background H_k (clutter)", "A: mixed  X = T + H  (dirty)",
                         "B: clean target  T (bleed)"]):
        im = a.imshow(d, aspect="auto", cmap="magma", vmax=np.percentile(d, 99))
        a.set_title(ti, fontsize=10); a.set_xlabel("freq"); a.set_ylabel("antenna pair")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Phase 3 — constructing domain A (mixed) and domain B (clean) by superposition")
    fig.tight_layout(); fig.savefig(os.path.join(DIR_3, "fig_phase3_construction.png"), dpi=110)
    print("saved fig_phase3_construction.png")

    # sanity: dataset yields correct shapes and ranges
    ds = ABDataset("train", length=10)
    s = ds[0]
    print("sample A", tuple(s["A"].shape), "range", (float(s["A"].min()), float(s["A"].max())))
    print("sample B", tuple(s["B"].shape), "range", (float(s["B"].min()), float(s["B"].max())))
