"""
preprocess.py  (Phase 1)
------------------------
Convert every .s16p file into the cached 2-channel [real, imag] image, write an
index, compute normalisation scales, and save one explanatory figure.

Outputs:
  outputs/cache/raw/<split>__<label>__<file>.npy   float32 [2,256,256]  (raw S units)
  outputs/cache/index.csv                          one row per file + metadata
  outputs/1_preprocess/norm_stats.json             global scales for domain A and B
  outputs/1_preprocess/fig_phase1_sample.png       dirty vs healthy-template vs clean-target
"""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (list_split, load_image, parse_name, load_raw_S, S_to_image,
                    F_OUT, DATA_ROOT, CACHE, INDEX, DIR_1)


def cache_all():
    rows = []
    for split in ["train", "test"]:
        blood, healthy = list_split(DATA_ROOT, split)
        for path in blood + healthy:
            info = parse_name(path)
            img = load_image(path)                       # [2,256,256] raw
            key = f"{split}__{info['label']}__{os.path.splitext(info['file'])[0]}.npy"
            np.save(os.path.join(CACHE, key), img)
            rows.append({
                "cache": key, "split": split, "label": info["label"],
                "y": info["y"], "head": info["head"],
                "file": info["file"],
            })
            print(f"  cached {split}/{info['label']:7s} head{info['head']:02d}  {info['file']}")
    # write index
    with open(INDEX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def compute_norm_stats(rows):
    """Global scales: domain A (blood, dirty) and domain B (blood-healthy, clean)."""
    # A domain: raw blood magnitude
    a_vals = []
    for r in rows:
        if r["split"] == "train" and r["label"] == "blood":
            img = np.load(os.path.join(CACHE, r["cache"]))
            a_vals.append(np.abs(img).reshape(-1))
    a_scale = float(np.percentile(np.concatenate(a_vals), 99.9))

    # B domain: blood - matched healthy (same head), robust scale
    from collections import defaultdict
    by_head_blood = defaultdict(list); by_head_healthy = defaultdict(list)
    for r in rows:
        if r["split"] != "train":
            continue
        (by_head_blood if r["label"] == "blood" else by_head_healthy)[r["head"]].append(r["cache"])
    b_vals = []
    for head, bl in by_head_blood.items():
        hl = by_head_healthy.get(head, [])
        if not hl:
            continue
        h0 = np.load(os.path.join(CACHE, hl[0]))          # one healthy template
        for bcache in bl:
            diff = np.load(os.path.join(CACHE, bcache)) - h0
            b_vals.append(np.abs(diff).reshape(-1))
    b_scale = float(np.percentile(np.concatenate(b_vals), 99.9))

    stats = {"a_scale": a_scale, "b_scale": b_scale, "f_out": F_OUT}
    with open(os.path.join(DIR_1, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def make_figure():
    """Pick head 1: show a dirty blood sample, its healthy template, the clean target."""
    blood, healthy = list_split(DATA_ROOT, "train")
    b_path = [p for p in blood if "head01_" in os.path.basename(p)][0]
    h_path = [p for p in healthy if "HM01_" in os.path.basename(p)][0]
    B = load_image(b_path); H = load_image(h_path)
    clean = B - H

    def mag(img):  # magnitude from [2,H,W]
        return np.sqrt(img[0]**2 + img[1]**2)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, data, title in zip(
            ax, [mag(B), mag(H), mag(clean)],
            ["A: DIRTY  (raw Blood = bleed + clutter)",
             "Healthy template (same head)",
             "B: CLEAN target  (Blood - Healthy)"]):
        im = a.imshow(data, aspect="auto", cmap="magma",
                      vmax=np.percentile(data, 99))
        a.set_title(title, fontsize=10)
        a.set_xlabel("frequency bin"); a.set_ylabel("antenna-pair channel")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Phase 1 — what the network sees (|S| magnitude shown; "
                 "networks use real+imag)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(DIR_1, "fig_phase1_sample.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    # also report how much smaller the clean target is
    print(f"\n  mean|A dirty| = {np.abs(B).mean():.4f}   "
          f"mean|clean B| = {np.abs(clean).mean():.4f}   "
          f"ratio = {np.abs(clean).mean()/np.abs(B).mean():.3f}")
    return out


if __name__ == "__main__":
    print("Phase 1: caching all 300 files as [2,256,256] images ...")
    rows = cache_all()
    print(f"\nCached {len(rows)} files.")
    stats = compute_norm_stats(rows)
    print("Normalisation scales:", stats)
    fig = make_figure()
    print("Saved figure:", fig)
