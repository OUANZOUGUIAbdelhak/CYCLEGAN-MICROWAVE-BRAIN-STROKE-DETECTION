"""
phase2_baseline_cnn.py  (Phase 2)  -- bleed-vs-healthy on DIRTY data, the "before".
Requirement: >= 0.70 test accuracy.

We report two classifiers on the raw (dirty) data:
  * plain-CNN on raw [2,256,256]           -> shows how hard the raw problem is
  * symmetry-CNN on [raw + mirror-asym]    -> physically-motivated, head-invariant
The symmetry channels come from |S - mirror(S)|: a bleed is one-sided and breaks
the head's natural left-right symmetry, which generalises across heads.
"""
import os, sys, json, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (read_index, load_split, channel_stats, train_eval,
                    find_mirror_axis, make_transform, DIR_2)


def stratified_val(train_rows, frac=0.15, seed=0):
    """Split loaded list into (train, val) keeping class balance."""
    random.seed(seed)
    byc = {0: [], 1: []}
    for item in train_rows:
        byc[item[1]].append(item)
    tr, va = [], []
    for c, items in byc.items():
        random.shuffle(items)
        k = int(len(items) * frac)
        va += items[:k]; tr += items[k:]
    return tr, va


if __name__ == "__main__":
    rows = read_index()
    all_train = load_split(rows, "train")          # all 28 heads
    test = load_split(rows, "test")                # unseen heads 29, 31
    train, val = stratified_val(all_train, frac=0.15)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # mirror axis from healthy training scans only
    m = find_mirror_axis([a for a, y, _ in train if y == 0])
    print("mirror axis m =", m)
    json.dump({"mirror_axis": m}, open(os.path.join(DIR_2, "mirror_axis.json"), "w"))

    import torch
    results = {}
    # plain-CNN (contrast, 1 seed) and symmetry-CNN (3 seeds, report mean)
    plan = [("raw", 2, "plain-CNN raw", [0]),
            ("raw+asym", 4, "symmetry-CNN raw", [0, 1, 2])]
    for kind, in_ch, tag, seeds in plan:
        tf = make_transform(kind, m)
        tr = [(tf(a), y, h) for a, y, h in train]
        va = [(tf(a), y, h) for a, y, h in val]
        te = [(tf(a), y, h) for a, y, h in test]
        mean, std = channel_stats([a for a, _, _ in tr])
        print(f"\n--- training {tag}  (in_ch={in_ch}, seeds={seeds}) ---")
        seed_metrics, best = [], None
        for s in seeds:
            metrics, model, hist = train_eval(tr, va, te, mean, std, epochs=80,
                                              lr=2e-4, tag=tag, in_ch=in_ch,
                                              seed=s, verbose=(s == seeds[0]))
            print(f"    seed {s}: val={metrics['val_acc']:.3f}  test={metrics['test_acc']:.3f}")
            seed_metrics.append(metrics)
            if best is None or metrics["val_acc"] >= best[0]:
                best = (metrics["val_acc"], model, hist, mean, std)
        tests = [mm["test_acc"] for mm in seed_metrics]
        vals = [mm["val_acc"] for mm in seed_metrics]
        results[kind] = {"test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
                         "test_all": tests, "val_all": vals,
                         "history": best[2], "metrics": seed_metrics[int(np.argmax(vals))]}
        print(f"    -> test = {np.mean(tests):.3f} +/- {np.std(tests):.3f}")
        if kind == "raw+asym":
            torch.save(best[1].state_dict(), os.path.join(DIR_2, "baseline_cnn.pt"))
            np.savez(os.path.join(DIR_2, "baseline_cnn_norm.npz"), mean=best[3], std=best[4])

    json.dump(results, open(os.path.join(DIR_2, "baseline_metrics.json"), "w"), indent=2)
    print("\n=== Phase 2 summary (DIRTY data) ===")
    for k, v in results.items():
        print(f"  {k:10s}  test = {v['test_mean']:.3f} +/- {v['test_std']:.3f}  "
              f"(per-seed {['%.2f'%t for t in v['test_all']]})")

    # figure: symmetry-CNN training curve
    hist = results["raw+asym"]["history"]
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train_acc"] for h in hist], label="train")
    plt.plot(ep, [h["val_acc"] for h in hist], label="val")
    plt.axhline(0.7, ls="--", c="grey", label="0.70 target")
    plt.title(f"Phase 2 symmetry-CNN on DIRTY data "
              f"(test={results['raw+asym']['metrics']['test_acc']:.2f})")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.ylim(0, 1.02); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_2, "fig_phase2_curve.png"), dpi=110)
    print("saved fig_phase2_curve.png")
