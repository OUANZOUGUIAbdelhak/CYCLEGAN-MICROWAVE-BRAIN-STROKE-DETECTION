"""
phase2_mlp.py  (Phase 2, final)
-------------------------------
Neural-network classifier on the physically-motivated symmetry feature.

Feature: for each scan compute the left-right asymmetry  a = |S - P S P|  (P is the
mirror permutation found from healthy scans), then average its magnitude over
frequency -> 256 numbers (one per antenna pair). A small MLP classifies these.
Head-invariant, sample-efficient, and needs no healthy baseline of the patient.

We also print the plain-CNN-on-raw number (from phase2_metrics.json) for contrast.
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from common import (read_index, load_split, find_mirror_axis, mirror_asym,
                    DEVICE, DIR_2)


def asym_energy(arr, m):
    """[2,256,256] -> 256-dim frequency-averaged asymmetry magnitude."""
    a = mirror_asym(arr, m)                      # [2,256,256] (re,im of S-PSP)
    mag = np.sqrt(a[0] ** 2 + a[1] ** 2)          # [256,256]
    return mag.mean(axis=1)                        # [256]


class MLP(nn.Module):
    def __init__(self, d=256, h=64, p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def run(train, val, test, m, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    def build(ds):
        X = np.stack([asym_energy(a, m) for a, _, _ in ds]).astype(np.float32)
        y = np.array([b for _, b, _ in ds], dtype=np.float32)
        return X, y
    Xtr, ytr = build(train); Xva, yva = build(val); Xte, yte = build(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, Xte_t = torch.tensor(Xva).to(DEVICE), torch.tensor(Xte).to(DEVICE)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    acc = lambda X, y: float(((torch.sigmoid(model(X)).detach().cpu().numpy() > .5)
                              == y).mean())
    best_va, best_state, best_te = -1, None, 0
    for ep in range(200):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            xb = (Xtr[idx] + 0.05 * torch.randn_like(Xtr[idx])).to(DEVICE)
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval()
        va = acc(Xva_t, yva); te = acc(Xte_t, yte)
        if va >= best_va:
            best_va, best_te = va, te
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_va, best_te, best_state, (mu, sd)


if __name__ == "__main__":
    rows = read_index()
    all_train = load_split(rows, "train")
    test = load_split(rows, "test")
    # stratified val
    import random; random.seed(0)
    byc = {0: [], 1: []}
    for it in all_train: byc[it[1]].append(it)
    tr, va = [], []
    for c in byc:
        random.shuffle(byc[c]); k = int(0.15 * len(byc[c]))
        va += byc[c][:k]; tr += byc[c][k:]
    m = find_mirror_axis([a for a, y, _ in tr if y == 0])
    print("mirror axis:", m, " train/val/test:", len(tr), len(va), len(test))

    vals, tests, best = [], [], None
    for s in range(5):
        bva, bte, state, norm = run(tr, va, test, m, s)
        vals.append(bva); tests.append(bte)
        print(f"  seed {s}: val={bva:.3f}  test={bte:.3f}")
        if best is None or bva >= best[0]:
            best = (bva, state, norm)
    print(f"\nMLP-on-symmetry  test = {np.mean(tests):.3f} +/- {np.std(tests):.3f}"
          f"   (val {np.mean(vals):.3f})")

    torch.save({"state": best[1], "mu": best[2][0], "sd": best[2][1], "mirror": m},
               os.path.join(DIR_2, "symmetry_mlp.pt"))
    summary = {"model": "MLP-on-symmetry", "mirror_axis": m,
               "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
               "test_all": tests, "val_mean": float(np.mean(vals))}
    json.dump(summary, open(os.path.join(DIR_2, "symmetry_mlp_metrics.json"), "w"), indent=2)
    print("saved 2_classification_before/symmetry_mlp.pt + symmetry_mlp_metrics.json")
