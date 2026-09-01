"""Phase-5 evaluation for the SYMMETRY-INFORMED CycleGAN.
Clutter removal on raw data = symmetric subtraction (S - mirror S) THEN the trained
generator denoises it. We then classify the cleaned output (blood -> structure,
healthy -> ~0) and compare test accuracy to the raw baselines."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from phase4_models import ResnetGenerator
from phase3_build_domains import downsample_freq
from common import read_index, CACHE, mirror_asym, DIR_7
from phase2_symmetry_mlp import run   # the exact Phase-2 MLP + training loop

ck = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DIR_7, "symmetry_informed_G.pt")
st = torch.load(ck, map_location="cpu")
G = ResnetGenerator(in_ch=2); G.load_state_dict(st["G"]); G.eval()
a_scale = st["a_scale"]; m = 8
print("asym-G from iter", st["iter"], "a_scale=%.4f" % a_scale)

@torch.no_grad()
def cleaned_feat(cache):
    raw = np.load(os.path.join(CACHE, cache))          # [2,256,256] raw S
    asym = mirror_asym(raw, m)                          # symmetric subtraction (clutter removal)
    x = downsample_freq(np.clip(asym / a_scale, -1, 1), 128)
    out = G(torch.tensor(x[None], dtype=torch.float32))[0].numpy()  # denoised bleed
    return np.sqrt(out[0] ** 2 + out[1] ** 2).mean(axis=1)          # 256-dim energy/pair

rows = read_index()
feats = {r["cache"]: cleaned_feat(r["cache"]) for r in rows}
print("decluttered+featurised", len(feats), "scans")

def split(which, heads=None):
    out = []
    for r in rows:
        if r["split"] != which: continue
        if heads is not None and int(r["head"]) not in heads: continue
        # phase2_mlp.run expects (arr[2,256,256], y, head) and calls asym_energy on it;
        # here we already have features, so wrap them to bypass -> use a tiny shim below
        out.append((feats[r["cache"]], int(r["y"]), int(r["head"])))
    return out

# --- minimal MLP training directly on the 256-dim features (no re-asymmetry) ---
import torch.nn as nn
from phase2_symmetry_mlp import MLP
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def train_on_feats(tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    def build(ds):
        X = np.stack([f for f, _, _ in ds]).astype(np.float32)
        y = np.array([b for _, b, _ in ds], dtype=np.float32)
        return X, y
    Xtr, ytr = build(tr); Xva, yva = build(va); Xte, yte = build(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr-mu)/sd, (Xva-mu)/sd, (Xte-mu)/sd
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, Xte_t = torch.tensor(Xva).to(DEVICE), torch.tensor(Xte).to(DEVICE)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    acc = lambda X, y: float(((torch.sigmoid(model(X)).detach().cpu().numpy() > .5) == y).mean())
    bva, bte = -1, 0
    for ep in range(200):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            xb = (Xtr[idx] + 0.05*torch.randn_like(Xtr[idx])).to(DEVICE)
            opt.zero_grad(); lossf(model(xb), ytr[idx].to(DEVICE)).backward(); opt.step()
        model.eval(); va = acc(Xva_t, yva)
        if va >= bva: bva, bte = va, acc(Xte_t, yte)
    return bva, bte

import random; random.seed(0)
allt = split("train"); byc = {0: [], 1: []}
for it in allt: byc[it[1]].append(it)
tr, va = [], []
for c in byc:
    random.shuffle(byc[c]); k = int(0.15*len(byc[c])); va += byc[c][:k]; tr += byc[c][k:]
te = split("test")

tests, vals = [], []
for s in range(5):
    bva, bte = train_on_feats(tr, va, te, s)
    vals.append(bva); tests.append(bte)
print(f"symmetry-informed CycleGAN, Phase-5 test = {np.mean(tests):.3f} +/- {np.std(tests):.3f} "
      f"(val {np.mean(vals):.3f})   [raw symmetric-subtraction baseline = 0.79, blind CycleGAN = 0.50]")

summary = {"model": "symmetry-informed CycleGAN (S - mirror S, then G), symmetry-MLP",
           "checkpoint": os.path.basename(ck), "gen_iter": int(st["iter"]),
           "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
           "test_all": tests, "val_mean": float(np.mean(vals)),
           "classical_baseline": 0.79, "blind_cyclegan": 0.50}
json.dump(summary, open(os.path.join(DIR_7, "symmetry_informed_metrics.json"), "w"), indent=2)
print("saved 7_final_comparison/symmetry_informed_metrics.json")
