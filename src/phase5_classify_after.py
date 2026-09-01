"""Quick Phase-5 preview: declutter all scans with the current G, then re-train the
Phase-2 symmetry-MLP on the CLEANED data and compare test accuracy to raw (0.79)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from phase4_models import ResnetGenerator
from phase3_build_domains import downsample_freq
from common import read_index, CACHE, find_mirror_axis, mirror_asym, FINAL_MODEL, DIR_5
from phase2_symmetry_mlp import MLP, run   # reuse the exact Phase-2 MLP + training

ck = sys.argv[1] if len(sys.argv) > 1 else FINAL_MODEL
st = torch.load(ck, map_location="cpu")
G = ResnetGenerator(in_ch=2); G.load_state_dict(st["G"]); G.eval(); a = st["a_scale"]
print("probe G from iter", st["iter"])

@torch.no_grad()
def clean(cache):
    x = downsample_freq(np.clip(np.load(os.path.join(CACHE, cache)) / a, -1, 1), 128)
    return G(torch.tensor(x[None], dtype=torch.float32))[0].numpy()   # [2,256,128]

# declutter every scan -> cleaned arrays keyed by cache
rows = read_index()
cleaned = {r["cache"]: clean(r["cache"]) for r in rows}
print("decluttered", len(cleaned), "scans")

def split(which, heads=None):
    out = []
    for r in rows:
        if r["split"] != which: continue
        if heads is not None and int(r["head"]) not in heads: continue
        out.append((cleaned[r["cache"]], int(r["y"]), int(r["head"])))
    return out

# stratified val within train heads
import random; random.seed(0)
allt = split("train"); byc = {0: [], 1: []}
for it in allt: byc[it[1]].append(it)
tr, va = [], []
for c in byc:
    random.shuffle(byc[c]); k = int(0.15 * len(byc[c])); va += byc[c][:k]; tr += byc[c][k:]
te = split("test")

# mirror axis from cleaned healthy (m may differ on cleaned data)
m = find_mirror_axis([a_ for a_, y, _ in tr if y == 0])
print("mirror axis on cleaned:", m)

tests, vals = [], []
for s in range(5):
    bva, bte, _, _ = run(tr, va, te, m, s)
    vals.append(bva); tests.append(bte)
print(f"symmetry-MLP on CLEANED data: test = {np.mean(tests):.3f} +/- {np.std(tests):.3f}  "
      f"(val {np.mean(vals):.3f})   [raw baseline was 0.79]")

summary = {"model": "blind CycleGAN, then symmetry-MLP on cleaned data",
           "checkpoint": os.path.basename(ck), "gen_iter": int(st["iter"]),
           "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
           "test_all": tests, "val_mean": float(np.mean(vals)),
           "raw_baseline": 0.79}
json.dump(summary, open(os.path.join(DIR_5, "phase5_metrics.json"), "w"), indent=2)
print("saved 5_classification_after/phase5_metrics.json")
