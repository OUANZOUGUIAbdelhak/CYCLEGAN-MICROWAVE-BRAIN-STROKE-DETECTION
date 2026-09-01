"""Phase-4 showcase figure: run the trained generator G on constructed test examples
and show dirty input -> G decluttered -> cycle reconstruction -> true target.
Uses the best model. This is the visual evidence that G removes clutter."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from phase4_models import ResnetGenerator
from phase3_build_domains import build_bank, compute_scales, downsample_freq
from common import FINAL_MODEL, DIR_4

ck = sys.argv[1] if len(sys.argv) > 1 else FINAL_MODEL
st = torch.load(ck, map_location="cpu")
G = ResnetGenerator(in_ch=2); G.load_state_dict(st["G"]); G.eval()
F = ResnetGenerator(in_ch=2)
if "F" in st: F.load_state_dict(st["F"]); F.eval()
a_scale = st["a_scale"]; b_scale = st["b_scale"]

targets, backgrounds, _, _ = build_bank("train")
rng = np.random.default_rng(3)

def mag(x): return np.sqrt(x[0]**2 + x[1]**2)

rows = 3
fig, ax = plt.subplots(rows, 4, figsize=(15, 3.4*rows))
for r in range(rows):
    t = targets[rng.integers(len(targets))]["arr"]
    bg = backgrounds[rng.integers(len(backgrounds))]
    A = downsample_freq(np.clip((t+bg)/a_scale, -1, 1), 128)
    T = downsample_freq(np.clip(t/b_scale, -1, 1), 128)
    with torch.no_grad():
        gA = G(torch.tensor(A[None], dtype=torch.float32))[0].numpy()
        rec = F(torch.tensor(gA[None], dtype=torch.float32))[0].numpy()
    panels = [(mag(A), "A: dirty input (bleed+clutter)"),
              (mag(gA), "G(A): decluttered"),
              (mag(rec), "F(G(A)): cycle reconstruction"),
              (mag(T), "true clean target (bleed)")]
    for c, (d, title) in enumerate(panels):
        a = ax[r, c]
        im = a.imshow(d, aspect="auto", cmap="magma", vmax=np.percentile(d, 99)+1e-9)
        if r == 0: a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(f"Phase 4 — trained CycleGAN generator (corr={st.get('corr',0):.2f}, iter {st['iter']})",
             fontsize=12)
fig.tight_layout()
out = os.path.join(DIR_4, "fig_phase4_showcase.png")
fig.savefig(out, dpi=105); print("saved", out)
