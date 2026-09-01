"""Phase 6 — physical-constraint optimization.

A plain generator can produce scattering matrices that violate reciprocity (S != S^T),
which is physically meaningless for a reciprocal antenna array (the real data obeys
S = S^T to ~0.08%). We add a reciprocity penalty to the loss and show it drives the
generated output's reciprocity error down, at little cost to the recovery quality.

Runs two short trainings (baseline vs +physics) and plots reciprocity error vs iters.
"""
import os, sys, json, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import PROJ, DIR_6

CK = DIR_6                      # the physics experiment lives entirely in 6_physics/
PY = sys.executable            # the same python running this script (portable)
TRAIN = os.path.join(os.path.dirname(__file__), "phase4_train.py")


def run(tag, lam_phys, iters=2000):
    cmd = [PY, TRAIN,
           "--iters", str(iters), "--batch", "2", "--f_out", "128",
           "--lambda_cyc", "10", "--lambda_id", "5", "--lambda_sup", "15",
           "--lambda_phys", str(lam_phys), "--lr", "2e-4", "--clip", "1.0",
           "--inst_noise", "0.05", "--tag", tag, "--out_dir", CK,
           "--sample_every", "99999", "--ckpt_every", "99999"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run(cmd, env=env, cwd=PROJ,
                   stdout=open(os.path.join(CK, f"{tag}.log"), "w"), stderr=subprocess.STDOUT)
    return json.load(open(os.path.join(CK, f"{tag}_log.json")))


if __name__ == "__main__":
    print("Phase 6: baseline (no physics) ...")
    base = run("p6_base", 0.0)
    print("Phase 6: with reciprocity loss ...")
    phys = run("p6_phys", 20.0)

    def series(log, key):
        return [r["iter"] for r in log], [r[key] for r in log]

    ib, rb = series(base, "recip"); ip, rp = series(phys, "recip")
    _, cb = series(base, "corr");   _, cp = series(phys, "corr")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(ib, rb, label="baseline (no physics)", color="#d95f02")
    ax[0].plot(ip, rp, label="+ reciprocity loss", color="#1b9e77")
    ax[0].set_title("Reciprocity error of G's output  |S - Sᵀ| / |S|")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("relative reciprocity error")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # smoothed corr for readability
    def smooth(x, k=5):
        x = np.array(x); return np.convolve(x, np.ones(k)/k, mode="valid")
    ax[1].plot(ib[4:], smooth(cb), label="baseline", color="#d95f02")
    ax[1].plot(ip[4:], smooth(cp), label="+ reciprocity", color="#1b9e77")
    ax[1].set_title("Recovery quality (corr) — physics term keeps it comparable")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("corr (smoothed)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(DIR_6, "fig_phase6_reciprocity.png"), dpi=115)

    summary = {
        "baseline_recip_final": float(np.mean(rb[-5:])),
        "physics_recip_final": float(np.mean(rp[-5:])),
        "baseline_corr_final": float(np.mean(cb[-10:])),
        "physics_corr_final": float(np.mean(cp[-10:])),
    }
    json.dump(summary, open(os.path.join(DIR_6, "phase6_metrics.json"), "w"), indent=2)
    print("\n=== Phase 6 summary ===")
    print(json.dumps(summary, indent=2))
    print(f"reciprocity error: {summary['baseline_recip_final']:.4f} (baseline) "
          f"-> {summary['physics_recip_final']:.4f} (+physics)")
