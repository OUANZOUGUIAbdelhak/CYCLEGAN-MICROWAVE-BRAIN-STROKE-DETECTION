"""
train_cyclegan.py  (Phase 4)
----------------------------
Train G,F,D_A,D_B with the CycleGAN objective:
    L = L_GAN(G,D_B) + L_GAN(F,D_A) + lambda_cyc * L_cycle + lambda_id * L_identity
LSGAN (MSE) adversarial loss; L1 cycle + identity. Optional physics loss (Phase 6).

Runs are checkpointed and resumable. Besides the GAN losses we log an honest
paired metric: we build mixed = target + background (known target), run G, and
measure how well G recovers the true target (correlation + relative error).

Usage:
    python train_cyclegan.py --iters 50000            # full run (hours)
    python train_cyclegan.py --iters 200 --tag smoke  # quick smoke test
    python train_cyclegan.py --resume                 # continue latest ckpt
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from phase4_models import (ResnetGenerator, NLayerDiscriminator, init_weights,
                           ImagePool, SymGenerator)
from phase3_build_domains import ABDataset, build_bank, compute_scales, downsample_freq
from common import DIR_4, DEVICE

CKPT_DIR = DIR_4          # all Phase-4 checkpoints/logs/samples land in outputs/4_cyclegan/


def physics_loss(x_norm, b_scale=None):
    """Reciprocity penalty on a generated signal (Phase 6).
    For reciprocal media the scattering matrix is symmetric: S_ij = S_ji. This holds
    for full matrices AND their differences (our clean target), and the real data
    obeys it to ~0.08%. A plain generator can break this symmetry, so we penalise it.
    x_norm: [B, 2, 256, F] with the 256 antenna-pair axis = flattened 16x16.

    We penalise the antisymmetric part RELATIVE to the signal magnitude, so the
    objective matches the reported reciprocity_error (a purely absolute penalty is
    trivially satisfied by shrinking the output)."""
    B, C, P, F = x_norm.shape
    x = x_norm.reshape(B, C, 16, 16, F)
    xt = x.transpose(2, 3)                          # swap transmit/receive indices
    anti = ((x - xt) ** 2).mean()
    energy = (x ** 2).mean() + 1e-6
    return anti / energy


def content_loss(pred, target):
    """Self-supervised content anchor. The target is SPARSE (mostly ~0 with faint
    bleed bands), so plain L1 is minimised by outputting zeros. We use
    (1 - cosine similarity) to force the STRUCTURE to match on bleed samples, plus an
    L1 term that also correctly drives NO-BLEED samples (zero target) to zero."""
    B = pred.shape[0]
    p = pred.reshape(B, -1); t = target.reshape(B, -1)
    tnorm = t.norm(dim=1)
    pc = p - p.mean(1, keepdim=True); tc = t - t.mean(1, keepdim=True)
    cos = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)
    struct = (1 - cos) * (tnorm > 1e-6).float()      # cosine only for real bleeds
    l1 = (p - t).abs().mean(1)                        # drives zero-targets to zero
    return (struct + 0.5 * l1).mean()


def reciprocity_error(x_norm):
    """Reported (not-for-grad) relative reciprocity error |S - S^T| / |S|."""
    B, C, P, F = x_norm.shape
    x = x_norm.reshape(B, C, 16, 16, F)
    xt = x.transpose(2, 3)
    num = (x - xt).abs().mean()
    den = x.abs().mean() + 1e-9
    return (num / den).item()


def build_paired_eval(n=32, seed=1, f_out=None, asym_mode=False, mirror_m=8):
    """Fixed (mixed, target) pairs for honest quality tracking."""
    targets, backgrounds, _, _ = build_bank("train")
    if asym_mode:
        from common import mirror_asym
        for t in targets:
            t["arr"] = mirror_asym(t["arr"], mirror_m)
        backgrounds = [mirror_asym(b, mirror_m) for b in backgrounds]
    a_scale, b_scale = compute_scales(targets, backgrounds)
    rng = np.random.default_rng(seed)
    A, T = [], []
    for _ in range(n):
        t = targets[rng.integers(len(targets))]["arr"]
        bg = backgrounds[rng.integers(len(backgrounds))]
        A.append(downsample_freq(np.clip((t + bg) / a_scale, -1, 1), f_out))
        T.append(downsample_freq(np.clip(t / b_scale, -1, 1), f_out))
    return (torch.tensor(np.stack(A), dtype=torch.float32),
            torch.tensor(np.stack(T), dtype=torch.float32), a_scale, b_scale)


@torch.no_grad()
def paired_quality(G, A, T):
    G.eval()
    pred = G(A.to(DEVICE)).cpu()
    G.train()
    p = pred.reshape(pred.shape[0], -1); t = T.reshape(T.shape[0], -1)
    p = p - p.mean(1, keepdim=True); t = t - t.mean(1, keepdim=True)
    corr = ((p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)).mean().item()
    rel = ((pred - T).norm() / (T.norm() + 1e-8)).item()
    return corr, rel


def save_samples(G, F, A, T, path, b_scale):
    G.eval()
    with torch.no_grad():
        cleaned = G(A.to(DEVICE)).cpu()
        recon = F(cleaned.to(DEVICE)).cpu()
    G.train()
    def mag(x): return torch.sqrt(x[0]**2 + x[1]**2).numpy()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, d, ti in zip(ax, [mag(A[0]), mag(cleaned[0]), mag(recon[0]), mag(T[0])],
                        ["A: dirty (input)", "G(A): decluttered", "F(G(A)): cycle recon",
                         "true clean target"]):
        im = a.imshow(d, aspect="auto", cmap="magma", vmax=np.percentile(d, 99) + 1e-9)
        a.set_title(ti, fontsize=10); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_cyc", type=float, default=10.0)
    ap.add_argument("--lambda_id", type=float, default=5.0)
    ap.add_argument("--lambda_phys", type=float, default=0.0)   # Phase 6 sets >0
    ap.add_argument("--lambda_sup", type=float, default=0.0)    # self-supervised content anchor
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--f_out", type=int, default=128)   # frequency bins (speed lever)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", type=str, default="")  # warm-start weights (Phase 6)
    ap.add_argument("--gen_asym", action="store_true")     # feed G the symmetry channels
    ap.add_argument("--asym_mode", action="store_true")    # train in symmetry-difference space
    ap.add_argument("--clip", type=float, default=5.0)     # gradient-norm clip
    ap.add_argument("--d_lr_mult", type=float, default=1.0)  # discriminator LR multiplier (<1 tames D)
    ap.add_argument("--inst_noise", type=float, default=0.0)  # instance-noise std (decays over run)
    ap.add_argument("--out_dir", type=str, default=DIR_4)  # where checkpoints/logs land
    args = ap.parse_args()

    global CKPT_DIR
    CKPT_DIR = args.out_dir; os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    ds = ABDataset("train", length=args.batch * 500, f_out=args.f_out, asym_mode=args.asym_mode)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=0, drop_last=True)
    b_scale = ds.b_scale
    json.dump({"a_scale": ds.a_scale, "b_scale": ds.b_scale},
              open(os.path.join(CKPT_DIR, "scales.json"), "w"))

    if args.gen_asym:
        G = init_weights(SymGenerator(ResnetGenerator(in_ch=4), use_asym=True)).to(DEVICE)
    else:
        G = init_weights(ResnetGenerator(in_ch=2)).to(DEVICE)   # A->B (plain)
    F = init_weights(ResnetGenerator()).to(DEVICE)   # B->A
    D_A = init_weights(NLayerDiscriminator()).to(DEVICE)
    D_B = init_weights(NLayerDiscriminator()).to(DEVICE)
    optG = torch.optim.Adam(list(G.parameters()) + list(F.parameters()),
                            lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                            lr=args.lr, betas=(0.5, 0.999))
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    poolA, poolB = ImagePool(), ImagePool()

    start_iter, log = 0, []
    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}_latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(st["G"]); F.load_state_dict(st["F"])
        D_A.load_state_dict(st["D_A"]); D_B.load_state_dict(st["D_B"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        start_iter = st["iter"]; log = st.get("log", [])
        print(f"resumed from iter {start_iter}")
    elif args.init_from:
        st = torch.load(os.path.join(CKPT_DIR, args.init_from), map_location=DEVICE)
        G.load_state_dict(st["G"]); F.load_state_dict(st["F"])
        D_A.load_state_dict(st["D_A"]); D_B.load_state_dict(st["D_B"])
        print(f"warm-started weights from {args.init_from} (iter {st.get('iter','?')}), "
              f"training fresh with lambda_phys={args.lambda_phys}")

    Aev, Tev, _, _ = build_paired_eval(f_out=args.f_out, asym_mode=args.asym_mode)

    def lr_at(it):   # constant first half, linear decay to 0 in second half
        half = args.iters // 2
        return args.lr if it < half else args.lr * max(0.0, (args.iters - it) / (args.iters - half))

    def set_target(pred, real):
        return (torch.ones_like if real else torch.zeros_like)(pred)

    it = start_iter; t0 = time.time(); data_iter = iter(dl); best_corr = -1.0
    while it < args.iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); batch = next(data_iter)
        a = batch["A"].to(DEVICE); b = batch["B"].to(DEVICE)
        a_tgt = batch["A_target"].to(DEVICE)
        for g in optG.param_groups:
            g["lr"] = lr_at(it)
        for g in optD.param_groups:
            g["lr"] = lr_at(it) * args.d_lr_mult

        # instance noise: keeps the discriminators from becoming "perfect" (dA,dB->0),
        # which is what collapses the adversarial game and blows up to NaN. Decays to 0.
        nstd = args.inst_noise * max(0.0, 1.0 - it / (0.8 * args.iters))
        def Dn(net, x):
            return net(x + nstd * torch.randn_like(x) if nstd > 0 else x)

        # ---- generators ----
        optG.zero_grad()
        fake_b = G(a); fake_a = F(b)
        rec_a = F(fake_b); rec_b = G(fake_a)
        # adversarial
        loss_g = mse(Dn(D_B, fake_b), set_target(D_B(fake_b), True)) \
               + mse(Dn(D_A, fake_a), set_target(D_A(fake_a), True))
        # cycle
        loss_cyc = l1(rec_a, a) + l1(rec_b, b)
        # identity
        loss_id = l1(G(b), b) + l1(F(a), a)
        # self-supervised content anchor: G(A) should equal A's true bleed target
        loss_sup = content_loss(fake_b, a_tgt) if args.lambda_sup > 0 else torch.tensor(0.0, device=DEVICE)
        loss_phys = physics_loss(fake_b) if args.lambda_phys > 0 else torch.tensor(0.0, device=DEVICE)
        gloss = loss_g + args.lambda_cyc * loss_cyc + args.lambda_id * loss_id \
              + args.lambda_sup * loss_sup + args.lambda_phys * loss_phys
        if not torch.isfinite(gloss):
            print(f"[stop] non-finite generator loss at iter {it}; best checkpoint kept.")
            break
        gloss.backward()
        torch.nn.utils.clip_grad_norm_(list(G.parameters()) + list(F.parameters()), args.clip)
        optG.step()

        # ---- discriminators ----
        optD.zero_grad()
        fb = poolB.query(fake_b.detach()); fa = poolA.query(fake_a.detach())
        lD_B = 0.5 * (mse(Dn(D_B, b), set_target(D_B(b), True)) +
                      mse(Dn(D_B, fb), set_target(D_B(fb), False)))
        lD_A = 0.5 * (mse(Dn(D_A, a), set_target(D_A(a), True)) +
                      mse(Dn(D_A, fa), set_target(D_A(fa), False)))
        (lD_A + lD_B).backward()
        torch.nn.utils.clip_grad_norm_(list(D_A.parameters()) + list(D_B.parameters()), args.clip)
        optD.step()

        it += 1
        if it % 50 == 0:
            dt = (time.time() - t0) / 50; t0 = time.time()
            corr, rel = paired_quality(G, Aev, Tev)
            rec = {"iter": it, "g": float(loss_g.item()), "cyc": float(loss_cyc.item()),
                   "id": float(loss_id.item()), "sup": float(loss_sup.item()),
                   "dA": float(lD_A.item()), "dB": float(lD_B.item()),
                   "phys": float(loss_phys.item()),
                   "recip": reciprocity_error(fake_b.detach()),
                   "corr": corr, "rel": rel, "sec_it": dt}
            log.append(rec)
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                torch.save({"iter": it, "G": G.state_dict(), "F": F.state_dict(),
                            "a_scale": ds.a_scale, "b_scale": ds.b_scale, "corr": corr,
                            "args": vars(args)}, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            print(f"it {it:6d}/{args.iters}  g={rec['g']:.3f} cyc={rec['cyc']:.3f} "
                  f"dA={rec['dA']:.3f} dB={rec['dB']:.3f}  corr={corr:.3f} rel={rel:.3f} "
                  f"({dt*1000:.0f} ms/it)", flush=True)
        if it % args.sample_every == 0:
            save_samples(G, F, Aev, Tev, os.path.join(CKPT_DIR, f"{args.tag}_it{it:06d}.png"), b_scale)
        if it % args.ckpt_every == 0 or it == args.iters:
            torch.save({"iter": it, "G": G.state_dict(), "F": F.state_dict(),
                        "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "log": log, "args": vars(args),
                        "a_scale": ds.a_scale, "b_scale": ds.b_scale}, ckpt_path)
            json.dump(log, open(os.path.join(CKPT_DIR, f"{args.tag}_log.json"), "w"))

    print("training done at iter", it)


if __name__ == "__main__":
    main()
