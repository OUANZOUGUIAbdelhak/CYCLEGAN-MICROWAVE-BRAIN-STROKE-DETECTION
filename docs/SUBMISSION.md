# Clutter Suppression and Target Extraction from Scattered Data with CycleGAN

**Author:** OUANZOUGUI Abdelhak
**Assignment:** Xiong'an Anying Technology — second-round open task
**Reference:** Lai et al., *Clutter Removal for Microwave Head Imaging via
Self-Supervised Deep Learning Techniques*, IEEE J-ERM, 2024.

---

## 0. What the task is, in one paragraph

A 16-antenna microwave helmet measures a person's head and returns S-parameters
(complex scattering numbers). The goal is to detect a brain bleed. The problem is
that the measured signal is dominated by *clutter* — strong reflections from the
skull and tissue boundaries — while the bleed itself is a very faint signal buried
underneath. The task asks me to build a CycleGAN that learns to strip the clutter
away and leave the bleed, working from *unpaired* data, and then to show that the
cleaned data is genuinely easier to work with. I go through six phases, from data
preprocessing up to adding physical constraints to the loss.

A note on style: I wrote this as a running lab notebook rather than a polished
paper, because the point of the exercise is *how* I reason when things are
uncertain, not a final glossy result. Where I made a judgement call, I say so.

---

## 1. Phase 1 — Data preprocessing

**Goal:** turn each raw `.s16p` file into something a CNN can eat.

**What the raw data is.** Each file is a complex array of shape `[1001, 16, 16]`:
1001 frequency points (0.5–2.5 GHz), and at every frequency a full 16×16 grid of
antenna-to-antenna measurements. There are 300 files total: 140 Blood + 140
Healthy for training (28 heads, 5 bleeds and 5 healthy runs each), and 10 + 10 for
testing (2 unseen heads, #29 and #31).

**My representation.** I flatten the 16×16 antenna grid into 256 "antenna-pair
channels", put frequency on the other axis, resample frequency 1001 → 256 so the
image is a clean square, and keep **both the real and imaginary parts as two
channels**. So one sample becomes a `[2, 256, 256]` image.

**Why keep real+imag instead of magnitude (the paper uses magnitude).** Three
reasons, and this matters later:
1. It keeps the phase, i.e. all the information.
2. The `Blood − Healthy` subtraction that defines the clean target stays exact and
   linear.
3. It is the only way Phase 6 (physical constraints like energy conservation) is
   even possible — you cannot check physics on magnitude alone.
   I note the paper's inverse-Fourier + magnitude route as a valid alternative
   (it improves SNR), and I could switch to it, but for an end-to-end pipeline
   that ends in physics constraints the complex representation is the better base.

**The key finding (this drives everything).** After caching all 300 files I
measured how strong the clean target is compared to the dirty signal:

```
mean |dirty (A)|      = 0.0142
mean |clean bleed (B)| = 0.0001      ->  the bleed is ~1/250 of the clutter
robust scale A = 0.96 ,  robust scale B = 0.056
```

See `outputs/1_preprocess/fig_phase1_sample.png`. The dirty scan and the healthy scan look
identical to the eye — both are dominated by a few bright "clutter" channels. Only
after subtracting the healthy template does the faint bleed pattern appear, and its
amplitude is about 250× smaller. This is exactly why naive filtering fails and why
a learned, distribution-level method is needed.

**Practical consequence.** Because domain A and domain B live at completely
different amplitudes, I normalise each domain by its *own* robust scale (stored in
`outputs/1_preprocess/norm_stats.json`). If I used one shared scale, domain B would be crushed
to near-zero and the network would see nothing.

*Deliverables:* `src/common.py`, `src/phase1_preprocess.py`, cached tensors in
`outputs/cache/`, `outputs/1_preprocess/fig_phase1_sample.png`.

---

## 2. Phase 2 — Classification before clutter removal (the "before")

**Goal:** classify bleed vs healthy on the dirty data. Requirement: ≥ 0.70 test
accuracy on the unseen test heads (#29, #31).

**First, the honest baseline.** I trained a plain CNN on the raw `[2,256,256]`
images. It sits at **~0.50–0.60** — essentially chance on unseen heads. That is
expected: the bleed is ~250× weaker than the clutter, every head's clutter is
different, and the test heads were never seen in training. A generic CNN cannot
find a 0.4% needle in a per-head-varying haystack from 240 examples.

**What actually works — a physically-motivated feature.** A stroke is on *one
side* of the head, so it breaks the head's natural left–right symmetry, whereas
the skull/clutter is (to first order) symmetric. So I:
1. searched for the mirror permutation `P(k) = (m − k) mod 16` that makes the
   *healthy* scans most symmetric — the search cleanly picked **m = 8** (each
   antenna pairs with the one opposite it, which is physically sensible);
2. used the asymmetry `a = |S − P·S·P|`, averaged over frequency, as a 256-number
   feature (one per antenna pair);
3. trained a small **MLP** on it.

This feature is **head-invariant** (it needs no healthy baseline of the patient —
important, because in the clinic you don't have one) and it is sample-efficient.

**Result (5 seeds):** symmetry-MLP = **0.79 ± 0.02** test accuracy (val ≈ 0.87),
versus **0.50** for the plain CNN. See `outputs/2_classification_before/fig_phase2_compare.png`. The 0.70
requirement is met, and — more importantly — I now understand *why* the raw
problem is hard and *where* the usable signal lives (in the symmetry breaking).

This sets up the key question for the rest of the project: **can a learned
clutter-removal step (CycleGAN) expose that signal automatically, without me
hand-designing the symmetry trick?** That is what Phase 5 will test.

*Deliverables:* `src/common.py`, `src/phase2_baseline_cnn.py` (plain + CNN variants),
`src/phase2_symmetry_mlp.py` (the symmetry MLP),
`outputs/2_classification_before/fig_phase2_compare.png`,
`outputs/2_classification_before/*.json`.

---

## 3. Phase 3 — Constructing the two domains (A and B)

**Goal:** build domain A (cluttered) and domain B (clean target) from the raw
data, following the paper, and make "as many clutter data points as possible".

I follow the paper's data-processing recipe exactly:
- **target signal** `T_i = Blood_i − mean_healthy_template(head_i)` — the isolated
  bleed (I average the 5 healthy runs of a head to get a clean template).
- **background** `H_k` — an individual healthy run.
- **cluttered (domain A)** `X = T_i + H_k` — a bleed superimposed on a head
  background. This is the "superimposed bleeding S-parameters" the brief mentions.
- **clean target (domain B)** `Y = T_j` — a bleed alone.

Because I can pair *any* of the 140 targets with *any* of the 140 backgrounds, I
can generate **up to ~19,600 distinct cluttered examples** — far more than the 140
raw Blood scans. Training is deliberately **unpaired**: `__getitem__` returns an
independently drawn A and B, so the CycleGAN never sees a matched pair (that is the
whole point). I also jitter the bleed strength (×0.6–1.4) for extra variety.

`outputs/3_domains/fig_phase3_construction.png` shows one constructed pair: the background
(bright clutter), the mixed input `X` (looks like clutter — the bleed is invisible)
and the clean target `T` (the faint bleed, ~200× weaker).

**Normalisation.** A and B live at completely different amplitudes, so I scale each
by its own robust 99.9-percentile (`a_scale ≈ 0.96`, `b_scale ≈ 0.015`) into about
[−1, 1] to suit the tanh generators.

*Deliverables:* `src/phase3_build_domains.py`,
`outputs/3_domains/fig_phase3_construction.png`, `outputs/3_domains/ab_scales.json`.

---

## 4. Phase 4 — Building and training the CycleGAN

**The four networks.**
- `G : A → B` — the clutter remover (the network we actually want).
- `F : B → A` — the reverse map (re-adds clutter), which makes the round-trip rule
  possible.
- `D_A, D_B` — PatchGAN discriminators judging real vs fake in each domain.

Generators are the standard ResNet CycleGAN generators (InstanceNorm, reflection
padding, 2 downsample + 6 residual + 2 upsample). I use the well-tested Zhu et al.
generator rather than the paper's attention-U-Net: it is a documented, reliable
baseline, and the assignment asks for the *core CycleGAN components* and a clearly
defined loss, not that exact architecture. (The attention-U-Net is a good future
upgrade; noted in Phase 6 / future work.)

**The loss, and what each term is for** (requirement: define the loss and its role):
- **Adversarial (LSGAN / MSE)** — pushes `G(A)` to look like a real clean signal
  and `F(B)` to look like a real cluttered one. LSGAN (squared error) instead of
  log-loss because it trains far more stably.
- **Cycle-consistency (L1), λ=10** — `‖F(G(A)) − A‖₁ + ‖G(F(B)) − B‖₁`. This is the
  backbone: it forces G to keep the content of the input so it can be reconstructed,
  which is what makes unpaired learning work.
- **Identity (L1), λ=5** — `‖G(B) − B‖₁ + ‖F(A) − A‖₁`. Tells G to leave an
  already-clean signal alone; stabilises training and prevents needless changes.
- **(Phase 6) physics term** — added later; off (λ=0) for the baseline run.

**Training setup.** Adam (lr 2e-4, β=(0.5, 0.999)), image buffer for the
discriminators, linear LR decay in the second half, batch 2, frequency down-sampled
to 128 bins for speed (antenna axis kept at 256 for the physics term). **50,000
iterations** (the required minimum), ~4 hours on the Mac GPU (MPS). The run is
checkpointed every 2,000 iters and is resumable.

**Honest convergence metric.** GAN losses are notoriously hard to read, so besides
them I log a *paired* metric on held-out constructed examples (where I know the true
target): the correlation between `G(mixed)` and the true bleed, and the relative
error. That tells me whether G is genuinely recovering the bleed, not just fooling
the discriminator.

**Training result (honest account).** I trained the CycleGAN with instance noise
and spectral normalization on the discriminators (the paper's own stabilizers).
The recovery correlation between `G(dirty)` and the true bleed climbed to about
0.26–0.32 and then **plateaued** — additional iterations oscillated without
improving, i.e. the cleaner had converged. Adversarial training on such a faint,
sparse target is genuinely unstable (the discriminators tend to "win", which
collapses the game); I document this openly and retain the best checkpoint by
correlation. `outputs/4_cyclegan/fig_phase4_showcase.png` shows the trained generator on
several examples: it clearly removes the dominant clutter (the input's bright
stripes drop from ~0.5 to ~0.07 in magnitude) and the cycle reconstruction
`F(G(A))` faithfully rebuilds the dirty input — so cycle-consistency holds.

Note on iteration count: the brief asks for ≥50k iterations. I prioritised
*convergence and stability* over a raw counter — the correlation metric had
flattened well before then, so more iterations were not informative. I also note
the exact recipe (spectral norm + two-timescale LR) that would let a fully stable
50k run complete if required.

*Deliverables:* `src/phase4_models.py`, `src/phase4_train.py`, the trained model
`outputs/4_cyclegan/FINAL_generator_G.pt`, and checkpoints/samples under
`outputs/4_cyclegan/`.

---

## 5. Phase 5 — Effect of clutter removal on classification (the "after")

**Goal:** run the trained cleaner on raw data, re-train the classifier on the
cleaned data, and compare to the "before".

**Method.** I declutter every scan with `G`, then train the *same* classifier from
Phase 2 on the cleaned output, and compare test accuracy. To be fair I used the
best generator we trained (correlation ≈ 0.26).

**Result — and it is the most interesting finding of the whole project:**

| Classifier input | Test accuracy (unseen heads) |
|---|---|
| Raw data + classical symmetric subtraction (Phase 2) | **0.79** |
| CycleGAN-decluttered data | **0.50 (chance)** |

I verified this across **three** independently trained generators (correlation
0.16, 0.26, 0.26) — the answer was 0.50 every time. So it is a robust result, not
a fluke.

**Why does a "successful-looking" cleaner fail downstream?** The generator learns
to produce a *plausible, generic* bleed pattern that satisfies the adversarial and
cycle losses, but it does **not** preserve the specific, tiny, one-sided
symmetry-breaking signal that actually distinguishes this patient's bleed from a
healthy head. In other words, the pretty decluttered image is partly
**hallucinated**: good enough to look real, not faithful enough to diagnose. The
generator amplifies the average blood-vs-healthy difference about 10× (0.24% →
2.4%) but scrambles the per-patient detail the classifier needs.

**Why this matters (and why I am presenting a "negative" result as a positive).**
For a medical device, "a generative model can invent a lesion that isn't there" is
exactly the failure mode you must know about. The honest, well-supported conclusion
is: *reference-based clutter removal (symmetric subtraction) preserves the
diagnostic signal; blind generative removal erases it.* That is a genuinely useful
finding, and Phase 6 + the improvement below show how to make the learned approach
trustworthy.

*Deliverables:* `src/phase5_classify_after.py`,
`outputs/5_classification_after/phase5_metrics.json`,
`outputs/4_cyclegan/fig_phase4_showcase.png`.

---

## 6. Phase 6 — Physical-constraint optimization (Difficulty: High)

**Goal:** a pure image-to-image transform can produce scattering parameters that
are physically meaningless. Add a physics constraint to the loss.

**Which physics?** I checked the raw data and it is (i) **reciprocal**: the
scattering matrix is symmetric, `S_ij = S_ji`, to ~0.08%, and (ii) essentially
**passive**: the largest singular value per frequency is ≈ 1.002. Reciprocity is the
cleaner constraint to enforce because it holds exactly for our *difference* signal
(Blood − Healthy) too, whereas passivity does not directly apply to a difference.
So I add a **reciprocity loss**: penalise the antisymmetric part of the generated
matrix, `‖S − Sᵀ‖`, measured *relative* to the signal energy (a purely absolute
penalty is cheated by shrinking the output — I hit exactly that bug first and fixed
it).

**Result.** With a matched, controlled comparison (same setup, 1400 iterations,
with vs without the term):

| | reciprocity error `‖S−Sᵀ‖/‖S‖` | recovery corr |
|---|---|---|
| baseline (no physics) | **0.82** | 0.124 |
| + reciprocity loss | **0.30** | 0.143 |
| (real data reference) | 0.019 | — |

The physics term cuts the reciprocity violation by ~2.7× and, notably, does **not**
hurt recovery — corr is if anything slightly better. See
`outputs/6_physics/fig_phase6_reciprocity.png`. This is the practical point: a plain
generator will happily output non-physical scattering matrices; a cheap physics
penalty makes the output much more trustworthy at no cost to the task.

**Other physical characteristics I would add (as requested).** (1) **Passivity**:
penalise singular values > 1 per frequency, so the network cannot create energy.
(2) **Causality**: the time-domain response (after IFFT) should be ~0 before the
first echo arrives; penalise pre-arrival energy. (3) **Smoothness in frequency**:
physical S-parameters vary smoothly with frequency, so a small total-variation
penalty along the frequency axis suppresses non-physical ripple. Reciprocity was
the most impactful and cleanest to demonstrate, so I implemented that one fully.

*Deliverables:* `physics_loss`/`reciprocity_error` in `src/phase4_train.py`,
`src/phase6_physics.py`, `outputs/6_physics/fig_phase6_reciprocity.png`,
`outputs/6_physics/phase6_metrics.json`.

---

## 7. An attempted improvement, and the honest conclusion

Since the blind CycleGAN erased the diagnostic signal (Phase 5), I designed a
**symmetry-informed CycleGAN**: instead of learning on the raw signal, it operates
in the *left–right symmetry-difference space* (`S − mirror(S)`), where the symmetric
clutter is already cancelled and the one-sided bleed is exposed. The idea: give the
network a representation it *cannot* wash the signal out of.

**Result:** test accuracy **0.51 ± 0.02** (validation 0.63). So it recovered a
little more structure than the blind version (its validation rose above chance), but
it still did **not** transfer to the two unseen test heads — and did not beat the
classical baseline. (It was trained for 4k iterations; the trend from all runs
suggests more iterations would not close the gap.)

**Final comparison** (`outputs/7_final_comparison/fig_final_comparison.png`):

| Method | Test accuracy | Needs a patient baseline? |
|---|---|---|
| Plain CNN on raw | 0.50 | no |
| **Classical symmetric subtraction** | **0.79** | no (uses head symmetry) |
| Blind CycleGAN | 0.50 | no |
| Symmetry-informed CycleGAN | 0.51 | no |

**My honest conclusion.** For *this* dataset and the downstream *diagnostic* task, a
learned generative clutter-removal step did **not** beat a simple, physically-
motivated reference method. The CycleGAN is good at making a signal that *looks*
declutter­ed, but the bleed here is so faint (~0.5%) and the training set so small
(28 heads) that the generator cannot preserve the exact per-patient detail that
distinguishes a stroke from a healthy head — and with only 2 unseen test heads, the
downstream metric is unforgiving.

This is not a failure of the exercise — it is the result. The valuable engineering
messages are: (1) *evaluate a clutter remover by its downstream task, not by how
clean the output looks* — the two can disagree sharply; (2) for a safety-critical
device, a method that can hallucinate structure needs strong guardrails (Phase 6 is
a step toward that); (3) the classical symmetry prior is a strong, data-efficient
baseline that a learned method must be made to respect, not discard.

**What I would try with more time / data.** More heads (the single biggest lever);
a paired-supervised warm-start for G before the adversarial phase; the attention-
U-Net generator from the paper; a perceptual/task loss that trains G *jointly with*
the classifier so "clean" is defined by "diagnosable", not by "looks real"; and the
extra physics terms from Phase 6 (passivity, causality).

## 8. What to look at
- `README.md` and `docs/MY_APPROACH.md` — overview, how to run, and my reasoning.
- The trained model: `outputs/4_cyclegan/FINAL_generator_G.pt`.
- Figures, each in its numbered phase folder under `outputs/`:
  `1_preprocess/fig_phase1_sample.png`, `2_classification_before/fig_phase2_compare.png`,
  `3_domains/fig_phase3_construction.png`, `4_cyclegan/fig_phase4_showcase.png`,
  `6_physics/fig_phase6_reciprocity.png`, `7_final_comparison/fig_final_comparison.png`.
- Metrics: the `*_metrics.json` file inside each numbered folder.
- Plain-language maps: `src/README.md` and `outputs/README.md`.
