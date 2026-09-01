# Clutter suppression and target extraction from scattered data with CycleGAN

Second-round assignment for Xiong'an Anying Technology.
Author: **OUANZOUGUI Abdelhak**

The goal is to detect a brain bleed from a 16-antenna microwave helmet by removing the
dominant clutter (skull and tissue reflections) from the measured S-parameters, using a
self-supervised CycleGAN, following Lai et al. (IEEE J-ERM, 2024).

## Read these first

- **`docs/MY_APPROACH.md`** — my story of how I did the test, from start to end, in
  plain language.
- **`docs/SUBMISSION.md`** — the formal write-up, with all the numbers per phase.
- **`data/DATA_EXPLAINED.md`** — the data explained simply, with drawings.
- **`src/README.md`** and **`outputs/README.md`** — a plain map of the code and the
  results.

## The six phases (all completed)

| Phase | What | Key result |
|---|---|---|
| 1 Preprocess | `.s16p` → `[2,256,256]` real/imag images | the bleed is about 250x fainter than the clutter |
| 2 Classify (before) | bleed vs healthy on dirty data | **0.79** (symmetry feature) vs 0.50 (plain CNN) |
| 3 Build A/B | dirty = bleed+background, clean = bleed | about 19,600 unpaired examples |
| 4 CycleGAN | G, F, D_A, D_B + full loss | trained, clutter removed, cycle holds |
| 5 Classify (after) | re-classify decluttered data | **0.50** (blind CycleGAN) vs 0.79 (classical) |
| 6 Physics | reciprocity loss | reciprocity error 0.82 → **0.30**, recovery unharmed |

## The main finding

A blind CycleGAN removes clutter but partly hallucinates the clean signal, so the
downstream classification drops to chance (0.50), while a simple classical
symmetric-subtraction method keeps the diagnostic signal (0.79). For a medical device,
that is the key safety message: a clutter remover must be judged by the diagnostic task,
not by how clean the output looks. A cheap physics (reciprocity) loss then makes the
generator's output far more physically valid at no cost to recovery.

## The trained model

The deliverable, the trained clutter-removing generator G, is at:

```
outputs/4_cyclegan/FINAL_generator_G.pt
```

## How to run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# put the dataset back at data/CST-New-300-16Q-1216-Final_split/ first

./.venv/bin/python src/phase1_preprocess.py        # Phase 1: cache images + scales
./.venv/bin/python src/phase2_symmetry_mlp.py       # Phase 2: symmetry classifier (0.79)
./.venv/bin/python src/phase3_build_domains.py      # Phase 3: build/inspect A and B
./run_until_done.sh                                 # Phase 4: train the CycleGAN
./.venv/bin/python src/phase5_classify_after.py     # Phase 5: classify cleaned data (0.50)
./.venv/bin/python src/phase6_physics.py            # Phase 6: physics comparison
```

## Repository layout

```
src/         all the code, one file per phase (see src/README.md)
outputs/     all the results, numbered per phase (see outputs/README.md)
docs/        MY_APPROACH.md, SUBMISSION.md
data/        DATA_EXPLAINED.md (the raw dataset is local only, too large for GitHub)
```

The raw dataset (about 2.7 GB) and the preprocessing cache are not pushed to GitHub;
they are regenerated locally by `src/phase1_preprocess.py`.
