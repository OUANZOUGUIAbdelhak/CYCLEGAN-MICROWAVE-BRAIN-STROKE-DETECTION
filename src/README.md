# The code, explained simply

All the code lives here in `src/`, as flat files, one per phase, named in the order I
did them. There are no hidden subfolders, so you can read them top to bottom like a
story. When I run a phase, its results are written into the matching numbered folder
in `outputs/`, so Phase 2 code writes into `outputs/2_classification_before/`, and so
on. I kept it this way because when the folder was messy I could never find anything,
and this fixed it.

## The files, in order

```
   common.py                    the shared toolbox: paths, reading .s16p files,
                                turning them into images, the symmetry feature,
                                and the small CNN classifier. Everything else
                                imports from here.

   phase1_preprocess.py         Phase 1. Reads all 300 raw files, caches them as
                                fast .npy images, computes the normalisation scales,
                                and saves the first figure.

   phase2_baseline_cnn.py       Phase 2, the honest baseline. A plain CNN on the raw
                                data (it scores about 0.50, chance), and a
                                symmetry-CNN for contrast.

   phase2_symmetry_mlp.py       Phase 2, the winner. A small MLP on the left-right
                                symmetry feature. This is the 0.79 classifier.

   phase3_build_domains.py      Phase 3. Builds domain A (dirty) and domain B (clean)
                                by the subtraction trick, and the unpaired sampler
                                the CycleGAN trains on.

   phase4_models.py             Phase 4. The four networks: G, F, and the two
                                discriminators D_A and D_B.

   phase4_train.py              Phase 4. The training loop, with all the losses
                                (adversarial, cycle, identity, content anchor, and
                                the optional physics term).

   phase4_showcase.py           Phase 4. Makes the picture that shows G removing the
                                clutter and the cycle coming back.

   phase5_classify_after.py     Phase 5. Cleans every scan with the trained G, then
                                runs the Phase 2 classifier again, to see if cleaning
                                helped. (It drops to 0.50, which is the key finding.)

   phase6_physics.py            Phase 6. Trains with and without the reciprocity
                                physics loss and compares, to show the output becomes
                                physically valid.

   phase7_symmetry_informed.py  My improvement attempt. A CycleGAN that works in the
                                symmetry-difference space. It reaches 0.51, a little
                                better than blind but still below the classical 0.79.
```

## The one idea behind the layout

`common.py` knows where everything is. It computes the project folder from its own
location, so there are no hardcoded absolute paths, and the repo runs on any machine.
It also creates the numbered output folders. Every other file just says "put my result
in `DIR_2`" and does not worry about the details.

## How to reproduce, step by step

From the project root, with the virtual environment set up:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# put the dataset back in place first: data/CST-New-300-16Q-1216-Final_split/

./.venv/bin/python src/phase1_preprocess.py        # Phase 1: cache images + scales
./.venv/bin/python src/phase2_baseline_cnn.py       # Phase 2: plain CNN (~0.50)
./.venv/bin/python src/phase2_symmetry_mlp.py       # Phase 2: symmetry MLP (0.79)
./.venv/bin/python src/phase3_build_domains.py      # Phase 3: build/inspect A and B
./run_until_done.sh                                 # Phase 4: train the CycleGAN
./.venv/bin/python src/phase4_showcase.py           # Phase 4: the clutter-removal picture
./.venv/bin/python src/phase5_classify_after.py     # Phase 5: classify cleaned data (0.50)
./.venv/bin/python src/phase6_physics.py            # Phase 6: physics comparison
./.venv/bin/python src/phase7_symmetry_informed.py  # improvement attempt (0.51)
```

The trained generator that everything uses is already saved at
`outputs/4_cyclegan/FINAL_generator_G.pt`, so Phases 5, 6 (uses the same trainer) and
the showcase can run straight away without retraining.
