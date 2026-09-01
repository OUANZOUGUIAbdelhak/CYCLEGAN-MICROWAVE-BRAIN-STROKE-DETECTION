# The results, explained simply

Every result lives in a numbered folder that matches the phase that made it. So if you
want to check what the code produced at each step, you open the folder with the same
number. I organised it this way on purpose, because before it was one big pile and I
could not tell which file was which, or where the final model was.

## Where is the trained model

The deliverable, the trained clutter-removing generator G, is here:

```
   outputs/4_cyclegan/FINAL_generator_G.pt
```

That single file is the network the whole project is about. It takes a dirty scan and
returns a decluttered one. The showcase and the Phase 5 evaluation both load it.

## The folders, in order

```
   1_preprocess/
       fig_phase1_sample.png     dirty scan vs healthy template vs the tiny clean bleed
       norm_stats.json           the amplitude scales for domain A and domain B

   2_classification_before/      the "before cleaning" classifier
       fig_phase2_compare.png    plain CNN (0.50) next to the symmetry method (0.79)
       fig_phase2_curve.png      the training curve
       baseline_cnn.pt           the plain CNN, and baseline_cnn_norm.npz its scaler
       baseline_metrics.json     plain CNN = 0.50, symmetry-CNN = 0.67
       symmetry_mlp.pt           the winning classifier (the 0.79 one)
       symmetry_mlp_metrics.json test = 0.79 +/- 0.02, val = 0.87
       mirror_axis.json          the mirror it found (m = 8, opposite antennas)

   3_domains/                    building domain A (dirty) and domain B (clean)
       fig_phase3_construction.png   background, mixed input, clean target side by side
       ab_scales.json                the scales used to normalise A and B

   4_cyclegan/                   Phase 4, the CycleGAN itself
       FINAL_generator_G.pt      *** the trained model, the deliverable ***
       fig_phase4_showcase.png   G removing the clutter, and the cycle coming back
       training_log.json         the full training history (losses, correlation)
       scales.json               the A and B scales the model was trained with
       samples/                  snapshots of the output every 2500 iterations

   5_classification_after/       Phase 5, the "after cleaning" classifier
       phase5_metrics.json       symmetry-MLP on cleaned data = 0.50 (dropped to chance)

   6_physics/                    Phase 6, the reciprocity physics constraint
       fig_phase6_reciprocity.png    reciprocity error going down when we add physics
       phase6_metrics.json           error 0.82 without physics, 0.30 with physics
       p6_base_best.pt               the model trained without the physics loss
       p6_phys_best.pt               the model trained with the physics loss
       p6_base_log.json, p6_phys_log.json   their training logs

   7_final_comparison/           the improvement attempt and the final table
       fig_final_comparison.png       all four methods on one bar chart
       symmetry_informed_G.pt         my symmetry-informed generator
       symmetry_informed_metrics.json test = 0.51 (val 0.63)
       symmetry_informed_train_log.json, symmetry_informed_sample.png
```

## The results in one small table

| Method | Test accuracy on the 2 unseen heads |
|---|---|
| Plain CNN on the raw data | 0.50 (chance) |
| Classical symmetry method (my baseline) | 0.79 |
| Blind CycleGAN, then classify | 0.50 |
| Symmetry-informed CycleGAN, then classify | 0.51 |

And for the physics, the reciprocity error of the generator output went from 0.82
down to 0.30 when I added the physics loss, and the recovery quality did not get worse.

## A note on the cache

There is also a `cache/` folder that Phase 1 creates. It holds the 300 files turned
into fast `.npy` images. It is not pushed to GitHub because it is large and it is
regenerated in a minute by running `src/phase1_preprocess.py`.

## A note on what is not here

Some early experiments failed and I did not keep their heavy files, because they were
large and not useful to keep. The main one was a first CycleGAN with no content anchor
and full frequency resolution, which produced generic output with almost zero
correlation to the true bleed. That failure is what led me to add the content anchor.
I describe this story in `docs/MY_APPROACH.md`.
