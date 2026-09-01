# My approach to this test

This document is my story of how I did the test, from the very beginning to the end.
I wrote it the way I actually think, which is in frames. I take a big thing, I put a
simple frame around it, I say how many pieces it has, and then I explain inside the
frame. I kept the language plain on purpose, because I would rather be clear than
sound clever.

If you want the formal write-up with all the numbers and the per-phase detail, that is
in `SUBMISSION.md`. This file is the shorter human version, so you can follow my
reasoning.

---

## The whole test in one frame

The way I see it, this whole test has two big parts, and that is all.

Part one, understand the problem and see how hard it really is.
Part two, build the CycleGAN cleaner, and then honestly measure if it helps.

Everything I did fits in one of those two parts. So let me go through them.

---

## Part one, understand the problem

The company wants to detect a brain bleed from a helmet with 16 antennas. The antennas
send weak microwaves into the head and listen to the echoes. The echoes are stored as
S-parameters, which are complex numbers, one per send and listen pair, across many
frequencies. So one measurement is a stack of 16 by 16 complex grids over frequency,
with shape 1001 by 16 by 16.

The first thing I did was not to code a model, it was to measure how hard the problem
is. And the answer was humbling. The bleed signal is about 0.4 to 0.6 percent of the
scan, so it is roughly 250 times weaker than the skull clutter. The dirty scan and the
healthy scan look identical to the eye. So the whole game is hearing a whisper inside a
loud room.

Then I asked the natural question, can a normal classifier read the raw data. I trained
a plain CNN, and it sat at 0.50 on the two unseen test heads, which is pure chance.
That made sense to me, because the bleed is tiny, every head is different, and there
are only about 240 training scans, so the network memorises the training heads and has
nothing steady for a new head.

So I needed a physical idea, not a bigger network. Here is the idea that cracked it. A
bleed sits on one side of the head, so it breaks the head's left to right symmetry,
while the skull clutter is roughly symmetric. So if I compare a scan to its mirror and
look at what does not cancel, the symmetric skull mostly disappears and the one-sided
bleed stays. I found the right mirror by asking which flip makes the healthy scans most
symmetric, and it came out cleanly as antenna k pairing with the opposite antenna. A
small MLP on this symmetry feature reached 0.79, well above the 0.70 the test asked for.

The lesson from part one, the signal is real but buried, and removing the common skull
clutter is what unlocks it. That set up the real question for part two, can a learned
model remove that clutter by itself, instead of me hand-designing the symmetry trick.

---

## Part two, build the cleaner and measure it honestly

The paper's method is a self-supervised CycleGAN, so I built it properly.

First I built the two piles of data. Domain A is the dirty scan, blood plus skull.
Domain B is the clean target, which I get by subtracting the same head's healthy scan,
so the skull cancels and mostly the blood is left. I only use this subtraction to make
a training target, because in a real clinic you do not have a healthy scan of the same
patient. By mixing any bleed with any head background I could make about 19600 dirty
examples, which is a lot of training data from few raw files.

Then I built the CycleGAN itself. There are four networks, G that cleans, F that
re-dirties, and two discriminators that judge real against fake in each domain. The
losses are the adversarial loss to look real, the cycle loss to make the round trip
come back to the start, and the identity loss to leave clean things alone.

Here I hit my first real problem, and I think the way I handled it is the most useful
thing to show. My honest quality check was to build each dirty example from a known
bleed, so I could measure whether the cleaned output actually matched the true bleed.
It did not. The output looked clean, but its correlation with the true bleed was near
zero. When I thought about why, it clicked, all the standard losses are averages over
mostly empty pixels, so the network can satisfy them with a generic clean-looking
output that ignores the specific bleed. So I added a content anchor, a loss that pushes
the cleaned output to match the true bleed's shape. And I used cosine similarity, not
plain L1, because the target is mostly zero and L1 is minimised by just outputting
zeros, which is useless. After that, the correlation started rising.

I also hit the classic GAN pain, the training kept blowing up to NaN. I could see it
in the curves, the discriminators became too strong right before every crash. I fixed
it with two standard tools, instance noise on the discriminator inputs, and spectral
normalization to cap how sharp the discriminators get. The second one is what the paper
uses as well. After that it was stable, and the recovery quality climbed to about 0.26
and then flattened, which means it converged.

Now the honest measurement, and this is the heart of my submission. I cleaned every
scan with the trained cleaner and ran the classifier again. The accuracy dropped to
0.50, chance, while the classical symmetry method stayed at 0.79. I did not want to
trust a single run, so I checked three independently trained cleaners, and all three
gave 0.50.

I sat with that result and asked what it really means. The cleaner makes a plausible,
generic bleed that satisfies the adversarial and cycle losses, but it does not keep the
exact faint one-sided detail that tells this patient's bleed from a healthy head. In
other words, the pretty cleaned image is partly hallucinated, it looks real, but it is
not faithful enough to diagnose. For a medical device, that is exactly the failure mode
you must know about, so I decided to present this negative result as the real finding,
because it is genuinely valuable.

I then did two more things to make the learned approach more trustworthy. For the
physics, I added a reciprocity constraint, because a real antenna array must give the
same signal from antenna i to j as from j to i, so the scattering matrix must equal its
transpose. A plain generator broke this badly, with error around 0.82, and my physics
loss brought it down to 0.30, without hurting the recovery. I even hit a small bug on
the way, my first penalty could be cheated by shrinking the output, so I switched to a
relative penalty, and that fixed it.

And for the improvement, I tried a symmetry-informed CycleGAN that works in the
symmetry-difference space, where the clutter is already cancelled and the bleed is
exposed, hoping the network could not wash the signal out. It reached 0.51, a little
better than blind on the validation heads, but it still did not beat the classical 0.79
on the two unseen test heads.

---

## What I conclude, and what I would do next

My honest conclusion is that on this small dataset, with such a faint target, a learned
generative cleaner did not beat a simple physics-based reference. The CycleGAN makes a
signal that looks decluttered, but it cannot preserve the tiny per-patient detail that
the diagnosis needs, and with only two unseen test heads the metric is very strict.

I am comfortable presenting this, because the value of the work is in the judgement, not
in a high score. Three things I would put forward as the real results. One, you must
judge a clutter remover by the downstream diagnostic task, not by how clean the output
looks, because the two can disagree sharply. Two, for a safety-critical device, a model
that can invent structure needs guardrails, and baking in physics like reciprocity is a
concrete step. Three, the simple symmetry prior is a strong, data-efficient baseline
that a learned method must be made to respect.

If I had more time and data, the biggest lever is clearly more heads. After that I
would warm-start the generator with a short paired-supervised phase before the
adversarial phase, use the attention U-Net generator from the paper, and most
importantly train the generator together with the classifier, so that clean is defined
as diagnosable, not as looks real. I believe that last idea is what would finally close
the gap.

---

## Where to look

- `SUBMISSION.md`, the formal write-up with all the numbers.
- `outputs/`, every result in a numbered folder, and the trained model at
  `outputs/4_cyclegan/FINAL_generator_G.pt`.
- `outputs/7_final_comparison/fig_final_comparison.png`, the four methods on one chart.
- `data/DATA_EXPLAINED.md` and `src/README.md`, if you want the data and the code in
  plain language.
