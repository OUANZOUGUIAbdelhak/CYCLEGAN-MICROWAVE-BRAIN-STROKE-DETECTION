# The data, explained simply

This note explains what is inside the `data/` folder, what one file actually is,
and what shape the numbers have. I wrote it for myself first, so that I never feel
lost about the data again, and then kept it clean so anyone can read it. I tried to
keep it as simple as the main explainer, and I added little drawings so the shapes
are easy to picture.

The raw dataset is not pushed to GitHub because it is about 2.7 GB. This file is the
only thing from the `data/` folder that is tracked. If you want to run the code, put
the folder `CST-New-300-16Q-1216-Final_split/` back inside `data/`.

---

## 1. The big picture first

The company wants to find a brain bleed using a helmet with 16 small antennas around
the head. One antenna sends a very weak microwave signal into the head, and all 16
antennas listen to what comes back. Different tissues (skin, skull, brain, blood)
bounce the microwaves differently, so the echoes carry a hint about what is inside.

A brain bleed, in medical words, is a hemorrhagic stroke, which just means blood is
leaking and forming a small pool inside the head. That pool reflects microwaves a
little differently from healthy brain, and that tiny difference is the whole thing we
are trying to catch.

So the data is not images of a head. The data is the echoes, written as numbers.

```
   antenna sends  ~~~>  [   H E A D   ]  ~~~>  16 antennas listen
                          skull, brain,
                          maybe a bleed
```

---

## 2. What one number means (the S-parameter)

When antenna i sends and antenna j listens, the machine writes down one number that
says how much signal came back and how much it was delayed. That number is called an
S-parameter, and people write it as `S(j, i)`. The S is for scattering, because the
signal scatters off the tissues.

This number is a complex number. That sounds scary but it only means it has two
parts, a size and a shift, and we store it as two plain numbers:

```
   S(j,i) = real part  +  imaginary part
            (how strong)   (how shifted in time / phase)
```

We keep both parts. The paper we follow uses only the size (the magnitude), but I
decided to keep both, because the phase carries real information, because the
subtraction trick I explain below stays exact, and because the physics checks in
Phase 6 are only possible if we keep the full complex number.

---

## 3. From one number to the full measurement (the shapes)

There are 16 antennas, and every antenna can send to every antenna, so for one single
"photo" of the head we get a full 16 by 16 grid of these numbers:

```
             receiver j  ->
           +---------------------------+
   sender  |  S(1,1)  S(1,2) ... S(1,16)|
     i      |  S(2,1)  S(2,2) ... S(2,16)|
     |      |    .        .         .    |     16 x 16 = 256 numbers
     v      |  S(16,1) ...        S(16,16)|    (each one is complex)
           +---------------------------+
```

But the machine does not use only one microwave color. It sweeps across many
frequencies, from about 0.5 GHz to 2.5 GHz. A frequency is like a color of the
microwave. At every frequency we get a fresh 16 by 16 grid. So one full measurement of
one head is a stack of grids, one grid per frequency:

```
   frequency 1     ->   [16 x 16 grid]
   frequency 2     ->   [16 x 16 grid]
        .                     .
   frequency 1001  ->   [16 x 16 grid]

   so the raw shape is:   [ 1001 , 16 , 16 ]   complex numbers
                            freq   ant  ant
```

That is the raw data for one file. In the code this is loaded by
`src/common.py` in the function `load_raw_S`, which uses a small library called
`scikit-rf` to read the special file format.

---

## 4. The file format and the file names

Every file ends with `.s16p`. This is a standard format for scattering data, called
Touchstone. The 16 in `s16p` just means 16 ports, which here means 16 antennas.

The file names are not random, they carry the truth about each file, which is very
lucky because it gives us the labels for free. Two examples:

```
   BLOOD_head29_id29_x-0.84_y20.15_z0_r22.85_v50_RI.s16p
   HEALTHY_HM01_Run1_TI101_Rot_P00d00_RI.s16p
```

Reading the first one:
- `BLOOD` means this head has a bleed. The other class is `HEALTHY`.
- `head29` means it is head number 29, one specific simulated person.
- `x`, `y`, `z` say where the bleed sits inside the head, in millimetres.
- `r22.85` is the radius of the bleed, and `v50` is its volume (about 50 units).
- `RI` means the numbers are stored as Real and Imaginary.

So from the name alone I know: bleed or healthy, which head, and where the blood is.
The code that reads these names is `parse_name` in `src/common.py`.

One honest note, this data is simulated, it was made in a physics simulator called
CST, it was not measured on real patients. That is normal and good for a test,
because a simulator can produce many perfectly labelled examples. The real paper used
lab phantoms, but the idea is the same.

---

## 5. How much data there is

The folder is split like this:

```
   data/CST-New-300-16Q-1216-Final_split/
       train/
           Blood/     140 files   (bleed scans, heads 1 to 28)
           Healthy/   140 files   (healthy scans, same heads)
       test/
           Blood/      10 files   (bleed scans, heads 29 and 31)
           Healthy/    10 files   (healthy scans, heads 29 and 31)
```

So 300 files in total, that is the 300 in the folder name, and 16Q means 16 antennas.
The important thing is that the two test heads (29 and 31) never appear in training.
That is on purpose, because we want to check that the method works on a new head it
has never seen, not on a head it already memorised.

Each head has 5 healthy runs (the same head measured 5 times), and several bleed
files with different bleed positions and sizes. Having 5 healthy runs per head is
useful later, because I average them to get a clean, stable healthy template.

---

## 6. How we turn the raw data into an image the network can eat

A convolutional network likes 2D grids, like small photos. Our raw data is a stack of
complex 16 by 16 grids over frequency, so I reshape it into a 2-channel image. The
steps are in `S_to_image` inside `src/common.py`:

```
   raw S            [1001, 16, 16]  complex
     |  flatten the 16x16 grid into 256 antenna-pair rows
     v
                   [1001, 256]      complex
     |  put the antenna pairs on the vertical axis, frequency on the horizontal
     v
                   [256, 1001]      complex
     |  resample the frequency axis from 1001 down to 256, so it is a clean square
     v
                   [256, 256]       complex
     |  split the complex number into two real channels (real, imaginary)
     v
                   [2, 256, 256]    float32     <- this is one training image
                    ^   ^    ^
                    |   |    freq bins (256)
                    |   antenna pairs (16x16 = 256)
                    two channels: channel 0 = real, channel 1 = imaginary
```

So in the end, one measurement becomes a small picture of size `[2, 256, 256]`. You
can think of it as two grayscale images stacked, one for the real part and one for
the imaginary part. During training I also shrink the frequency axis further, from
256 down to 128, only to make training faster. The antenna axis stays at 256, because
Phase 6 needs to fold it back into 16 by 16 to check the physics.

---

## 7. The two domains, and the subtraction trick

This is the most important idea about the data, so I want it very clear.

We cannot physically separate the "blood echo" from the "skull echo", because they
arrive mixed together in the same measurement. But there is a clever workaround, using
the fact that we have a healthy scan of the same head:

```
   BLOOD scan      =   skull echo   +   blood echo     (mixed, what we measure)
   HEALTHY scan    =   skull echo                      (same head, no bleed)
   -----------------------------------------------------------------------
   BLOOD - HEALTHY =                    blood echo      (the skull cancels out)
```

So by subtracting the healthy scan of the same head, the big skull echo cancels, and
what is left is mostly just the blood signal. This gives us the two piles of data that
the CycleGAN uses:

- Domain A, the dirty pile: the raw BLOOD scan. It is blood plus skull, all mixed.
- Domain B, the clean target pile: BLOOD minus HEALTHY. It is mostly just the blood.

The reason we do not simply always subtract and forget the neural network, is that in
a real clinic you almost never have a healthy scan of this exact patient from before.
You get one scan of one sick patient, and that is all. So the subtraction is only used
to build a training target, and then we teach a network to do the cleaning without
needing that healthy twin. That trained network is the real deliverable.

The code that builds these two piles is `src/phase3_build_domains.py`.

---

## 8. One number that explains why this is hard

After I cached all the files, I measured how strong the clean blood signal is compared
to the dirty scan. The blood echo is about 250 times weaker than the full scan, around
0.4 to 0.6 percent of it. In other words, the dirty scan is almost all skull, and the
bleed is a tiny whisper hidden inside a loud room.

```
   dirty scan  |###############################|   (skull, very strong)
   blood only  |#|                                  (about 0.4 percent, tiny)
```

That single fact drives every decision in the project. It is why a plain classifier on
the raw data is basically guessing, and it is why we need a smart way to bring the
whisper up.

You can see all of this in the picture `outputs/1_preprocess/fig_phase1_sample.png`,
which shows a dirty scan, the healthy template, and the tiny clean bleed side by side.
