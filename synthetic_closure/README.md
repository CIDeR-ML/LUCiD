# Minimal JUNO synthetic closure

This directory contains the review workflow for a deliberately restricted
JUNO optical calibration model.  The same model generates synthetic PEs and
evaluates every fit and validation forward.

## Physics and statistical contract

The retained detector model contains:

- LS--Acrylic--Water nested spherical media;
- Snell refraction, Fresnel reflection, and total internal reflection;
- LS absorption and scattering;
- the measured JUNO large-PMT layout and PMT front surface;
- a fixed-position, fixed-intensity, fixed-time isotropic point source.

The forward prediction is PMT-by-time-cell expected PE mass.  An identity
detector response leaves those masses unchanged.  A Gaussian interpolation
kernel then defines a continuous intensity at exact synthetic PE timestamps.
The kernel is a numerical interpolation device, not a physical TTS or
electronics response.

Synthetic events use the fixed contract

```text
event exposure = 1
event time offset = 0 ns
background rate = 0
```

The likelihood is the continuous unbinned Poisson point-process likelihood.
No rate floor is added.  A predicted zero rate at an observed PE is a hard
support failure and produces a diagnostic example.

Only two quantities are fitted:

```text
log(L_scat,LS), log(L_abs,LS)
```

Both are bounded in physical space by `config/JUNO_laser_config.json`.
Optimization uses Adam with fixed transport keys.  Independent transport keys
provide MC validation.  An optional centered AD--FD audit reuses the same keys
for plus and minus evaluations.

## Excluded scope

This review branch does not contain the comparison workflow, support rods,
chimneys, acrylic bowls, diffuser emission or hardware, real FSMP input,
measured PMT IRFs, Fisher analysis, variable exposure, variable event-time
offsets, background components, or alternative model profiles.

## Local verification

Run the focused CPU tests:

```bash
make -C synthetic_closure test
```

Run the one-source CPU smoke test in a new output directory:

```bash
make -C synthetic_closure minimal-smoke \
    OUTDIR=/tmp/lucid-minimal-smoke
```

The smoke target uses one event, 128 proposal photons, `K=4`, one Adam step,
common data/fit/validation transport keys, and a truth start.  It is a plumbing
regression, not a physics-quality closure.

## Five-source GPU workflow

The supported source set is

```text
(0, 0, 0.0) m
(0, 0, 9.0) m
(0, 0, 15.0) m
(0, 0, 16.1) m
(0, 0, 17.1) m
```

Always inspect the exact command first:

```bash
make -n -C synthetic_closure submit-minimal-five \
    OUTDIR=/junofs/C/liukainan/LUCiD/minimal_closure/five_sources
```

After reviewing the output directory, source list, budgets, and Slurm
resources, submission is performed explicitly with the same command without
`-n`.  A dry-run never submits a job.

The worker generates one HDF5 observation fragment per source in an isolated
Python process, then starts a fresh fit process.  This releases the ROCm
context between data generation and fitting and keeps each source's stable
absolute index.  The `per-source` objective evaluates independent one-source
forward/VJP modules and sums their unnormalized PPP losses and gradients on
the host before dividing once by the global PE count.

The submission target refuses the default output directory and refuses to
overwrite an existing `closure_result.json`.

## Outputs

- `generation/source_XX/closure_observations.h5`: event-wise synthetic PEs
  and truth time-basis masses for one source;
- `generation/source_XX/generation_result.json`: source-generation status;
- `closure_observations.h5`: validated merged fit input;
- `closure_result.json`: configuration, truth/start/fit values, gradients,
  AD--FD audit, MC validation, support checks, and timing;
- `closure_diagnostics.pdf` and `closure_diagnostics.json`: review report;
- `checkpoint_joint.npz`: optional recovery snapshot;
- `closure.log`: execution log.

Every JSON/HDF5 result records enough configuration and seed information to
reproduce the corresponding calculation.  Raw detector/configuration inputs
are read-only.
