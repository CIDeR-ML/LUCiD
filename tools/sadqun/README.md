# SadQun — Supplementary Accessory of Doom for fitQun

The one piece of fiTQun that LUCiD cannot do without, and nothing else.

## Why this exists

Four of fiTQun's five tuning inputs are pure LUCiD products: the Cherenkov
profile, the charge PDF, the angular response and the indirect-light tables.
The fifth, the direct time PDF, is not. Its histogram is binned in
`log10(mu)`, where `mu` is *fiTQun's own* predicted charge at each PMT, and a
table binned in anything else sits on a different axis than the one fiTQun
evaluates at reconstruction time.

Producing it therefore needs a working fiTQun in the middle of the chain --
which drags in WCSim, the SK libraries, CERNLIB and a bootstrap file that the
stage itself is supposed to produce. SadQun is the alternative: the ~750 lines
of fiTQun that actually compute that `mu`, lifted out and given a LUCiD-shaped
interface.

## What it is, precisely

`mu` for a single-ring hypothesis, the quantity `fiTQun::Get1Rmudist` returns:

    mu[icab] = PMTQE[icab] * fact * (j0*I0 + j1*I1 + j2*I2)

the three-term expansion of the Cherenkov emission profile over the track,
weighted by solid angle, attenuation and the photosensor's angular response.

**Direct light only.** `makehistWCSim.cc` runs with `SetScatflg(0)`, so the
scattered term `muscat` is identically zero there and the scattering table is
never consulted. SadQun therefore does not implement it -- see "Deliberate
omissions".

## What it is not

Not a reconstruction. There is no fit, no Minuit, no multi-ring, no time
likelihood. SadQun evaluates fiTQun's *forward* charge model at track
parameters you hand it, which for the time-PDF stage are the true ones.

## Deliberate omissions

| omitted | why |
|---|---|
| scattered charge (`muscat`) | gated by `SetScatflg`, which the time-PDF stage sets to 0 |
| the fitted-profile branch of `EvalIn` | our profiles are raw (`UseFitCProfile=0`), so `InterpolateIn` is the live path |
| fitting, multi-ring, time likelihood | not needed to produce `mu` |
| WCSim / SK geometry ingestion | geometry comes from LUCiD's `sk_geometry.npz` |
| the `tpdfpar` bootstrap load | `LoadProfiles` demands it, `mu` never reads it |

That last row is the whole reason SadQun is smaller than fiTQun: fiTQun
refuses to start without a time-PDF parameter file, *the file this stage
exists to produce*, even though the charge prediction never touches it.

## Provenance and the risk you are taking

The physics is lifted from fiTQun, not reimplemented, so it starts out
identical. It does not track upstream. If fiTQun's `SnglTrk` changes, SadQun
silently computes a slightly different `mu` and the resulting time PDF sits on
a slightly wrong axis, with nothing to catch it.

`tests/` therefore includes a comparison harness: given a real fiTQun, it
checks SadQun's `mu` against `Get1Rmudist` on the same events, PMT by PMT.
**Run it whenever a fiTQun becomes available.** Until then SadQun is
self-consistent, not verified against the thing it copies.
