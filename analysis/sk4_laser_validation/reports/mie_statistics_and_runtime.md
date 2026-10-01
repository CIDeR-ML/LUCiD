# Mie-only high-statistics and float32-fix runtime checks

## Higher-statistics TOF comparison

The original comparison used 5,000 SKDetSim events and eight million LUCiD
rays. Because only about 0.4% of rays Mie scatter before reaching the detector,
and only about 0.2% of detected light reaches the barrel, the B3 and B4 peaks
were particularly sensitive to Monte Carlo statistics.

The new comparison increases both samples by a factor of four:

- SKDetSim: 20,000 events and 13,205,707 accepted photoelectrons.
- LUCiD: 32 independent batches of one million rays.
- Water absorption, Rayleigh scattering, and reflection remain disabled.
- Both codes use the SK-IV `SGMIES` Mie phase function.
- The comparison remains SK accepted PE versus LUCiD average response.

The statistical standard errors use independent SK events and independent
LUCiD batches. The LUCiD batch boundaries are recovered exactly from the saved
per-batch total charges and cumulative deposition weights.

| Fixed TOF bin | Original residual | Original pull | High-stat residual | High-stat pull | Residual contraction |
|---|---:|---:|---:|---:|---:|
| B3, 1370–1380 ns | 2.261e-5 | 3.17 | 7.388e-6 | 2.02 | 0.327 |
| B4, 1410–1420 ns | 3.344e-5 | 3.55 | 1.139e-5 | 2.57 | 0.341 |

The combined statistical errors contracted by 0.512 in B3 and 0.471 in B4,
close to the expected factor of one half from quadrupling both samples. The
actual peak residuals contracted more strongly, by about two thirds. This is
evidence that much of the original peak difference was statistical.

The B4 maxima now fall in the same 10 ns bin. The B3 curve maxima remain in
adjacent bins, but the top is broad and the fixed-bin discrepancy is only
2.02 standard deviations. The remaining B3/B4 differences are worth tracking,
but this sample does not establish a systematic timing-shape error.

The detector-wide TOF TV changed from 0.001159 to 0.001218. That total is
dominated by the bright bottom cap and is not a useful measure of whether the
small B3/B4 peak fluctuations contracted. The section-specific and peak-bin
tests above answer that question directly.

## Runtime effect of the stable float32 PMT intersection

The benchmark compares the exact commit before the stability fix (`a9672cb`)
with the fix commit (`a268c31`). Both use identical geometry, source, PMT
response, wall assignment, 500,000 rays per batch, and four propagation steps.
Each version was run twice in alternating order. The first compilation batch
was excluded, leaving 16 warmed batches per version.

| Version | Median seconds / 500k rays | Throughput |
|---|---:|---:|
| Before fix (`a9672cb`) | 2.326 | 214,923 rays/s |
| Stable local solve (`a268c31`) | 3.465 | 144,287 rays/s |

The stable intersection therefore increases warmed runtime by **49.0%** and
reduces throughput by **32.9%** in this SK laser configuration.

The cost is not primarily the coordinate translation itself. The old torus
intersection performed 16 Newton iterations from a detector-scale origin. The
new algorithm rebases at the local GEANT PMT boundary, samples 16 points to
find the physical torus branch, performs 8 bisection iterations, and then uses
8 Newton refinements. It also preserves the locally computed hit position so
it is not reconstructed from a tens-of-metres float32 baseline.

The old version predicted about 626 PE/pulse in this benchmark while the fixed
version predicted about 718 PE/pulse, confirming that the faster result came
with missed PMT intersections. Reverting to the old solver would recover speed
by restoring the geometry error. A later optimization should preserve the
bounded local solve while reducing its fixed bracket/bisection work.

## Selected-PMT forward optimization

The wall-routing mode already makes a discrete SK-like choice of one PMT from
the detector grid cell before applying the curved photocathode decision. The
implementation nevertheless evaluated the full sphere-plus-torus intersection
for all eight candidate PMTs and masked seven results afterward. The hard
forward path now gathers the wall-selected PMT first and evaluates only that
intersection. It retains a length-one candidate axis, so the remaining photon
transport and hit accumulation code use the same array convention.

This optimization leaves the conservative float32 intersection unchanged: 16
bracket samples, 8 bisection iterations, 8 Newton iterations, local rebasing,
physical torus-branch cuts, and residual checks are all retained. The
finite-temperature calibration path also still evaluates every candidate so
its smooth all-candidate surrogate supplies the same forward- and reverse-mode
gradients.

On the same machine and fixed-seed 500,000-ray, four-step configuration:

| Forward path | Median seconds / 500k rays | Throughput |
|---|---:|---:|
| Stable solve, intersect then mask 8 candidates | 3.435 | 145,554 rays/s |
| Stable solve, intersect selected PMT only | 0.734 | 681,548 rays/s |

The selected-PMT path is **4.68 times faster**, a **78.6% runtime reduction**.
The representative Mie-only configuration took 0.751 s per 500,000 rays, or
665,389 rays/s after compilation.

The reordered tensor calculation changes float32 rounding for a tiny number of
grazing cases. In 4.5 million fixed-seed propagated rays, 2 of 1,829,573
accepted deposition records changed and total charge shifted by 1.13e-6
relative. PMT geometry, surface cuts, response functions, water physics, and
random seeds are unchanged. This is numerical boundary-level variation rather
than a change to the modeled physics.

## Artifacts

- `results_mie_only_20000_lucid32m/canonical_tof.png`
- `results_mie_only_20000_lucid32m/b3_b4_peak_statistics.png`
- `results_mie_only_20000_lucid32m/b3_b4_peak_uncertainties.json`
- `results_mie_only_20000_lucid32m/float32_runtime_benchmark.png`
- `results_mie_only_20000_lucid32m/float32_runtime_benchmark.json`
