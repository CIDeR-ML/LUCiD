# SK-IV laser validation against SKDetSim

This directory preserves the reproducible analysis used to bring LUCiD's
405 nm laser simulation into close agreement with SKDetSim. The implementation
it validates lives on the same `feature/sk4-laser-validation` branch. Raw SKDetSim
products, SK lookup tables, and large LUCiD caches are intentionally excluded
from Git; the
scripts, configuration choices, technical reports, and compact JSON results
are retained here.

## What is preserved

- `run_lucid.py` runs the controlled LUCiD laser samples, records the exact
  configuration and input hashes, retains per-deposition average-response
  weights and batch indices, and measures warmed batch runtimes.
- `plot_canonical_tof.py` makes the seven-section Top/B1--B5/Bottom TOF plot.
  It supports detector-total shape normalization and absolute PE-per-pulse
  normalization. LUCiD is orange, dashed, and drawn above the blue SKDetSim
  accepted-PE curve.
- `plot_tof_statistical_agreement.py` computes per-bin pulls. LUCiD uncertainty
  is obtained from independent ray batches using the actual deposition
  weights and the ratio-of-means covariance term.
- `plot_mie_tof_peak_statistics.py` performs the focused B3/B4 Mie peak test.
- `plot_float32_runtime_benchmark.py` compares the stable local PMT solve with
  the earlier detector-scale float32 solve.
- `reports/complete_fix_writeup.md` describes the geometry, routing, timing,
  response, and Rayleigh fixes in technical and plain language.
- `reports/mie_statistics_and_runtime.md` records the high-statistics Mie test
  and the selected-PMT speedup.
- `reports/rayleigh_mie_combined_tof.md` records the combined-scattering test.
- `results/` contains publication-safe plots and small machine-readable
  aggregate summaries. It does not contain event/deposition arrays, SK lookup
  tables, or the compact response caches needed to regenerate the figures.
- `PRIVATE_INPUTS.md` records the boundary that must be checked before future
  public pushes.

## Physics and modeling changes represented by this branch

The comparison uses the measured connection-table PMT positions and axes, the
SK `SKDONUTS` sphere-plus-torus surface and barrel inactive band, a stable
PMT-local float32 intersection, SK wall-cell PMT routing, the tabulated local
incidence response, SK group velocity, and the SK-IV accepted-PE timing
mixture. Water transport includes the corrected rotationally symmetric
Rayleigh transform and the selectable SK-IV `SGMIES` phase law
`p(mu) = 2 mu` for `0 <= mu <= 1`.

The geometry and response operations used by fitting remain differentiable in
both forward and reverse mode away from physical piece boundaries. The hard
wall-selected forward path evaluates only the selected PMT for speed; the
finite-temperature fitting path retains the smooth multi-candidate surrogate
and its gradients.

The relevant implementation history is deliberately split into reviewable
commits, from measured geometry through the selected-PMT optimization. Use
`git log origin/main..feature/sk4-laser-validation` to inspect that sequence.

## Reproducing a LUCiD sample

The runner needs the SK connection-table geometry already configured in this
checkout, the SK per-PMT QE table, and the converted 405.5 nm local-incidence
response table. The latter can be made with
`scripts/convert_sk_pmt_absorptance.py`. For example:

```bash
python analysis/sk4_laser_validation/run_lucid.py \
  --output /path/to/lucid_rayleigh_mie_32m.npz \
  --qe-table /path/to/qetable3_0.dat \
  --pmt-detection-response /path/to/sk4_pmt_absorptance_405nm.npz \
  --rays 1000000 --batches 32 --steps 4 \
  --water-rayleigh --water-mie --mie-phase-model sk4 \
  --sensor-shape sk20inch --sensor-candidate-selection wall \
  --pmt-timing sk4 --propagation-speed 0.217289524
```

This is the average-response estimator unless `--sampled-transport` is
provided. Absorption and all reflection remain off in this example.

Given a compact SK accepted-PE response cache and the LUCiD output, regenerate
the two canonical normalizations with:

```bash
python analysis/sk4_laser_validation/plot_canonical_tof.py \
  --sk-response /path/to/sk_response.npz \
  --lucid /path/to/lucid_rayleigh_mie_32m.npz \
  --output /tmp/canonical_tof.png \
  --summary /tmp/canonical_tof.json \
  --accepted-only --title 'Rayleigh + Mie'

python analysis/sk4_laser_validation/plot_canonical_tof.py \
  --sk-response /path/to/sk_response.npz \
  --lucid /path/to/lucid_rayleigh_mie_32m.npz \
  --output /tmp/canonical_tof_pe_per_pulse.png \
  --summary /tmp/canonical_tof_pe_per_pulse.json \
  --normalization pe-per-pulse --accepted-only \
  --title 'Rayleigh + Mie'
```

## Current combined-scattering result

For 20,000 SKDetSim pulses and 32 independent one-million-ray LUCiD batches,
the detector-total-normalized seven-section TOF total-variation distance is
`0.001904`. The accepted yields are `668.2556 +/- 0.1755 PE/pulse` in
SKDetSim and `728.9511 +/- 0.1778 PE/pulse` in LUCiD. The normalized TOF shape
is therefore close, while a `9.083%` absolute-yield difference remains. The
full bin-level uncertainty results are stored in
`results/rayleigh_mie/tof_statistical_agreement.json`.

The principal public plots are:

- [Rayleigh-only normalized TOF](results/rayleigh_only/canonical_tof.png)
- [Mie-only normalized TOF](results/mie_only/canonical_tof.png)
- [Mie B3/B4 high-statistics test](results/mie_only/b3_b4_peak_statistics.png)
- [stable-intersection runtime benchmark](results/mie_only/float32_runtime_benchmark.png)
- [combined Rayleigh+Mie normalized TOF](results/rayleigh_mie/canonical_tof.png)
- [combined Rayleigh+Mie absolute PE/pulse TOF](results/rayleigh_mie/canonical_tof_pe_per_pulse.png)
- [combined Rayleigh+Mie bin-level pulls](results/rayleigh_mie/tof_statistical_agreement.png)

The retained files are sufficient to audit the equations, exact configuration,
analysis code, and numerical conclusions. Re-running requires access to the
SK collaboration inputs, which are not redistributed by this repository.
