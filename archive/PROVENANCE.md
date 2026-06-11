# Provenance — LUCiD_recon reconstruction snapshot

**Snapshot date:** 2026-06-11

This directory is a **read-only provenance snapshot** taken for the LUCiD unification
effort, so that the reconstruction work developed in `LUCiD_recon` is preserved before
any deletion / archival of sprawling worktrees.

## (a) Source

- **Source tree:** `/sdf/group/neutrino/omara/LUCiD_recon`
- **Branch:** `reconstruction-mie`
- **Commit (HEAD):** `19a282a44586a5d3a866f48c67b832d6c6e80b30`
  (`added some mie scattering lines to test pathway of gradient production`,
  Riya Shah, 2026-05-25)
- **Working tree at snapshot time:** dirty — `git status --short` reports 349
  modified/untracked entries. The reconstruction drivers captured here are the
  **uncommitted top-level working-tree files** (most of the recon work lives as
  untracked `*.py` / `*.md` in the working tree, not in committed history), so the
  commit hash alone does **not** reconstruct this snapshot — the tarball is the
  authoritative copy.

Selected dirty paths relevant to the substrate (modified vs HEAD):
`lucid/losses.py`, `lucid/simulation/photon_step.py`,
`lucid/simulation/sensor_response.py`, `lucid/sources/siren_rays.py`,
`lucid/simulation/simulator.py`, `lucid/propagation/cylinder.py`,
`lucid/detector_params.py`, `lucid/overlap.py`.

## (b) Contents of `recon_drivers_snapshot.tar.gz`

Created with (run from `LUCiD_recon`):

```
tar --exclude='data' --exclude='.git' --exclude='__pycache__' \
    --exclude='*.npz' --exclude='*_trajectories' \
    -czf .../archive/recon_drivers_snapshot.tar.gz *.py *.md
```

- **229 files** total: every top-level `*.py` reconstruction driver and every
  top-level `*.md` doc/log in `LUCiD_recon`.
- **Drivers** include: `gn_fisher_recon.py`, `recon_harness.py`, `recon_ps_setup.py`,
  and the families `loss_*`, `align_*`, `recon_*`, `plot_*`, `crb_*`, `basin_*`,
  `hess_*`, `endtoend*`, `gen_*`, `loss_t*`, `time_*`, plus standalone diagnostics
  (`gn_recon.py`, `opt_lab.py`, `adfd_hessian.py`, `brute_force_charge.py`, etc.).
- **Docs** include: `RECO_PIPELINE.md`, `RECON_OPT_PLAN.md`, `DEGENERACY_FINDINGS.md`,
  `METHODOLOGY.md`, `RECIPE.md`, `COMPONENTS.md`, `LOSS_LOG.md`, `LOSS_PLAN.md`,
  `ALIGNMENT_PLAN.md`, `ALIGNMENT_FINDINGS.md`, `MAXCELL_PLAN.md`, `MISMATCH_PLAN.md`,
  `RECON_PHOTONSIM_ANALYSIS.md`, `RECON_RECIPE.md`, `RECON_SETUP.md`, `RUN_LOG.md`,
  `AUTORUN_LOG.md`, `SESSION_LOG_CHARGE_LONGITUDINAL.md`, `README.md`.

**Excluded** (by design): `data/` (large ROOT files), `.git/`, `__pycache__/`,
all `*.npz`, and `*_trajectories` dirs. The excluded SIREN net + geometry are instead
fingerprinted in `MD5SUMS.txt` (see below).

### Shared substrate (not in the tarball — fingerprinted only)

`MD5SUMS.txt` records md5 hashes of the KEY ported files and the shared substrate so
the exact versions can be re-identified later:
`gn_fisher_recon.py`, `recon_harness.py`, `recon_ps_setup.py`, `RECO_PIPELINE.md`,
`lucid/sources/siren_rays.py`, `lucid/losses.py`,
`lucid/simulation/sensor_response.py`, `lucid/simulation/photon_step.py`,
the SIREN net `data/water/muon/siren_training/trained_model/photonsim_siren_weights.npz`,
and the geometry `config/SK_geom_config.json`. All ten were present at snapshot time
(none missing).

## (c) Key findings → repro scripts (inside the tarball)

| Finding (from project memory / docs) | Repro script(s) in tarball |
|---|---|
| **Recon resolution floor** — vtx ~13 cm, dir ~1.8°, E ~2.3% (Fisher–Gauss–Newton, distance-independent bias-limited floor) | `gn_fisher_recon.py` + `RECO_PIPELINE.md` |
| **Order-statistic / first-arrival time loss** (per-PMT time term, TTS-aware) | `gn_fisher_recon.py` (per-PMT time path) |
| **SIREN cone-width bias** — SIREN cone width ~12.5° vs GEANT4 ~11.1° (and cone-mean/count comparison, track frame) | `gen_compare.py` |
| **Basin / capture maps** — capture radii and basin-of-attraction sweeps | `basin.py`, `basin_check.py`, `basin_cmp.py`, `jointbasin.py`, `temp_anneal_basin.py` |

## (d) Read-only guarantee

`LUCiD_recon` was **not modified** by this snapshot. No files were written, moved, or
committed in that tree; another person actively works there. This `archive/` copy
under `LUCiD_unification` exists purely for the unification's records.
