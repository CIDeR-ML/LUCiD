# 01 — Repo hygiene plan

**Status:** proposal (propose-first). No files are deleted or moved by this document.
All git operations below are to be reviewed and run on a dedicated branch.

## Branch setup

```bash
# bring local main up to the fetched origin/main (fast-forward only; non-destructive)
git checkout main
git merge --ff-only origin/main          # 023a96f -> b376809 (includes edep->step)
git checkout -b chore/cleanup            # all hygiene work happens here
```

Rationale: the `edep`→`step` rename and recent fixes live on `origin/main`; the current
`feature/two-boundary-nested-sphere` branch is 6 behind. Cleanup must target `main` so docs
and code agree on the `step` modality. In-flight branches are out of scope (maintainer will
delete most later) — do not merge or depend on them.

## Top-level directory dispositions

| Dir | Tracked? | Disposition | Notes |
|-----|----------|-------------|-------|
| `lucid/` | yes | **keep** | core package |
| `tests/` | yes | **keep** | |
| `examples/` | yes | **keep + promote** | `hello_*` scripts anchor the docs guides |
| `config/` | yes | **keep** | geometry/physics JSON, PMT npz, materials, QE |
| `container/` | yes | **keep** | |
| `docs/` | yes | **keep + restructure** | see `03_DOCS_SITE.md` |
| `tutorials/` | **untracked** | **track + keep** | the one canonical notebook set (`02`) |
| `viewer/` | yes | **keep** | self-contained web viewer (stdlib + Three.js, reads v3 HDF5); no `lucid`/scratch imports |
| `scripts/` | partial | **keep top-level, remove `campaign_recon/`** | keep `download_data.sh`, `campaign/` TBD |
| `s3df_jobs/` | partial | **keep** | cluster job templates |
| `ci_tests/` | yes | **keep** | `speed_test.py` backs the docs "Performance" page |
| `figures/` | partial | **keep whitelisted images only** | 2 images are gitignore-whitelisted |
| `good_notebooks/` | yes | **delete later** (after harvest) | see `02`; not now |
| `notebooks/` | yes | **delete later** | stale PyTorch attic; see `02`; not now |
| `studies/` | yes | **KEEP — see exception below** | ⚠️ active + code-referenced |
| `hessian_probe/` | yes* | **remove** | *tracked despite `.gitignore`; stale research |
| `baseline_scripts/` | yes | **remove** | refactor regression scaffolding; no shipped imports |
| `scripts/campaign_recon/` | yes* | **remove** | *tracked despite `.gitignore`; historical campaign |
| `mie_hunter/` | untracked | **remove from disk** | 4625-file research sandbox |
| `spatial_overlap_integrals/` | untracked | **disposable** | regenerable JSON, gitignored |
| `archive/` | yes | **keep or move to `docs/internal/`** | provenance snapshot |

### ⚠️ Exception: `studies/` — do NOT remove (needs confirmation)

The maintainer said "remove all those" but asked to look more thoroughly first. Finding:
`studies/` is an **active** two-boundary/TIR/Fresnel/conservation validation suite (newest
file 2026-06-23) and is **referenced by shipped code**:
`lucid/propagation/base.py` docstring points to `studies/sphere_detection_conservation.py`.
No shipped code *imports* it, but deleting it orphans that reference and loses validation.

**Decision (maintainer, 2026-06-30): KEEP `studies/` for now** — leave in place, do not delete.

## Tracked-vs-gitignored contradiction (fix)

`hessian_probe/` and `scripts/campaign_recon/` are listed in `.gitignore` yet committed
(the ignore rules were added after they were tracked). Since both are slated for removal,
the removal resolves it. If any were to be kept, use `git rm --cached` instead.

```bash
# planned removals (review before running)
git rm -r hessian_probe/ baseline_scripts/ scripts/campaign_recon/
rm -rf mie_hunter/ spatial_overlap_integrals/ studies/out/   # untracked / disposable on disk
# then clean the now-redundant .gitignore lines for hessian_probe/ and scripts/campaign_recon/
```

## Verify-before-delete checklist (knowledge already in code?)

The removed research dirs contain findings that should already be reflected in code. Confirm
before deleting so no live decision is lost (grep on `chore/cleanup`):

- [ ] `custom_vjp` / `make_photon_iteration_update_factors_safe` — `hessian_probe/FINDINGS.md`
      recommends dropping the custom_vjp wrapper (NaNs handled by eps-inside-sqrt). Confirm
      current state in `lucid/simulation/`.
- [ ] `jacfwd` usage — findings recommend jacfwd for Jacobian/Hessian (unblocked once
      custom_vjp is gone). Confirm in `lucid/fitting/`.
- [ ] `cherenkov_photon_norm=1.66` — `scripts/campaign_recon/REVAL_RESULTS.md` discovered the
      new SIREN emitter is 1.66× low; confirm the norm is applied in `setup_event_simulator`
      / config so removing the campaign dir loses nothing.

These are **confirmation checks**, not blockers — the maintainer does not want research
numbers preserved, and no shipped code imports any removed dir.

## `.gitignore` updates

- Remove now-redundant lines: `hessian_probe/`, `scripts/campaign_recon/`.
- Keep `mie_hunter/` off disk (it is untracked; add an explicit `mie_hunter/` ignore only if
  the folder will linger during transition).
- Existing extension-based ignores (`*.pkl`, `*.h5`, `*.png`, `*.npy`, `*.root`, `*.html`)
  already cover most regenerable outputs — keep.

## Notes

- **`mie_hunter/` `.md` findings are NOT salvaged** (maintainer: remove mie_hunter; docs are
  general guidance, not results).
- Nothing here executes; this is the reviewed script for the hygiene PR.
