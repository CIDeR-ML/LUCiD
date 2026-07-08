# LUCiD cleanup & public-release plan — Overview

**Status:** proposal (propose-first; nothing is executed by these documents).
**Author of plan:** janitorial pass, 2026-06-30.
**Scope:** consolidate notebooks to one set, build a general hosted docs site, tidy the
repo for a public release. **This is planning only** — no deletions, no code execution.
Each linked file below is a self-contained plan for one concern.

> These plan files live in `docs/cleanup/` and are themselves temporary — they are
> internal planning artifacts, not part of the published docs site, and should be
> deleted (or moved to `docs/internal/`) once the work lands.

## The plan set

| File | Concern |
|------|---------|
| [`01_REPO_HYGIENE.md`](01_REPO_HYGIENE.md) | directory dispositions, tracked-vs-ignored fixes, removals, gitignore |
| [`02_NOTEBOOK_CONSOLIDATION.md`](02_NOTEBOOK_CONSOLIDATION.md) | one canonical notebook set; harvest/delete plan |
| [`03_DOCS_SITE.md`](03_DOCS_SITE.md) | hosted-docs information architecture, `mkdocs.yml`, page-by-page disposition, source-verified API |
| [`04_RELEASE_READINESS.md`](04_RELEASE_READINESS.md) | LICENSE, `pyproject` metadata, `CITATION.cff`, README/CLAUDE.md fixes, CI, path-to-public |
| [`05_SCRIPTS_AND_WORKFLOWS.md`](05_SCRIPTS_AND_WORKFLOWS.md) | how notebooks/scripts/CLI divide the work; public-vs-internal scripts; the 6 canonical workflows |

## Goals (from the maintainer)

- **Going public** — this is a real public release, not just internal cleanup.
- **Audience: external-first, but also detailed for internal.** Docs read for a newcomer
  physicist / ML researcher who wants to model their detector, while still being complete
  enough for collaborators.
- **General guidance, not results.** Docs explain *how the framework works and how to use
  it* — **no research numbers, CRB/resolution tables, or `mie_hunter` findings** on the
  public site.
- Backed by the preprint: **"End-to-end Differentiable Calibration and Reconstruction for
  Optical Particle Detectors"**, Alterkait, Jesús-Valls, Matsumoto, de Perio, Terao,
  arXiv:2602.24129 (hep-ex, 2026). The site's one-line pitch should echo the abstract:
  *the first end-to-end differentiable optical particle detector simulator, unifying
  simulation, calibration, and tracking through gradient-based optimization, adaptable to
  diverse detector geometries and target materials.*

## Decisions register (confirmed with maintainer)

| # | Decision | Answer |
|---|----------|--------|
| Going public | yes | ✅ |
| Audience | external-first, detailed for internal | ✅ |
| Docs scope | general guidance only; no results/numbers | ✅ |
| PyPI | not for now (GitHub + hosted docs) | ✅ |
| Docs hosting | GitHub Pages under `CIDeR-ML/LUCiD` (same repo) | ✅ |
| LICENSE | pending institutional sign-off | ⏳ |
| `studies/` | keep for now (active + code-referenced) | ✅ |
| Branch | fresh `chore/cleanup` off updated `main` (fast-forward local `main` first) | ✅ |
| In-flight branches | most to be deleted later; do not depend on / touch them | ✅ |
| two-boundary & SIREN branches | land as separate PRs later; out of this scope | ✅ |
| Notebooks | one set → `tutorials/`; delete `good_notebooks/`+`notebooks/` **later**, plan now | ✅ |
| Executed-notebook / git-history hygiene (nbstripout) | separate future effort; **out of scope** | ✅ |
| API reference | yes, minimal curated | ✅ |
| Aspirational/planning `.md` docs | consolidate/remove during docs pass | ✅ |
| Update `CLAUDE.md` | yes | ✅ |
| Remove `mie_hunter/`, `hessian_probe/`, `baseline_scripts/`, `scripts/campaign_recon/` | yes | ✅ |
| Keep `viewer/` | yes (confirmed self-contained) | ✅ |
| Execution style | write thorough plan in multiple files | ✅ (this) |
| Deletion authority | propose-first | ✅ |
| Running/verification (execute notebooks, mkdocs build) | not now — content only | ✅ |

## Decisions locked (2026-06-30)

1. **LICENSE** — pending institutional sign-off; see `04_RELEASE_READINESS.md`.
2. **`studies/` = KEEP for now** — active validation suite + referenced by
   `lucid/propagation/base.py`; do not delete. See `01_REPO_HYGIENE.md`.
3. **Docs hosting = GitHub Pages under `CIDeR-ML/LUCiD`** (same repo, no custom domain).

## Ground-truth corrections baked into this plan

Established by reading source (not the stale docs):

- **Recon is not a 5-stage Adam pipeline.** `optimization/pipeline.py` does not exist. Real
  path = `lucid.optimization.grid_search` (seed) → `lucid.fitting.fit_track_multistart`
  (Fisher-Gauss-Newton). `CLAUDE.md` is stale on this and must be fixed.
- **Two docs describe code that does not exist** — `PRACTICE_ARCHITECTURE.md` and
  `CALIBRATION_FRAMEWORK.md` (`ParamRegistry`, `Source.emit()`, `ResponseModel`,
  `lucid/calibration/`, `fitting/registry.py` — all absent). Do **not** ship them as current.
- **`DETECTOR_PARAMS_VS_ARGS.md` is accurate** — reuse it.
- **`edep`→`step` modality rename is real on `origin/main`** (`V3_SUBDIRS = sensor, hits,
  step, labl`) but **not on the current feature branch** (still `edep`). Docs target `main`,
  so the schema page says `step/` (with `edep` retained as the per-segment energy-deposit
  *column*). This is why cleanup happens on a branch off updated `main`.

## Suggested phasing (each phase = its own PR)

1. **Hygiene** (`01`) — branch off updated `main`; fix tracked-vs-ignored; plan removals.
2. **Notebooks** (`02`) — commit `tutorials/`; harvest prose; (deletions deferred per maintainer).
3. **Docs site** (`03`) — `mkdocs.yml` + evergreen docs + author fresh Concepts/Reference.
4. **Release readiness** (`04`) — LICENSE, metadata, CITATION, README/CLAUDE fixes, CI.
