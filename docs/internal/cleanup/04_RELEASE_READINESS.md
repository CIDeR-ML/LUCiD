# 04 — Release-readiness plan

**Status:** proposal. Content-only; no publishing, no CI runs in this pass.

## LICENSE

The license decision is pending institutional sign-off; the LICENSE file and the
`license` / `license-files` keys in `pyproject.toml` land together as one commit once
that completes.

## `pyproject.toml` metadata (currently missing)

Distribution name is `lucid-sim` (import name `lucid`); version is `setuptools_scm`-driven.
Add:

```toml
[project]
authors = [
  {name = "Omar Alterkait", email = "omar.alterkait@tufts.edu"},
  {name = "César Jesús-Valls"}, {name = "Ryo Matsumoto"},
  {name = "Patrick de Perio"}, {name = "Kazuhiro Terao"},
]
keywords = ["differentiable-simulation", "jax", "particle-physics",
            "cherenkov", "neutrino", "calibration", "reconstruction"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "Programming Language :: Python :: 3.9",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Physics",
]
[project.urls]
Homepage = "https://github.com/CIDeR-ML/LUCiD"
Documentation = "https://cider-ml.github.io/LUCiD/"
Repository = "https://github.com/CIDeR-ML/LUCiD"
Issues = "https://github.com/CIDeR-ML/LUCiD/issues"

[project.optional-dependencies]
docs = ["mkdocs-material", "mkdocs-jupyter", "mkdocstrings[python]"]
```

**Cut a git tag** (e.g. `v0.1.0`) so `setuptools_scm` produces a real version — without a
tag there is no version string for a build/release.

## `CITATION.cff` (ready to drop in)

```yaml
cff-version: 1.2.0
title: "LUCiD: Differentiable photon simulation for optical particle detectors"
message: "If you use this software, please cite the accompanying paper."
type: software
authors:
  - {family-names: Alterkait, given-names: Omar}
  - {family-names: Jesús-Valls, given-names: César}
  - {family-names: Matsumoto, given-names: Ryo}
  - {family-names: de Perio, given-names: Patrick}
  - {family-names: Terao, given-names: Kazuhiro}
preferred-citation:
  type: article
  title: "End-to-end Differentiable Calibration and Reconstruction for Optical Particle Detectors"
  authors:
    - {family-names: Alterkait, given-names: Omar}
    - {family-names: Jesús-Valls, given-names: César}
    - {family-names: Matsumoto, given-names: Ryo}
    - {family-names: de Perio, given-names: Patrick}
    - {family-names: Terao, given-names: Kazuhiro}
  year: 2026
  journal: "arXiv preprint"
  url: "https://arxiv.org/abs/2602.24129"
  identifiers:
    - {type: other, value: "arXiv:2602.24129"}
```

## README rewrite (spec)

- **Remove** the "🚧 Under Construction" banner.
- **Remove** the stale "Notebooks" section (lists notebooks that don't exist / are being
  deleted).
- **Lead with the SIREN-only quickstart** (no PhotonSim): `pip install -e .` →
  `./scripts/download_data.sh` → `python examples/hello_simulate.py`.
- Move the PhotonSim / production-chain path into a clearly separate **"Data & Production"**
  subsection (PhotonSim is only needed for `lucid-run-job`, not simulate/calibrate/reconstruct).
- One-line pitch from the preprint abstract; link the docs site + `CITATION.cff`.
- Keep the module map (README's is code-accurate).

## `CLAUDE.md` fixes (maintainer approved update)

- **Recon description is stale** — it says a 5-stage Adam pipeline in
  `lucid/optimization/pipeline.py` (file does not exist). Replace with the real path:
  `lucid.optimization.grid_search` (seed) → `lucid.fitting.fit_track_multistart` (Fisher-GN);
  calibration via `lucid.fitting.build_calibration_problem` / `fit` / `crb`.
- **`download_data.sh` note is wrong** — it says "needs `pip install gdown`"; the script uses
  `curl` + CERNBox WebDAV. Fix the line.
- Remove any lingering `lucid/dice/` references (that project moved out to DTRAX).

## SIREN-weights install trap

`setup_event_simulator(..., particle='muon')` loads
`siren_training/trained_model/photonsim_siren_weights.npz` (not in git). So `pip install` +
run-example **silently fails** until `scripts/download_data.sh` runs.
- Make examples/`setup_event_simulator` **fail loudly** with "run `scripts/download_data.sh`"
  when weights are absent.
- Make `download_data.sh` step **explicit** in every quickstart (`QUICKSTART_LOCAL.md`
  currently omits it).

## New meta-files

| File | Priority | Note |
|------|----------|------|
| `LICENSE` | P0 | pending sign-off |
| `pyproject` metadata + tag `v0.1.0` | P0 | version/build |
| `CITATION.cff` | P1 | drafted above; GitHub renders "Cite this repository" |
| `CONTRIBUTING.md` | P1 | dev setup, `--slow` gating, inside/outside rule (harvest from `CLAUDE.md` + `good_notebooks/STRUCTURE.md`) |
| `CODE_OF_CONDUCT.md` | P2 | Contributor Covenant boilerplate |
| `CHANGELOG.md` | P2 | start at `v0.1.0` |
| `.github/ISSUE_TEMPLATE/`, `PULL_REQUEST_TEMPLATE.md` | P2 | |

## CI additions (noted; deferred — not this content-only pass)

Current CI (`.github/workflows/container.yml`) is container-only and tests nothing a
docs/notebook/pip release needs. Planned additions (ubuntu, CPU JAX, `JAX_PLATFORM_NAME=cpu`):
1. `pip-install-smoke.yml` — matrix py3.9–3.12: `pip install -e .[dev]` → run
   `hello_simulate.py` (after `download_data.sh`). Highest value (exercises the newcomer path).
2. `notebooks.yml` — `pytest --nbmake tutorials/*.ipynb` (`nbmake` already a dev dep).
3. `docs.yml` — `mkdocs build --strict` on PRs, `mkdocs gh-deploy` on `main`/tags.

## Best-practice patterns to adopt (Equinox / Diffrax / JAX-MD)

- A single end-to-end runnable "getting started" flow (our `00_quickstart`) promoted to front.
- Notebooks/examples as a first-class nav section, tiered intro → advanced.
- API reference split minimal/basic, auto-generated via `mkdocstrings` (not hand-maintained).
- "Cite this" as a standard nav item (pairs with `CITATION.cff`).
- "Open in Colab" badges on tutorials — the SIREN-only path is Colab-able (no GEANT4), giving
  a true zero-install trial.

## Path to public (priority-ordered)

**P0 (legal/build):** LICENSE (pending sign-off) · `pyproject` metadata · tag `v0.1.0` ·
examples fail loudly without SIREN weights.
**P1 (launch quality):** README rewrite · `CLAUDE.md` fixes · `mkdocs.yml` + `[docs]` extra ·
`CONTRIBUTING.md` + `CITATION.cff` · commit `tutorials/` · author the fresh Concept/Reference
pages · make `download_data.sh` explicit in quickstarts.
**P2 (polish):** CODE_OF_CONDUCT · CHANGELOG · issue/PR templates · move planning `.md` to
`docs/internal/` · Colab badges · CI jobs · (PyPI later — not now).


## Deferred for the initial public release (decided 2026-07-08, not forgotten)

- `reference/api.md` (mkdocstrings) — would pull JAX into the docs CI build.
- Retirement of `good_notebooks/` + `notebooks/` (~70 MiB) — tutorials/ is canonical.
- Heading-case + admonition style normalization across the runbooks.
- `mkdocs-redirects` for the pre-restructure URLs — old site was live ~1 week; revisit
  only if inbound 404s appear.
- Real SPICE-Lea ice optics (ice.json stays a documented water-form placeholder).
- Non-editable-install support (wheels don't ship `config/`/`data/`; documented in
  install.md).
