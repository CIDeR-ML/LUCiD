# Contributing to LUCiD

Thanks for your interest in LUCiD. This guide covers the dev setup, how to run the tests, and
the conventions the codebase follows.

## Development setup

```bash
pip install -e .[dev]        # core + pytest, jupyter, nbmake
./scripts/download_data.sh   # example SIREN emitter + PhotonSim tables (needed to run examples/tests)
```

Extras: `[training]` (torch, for training the SIREN surrogate), `[docs]`
(mkdocs-material / mkdocs-jupyter / mkdocstrings), `[all]` (everything).

## Running the tests

From the repo root:

```bash
pytest tests/            # fast tests only (~10 s) — this is what CI runs
pytest tests/ --slow     # everything (~15-20 min): builds detectors, compiles propagators, runs sims
pytest tests/ -m slow    # slow-marked tests only
pytest tests/path/test_x.py::test_name   # a single test
```

Slow tests are gated **two ways**: `conftest.py` skips a hardcoded list of slow *files* unless
`--slow` is passed, and individual tests carry `@pytest.mark.slow`. `conftest.py` forces
`JAX_PLATFORM_NAME=cpu`, so tests are deterministic and don't require a GPU.

## Conventions

- **Units are meters, nanoseconds, MeV** throughout.
- **Keep gradients flowing.** LUCiD's whole point is end-to-end differentiability. New forward
  code must be JAX-traceable (no Python branching on traced values; use `lax`/`vmap`/`scan`), and
  should be checked with `jax.grad`/finite-difference where practical.
- **No eager imports in `lucid/__init__.py`.** Submodules import on demand so lightweight tooling
  can run without pulling in JAX. Follow this when adding top-level code.
- **Detector times** in `sensor/`/`hits/`/`step/` outputs are in the *detector frame* (per-event
  `t0` already added); truth `t0` lives in `labl/`.
- **The inside/outside rule for notebooks & examples.** Anything *reusable* — forward,
  optimizers, losses, displays, sweeps, event I/O — belongs *inside* `lucid/` as a library
  function. A notebook (`tutorials/`) or script (`examples/`) should *import and narrate* one
  workflow, not reimplement machinery. If two notebooks need the same helper, promote it into
  `lucid/`.

## Extending LUCiD

- **Add a detector geometry**: subclass the base detector, decorate it with
  `@register_detector('yourtype')` in `lucid/geometry/`, and provide the matching propagator in
  `lucid/propagation/`. `generate_detector(config)` dispatches on the config's `detector_type`.
  See `cylinder.py` / `sphere.py` / `box.py` / `string.py` for reference.
- **Add a light source**: add an emitter under `lucid/sources/` returning photon rays in the
  form `setup_event_simulator` expects (see `calibration_sources.py`, `siren_rays.py`).
- **Add a material / optical model**: add a material or QE curve under `config/materials/` /
  `config/pmt/` and reference it from a `*_physics_config.json` (each optical property
  independently chooses a scalar or a wavelength-dependent representation).

## Pull requests

- Branch from `main`; keep PRs focused.
- Run `pytest tests/` (fast) before opening a PR; note if you validated `--slow` locally.
- For new forward/physics code, include a gradient or byte-identity check where it makes sense.
- Do not commit rendered notebook outputs (`*.executed.ipynb`) or generated artifacts
  (`*.gif`, plots, data) — they are gitignored; regenerate or build them in CI.

By contributing you agree that your contributions are licensed under the project's
[Apache-2.0](LICENSE) license.
