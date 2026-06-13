# good_notebooks — structure & the inside/outside rule

The principle that decides what is a notebook vs library code, and the target layout.

## The rule

- **INSIDE `lucid/`** = anything *reusable* or *canonical* — the **seams**: forward, optimizers,
  losses, displays, parameter sweeps, event I/O, result-analysis. If two notebooks need it, it is
  a library function. Duplicated machinery rots and hides the canonical path from new users.
- **OUTSIDE (a notebook)** = the *narrative of one workflow*: which source/event/params, the
  composition of library calls, the plots, the interpretation, the knobs to tweak. A good notebook
  is ~15 cells that **import and tell a story** — not a reimplementation.

### Why this matters (the evidence that drove it)
Across the 22 original notebooks, machinery was defined **inline**: the recon/track loss in **9**,
an Adam optimizer in **5**, bootstrap/convergence analysis in **5**, 2D/animation displays in **3**
— and `create_detector_display` was re-defined inline *even though `lucid.visualization` already
has it*. That duplication is why there were 22 notebooks and none read as canonical.

## Seam inventory (inside)

| Seam | Lives in | Status |
|------|----------|--------|
| forward / observables | `setup_event_simulator` | ✅ |
| reconstruction optimizer | `lucid.fitting.fit_track_multistart` (+ `seed_vertex_time`) | ✅ (kills 5 inline Adam) |
| calibration optimizer + CRB | `lucid.fitting.build_calibration_problem`/`fit`/`crb` | ✅ (kills inline optax) |
| **track loss** (scalar of θ) | `lucid.fitting.ReconModel.loss` (charge NLL + order-stat time) | ✅ (kills 9 inline `combined_product_loss` — *use ReconModel*, don't reassemble) |
| parameter sweeps / landscapes | `lucid.gradient_analysis` (`sweep_1d/2d`, `SweepParam`, `find_zero_crossing`) | ✅ (kills inline scan loops) |
| event I/O + padding | `lucid.sources.event_io` (`read_photon_data_from_photonsim`, `pad_photon_data`) | ✅ (kills inline `generate_event_data`) |
| 2D / 3D displays | `lucid.visualization.create_detector_display`, `geometry.visualize_event_data_plotly_discs` | ✅ (notebooks must IMPORT, not redefine) |
| **event animation** | `lucid.visualization` (event GIF) | 🔨 BUILD (3 inline copies) |
| **result analysis** | `lucid.fitting.analysis` (bootstrap CI, lon/tra residuals, containment) | 🔨 BUILD (5 inline copies) |

The core new-user journey needs **no new code** — only the two 🔨 seams (animation, analysis) are
genuinely missing, and they serve the secondary notebooks.

## Target layout (new-user journey, ~9 thin notebooks)

```
good_notebooks/
├── 00_quickstart.ipynb              simulate + DISPLAY one event            (forward + visualization)   ← first 5 min
├── reconstruction/
│   ├── reconstruct_a_track.ipynb    canonical two-start                     (fit_track_multistart)      ✅
│   └── recon_anatomy.ipynb          seeds + loss landscape + convergence    (+ fitting.analysis)        ← teaches WHY
├── calibration/
│   ├── calibrate_optics.ipynb       global + per-PMT QE                     (build_calibration_problem) ✅
│   ├── calibrate_wavelength.ipynb   λ-dependent scenario (thinned)
│   └── calibrate_per_pmt_qe.ipynb   per-PMT QE-map scenario (thinned)
├── gradients/
│   └── loss_landscapes.ipynb        1D/2D × counts/likelihood               (gradient_analysis)         ← ONE, not 5
├── displays/
│   └── event_displays.ipynb         2D + 3D + animation gallery             (visualization)
└── infrastructure/
    ├── train_siren.ipynb
    └── performance.ipynb
```

Journey: **quickstart → reconstruct → calibrate → understand (gradients) → extend (`../examples/`
+ `lucid/fitting/contracts.py`)**. Each notebook demonstrates ONE seam by calling the library.

## Disposition of the originals
The fat originals (inline Adam / inline loss / inline display) → `archive/` once their narrative is
captured by a thin canonical notebook. Kept-but-thinned: the two distinct calibration scenarios
(wavelength, per-PMT QE), train_siren, performance. The 5 gradient/scan notebooks collapse into the
single `loss_landscapes.ipynb`; the 3 display notebooks into `event_displays.ipynb`.
