# 02 — Notebook consolidation plan

**Status:** proposal (plan only; nothing deleted now).

## The plan

**Keep ONE notebook set in `tutorials/` = 7 notebooks:**

1. `00_quickstart` — already in `tutorials/`
2. `track_optimization` — already in `tutorials/`; **DONE: added a "reconstruct on a sphere (JUNO)" section** — same seed→`fit_track_multistart` pipeline, only GEOM/PHYS + `detector_type='sphere'` change (the seeder is geometry-aware). Validated end-to-end: vtx ~9.7 cm, dir 0.14° on JUNO (17.5 m). Demonstrates geometry generality in the recon tutorial. (Lighter photon/iter budget for the sphere so the notebook stays runnable.)
3. `calibration_optimization` — already in `tutorials/` (the calibration mode — global optics + per-PMT QE via Schur). **Timing calibration NOT added as a notebook cell**: `lucid.fitting.calibrate_timing` is a *moment-based* API decoupled from the simulator (its own test, `tests/test_timing_calibration.py`, feeds synthetic moments — no sim-integrated example exists), so a tutorial cell would be artificial. → surface it as a docs reference instead (see `03`).
4. `data_vs_prediction` — already in `tutorials/`
5. `track_gradients` — already in `tutorials/` (this is the "loss landscapes" notebook; keep its rich content — the differentiability selling point)
6. `calibration_gradients` — **NEW, built & validated**: the calibration analog of `track_gradients`. Loss landscapes (`gradient_analysis.sweep_1d/2d` over optical params) + the **Hessian/Fisher** (`crb` → normalised-Fisher & correlation heatmaps, condition number, CRB) + **before/after** the fit (perturbed start → converge, `fit` history trajectory). Grounded entirely in real APIs; 0 errors, 4 figures. Surfaces the degeneracy (absorption↔qe = −0.72) and the CRB — content no other notebook shows.
7. `event_displays` — **MERGED + display-audited (built & validated)**: one displays notebook (absorbs the former `detector_and_material_gallery`). Part 1: SK muon three ways (2D unroll + animation + 3D discs). Part 2: the **correct display per geometry** (a thorough audit showed one display does NOT fit all): oriented **discs** for sphere/box (thresholded to the bright ring, `log_scale=True`, `show_all_sensors=False` — keeps it light), and a **3D scatter** for the IceCube **string** (DOMs are point sensors with no surface normal, so discs are wrong — see code issue below; scatter matches `viewer/string/`). + water/ice/WbLS. 0 errors, divide-warning gone, executed ~32 MB (was 83). String now lights ~300 DOMs (100 GeV horizontal, 500k photons).

**Display code follow-ups (optional, from the audit — NOT fixed, code changes out of this notebook pass):**
- `lucid/geometry/utils.py::calculate_surface_normals` has branches only for Cylinder/Sphere/Box; for `StringTelescope` it returns **zeros** → `create_disc_mesh` does `0/0` → **all-NaN discs** + the "invalid value in divide" warning. Fix: add a string branch, or (better, since DOMs have no meaningful normal) add a proper string event-display method. The notebook already works around this with a scatter.
- `lucid/visualization.py::create_detector_display` / `animate_event` raise a bare `AttributeError` (`.r`/`.H`) on non-cylinder geometries. Fix: guard on `detector_type=='cylinder'` (or `hasattr`) and raise a clear message.

**→ docs, not a notebook** (conscious call): `work_with_a_dataset` ("I ran `lucid-run-job`, now what?"). Reading/looping a v3 HDF5 batch is a code snippet → a **"Working with v3 data" reference page** in `03`, seeded from `lucid/production/notebooks/read_production_output.ipynb`. Displaying an event from it is already covered by `event_displays` and the gallery.

**NOT a tutorial → a docs runbook:** `train_siren`. It's a pure CLI cheat-sheet (`%run siren/train.py …`), not a library-seam story. Capture the workflow ordering in `docs/SIREN_TRAINING.md` (extend `SIREN_TRAINING_INPUTS.md`), pointing at `lucid-train-siren`. Don't port it as a peer tutorial.

**Delete (later, after the port + adaptation):**
- All of `notebooks/` (24) — every one imports the removed `tools.*` package; dead and redundant.
- The rest of `good_notebooks/` (28) — superseded by the `tutorials/` set.

**Production notebooks** under `lucid/production/`:
- `read_production_output.ipynb` — source for the new `work_with_a_dataset` tutorial; keep as the production-side reference (fix its `edep`→`step` comment for `main`).
- `photon_shotgun/notebooks/` (3) — keep as-is (document the photon_shotgun research tool).
- `2D_/3D_event_visualization.ipynb` — **need real cleanup, not a one-line fix**: dead `import torch` + display helpers defined *inline* (the anti-pattern the repo's own rule forbids) + overlap with the interactive `viewer/`. Options: fold the per-track overlap-coloring into `lucid.visualization` and thin them, or demote to a `figures/` script. Decide during the docs pass.

That's the consolidation. `tutorials/` must be committed to git first (currently untracked);
deletions happen after. See `05_SCRIPTS_AND_WORKFLOWS.md` for how notebooks sit alongside the
scripts/CLI.

## Out of scope (maintainer)

- CI notebook execution, nbstripout, git-history cleanup — separate future effort.
- Actually running/executing notebooks — content only for now.

---

## Optional extras (nice-to-have, NOT required)

The deep read found a handful of unique bits in the to-be-deleted notebooks. Grab them only
if/when we want to enrich the kept notebooks — none of this blocks the plan above, and per the
"general guidance, not numbers" rule we take the *demos/explanations*, not research results.

- **JUNO sphere example** — `good_notebooks/reconstruction/visualize_3D_track_optimization.ipynb`
  is the only non-cylinder recon example. Could add a short "reconstruct on a sphere" section
  to `track_optimization`.
- **"How many rays?"** — `good_notebooks/infrastructure/data_vs_pred_hit_predictions.ipynb`
  has a ray-count convergence sweep that could close out `data_vs_prediction`.
- **Performance page** — `good_notebooks/infrastructure/computational_performance_evaluation.ipynb`
  → a short `docs/PERFORMANCE.md` (drop its dead `torch` import).
- **Wavelength / timing-τ / recon-anatomy** notes exist too, but are best as short docs pages
  if wanted, not tutorials.

If we don't do these, we lose nothing essential — the 7-notebook set stands on its own.

---

## Full inventory (all notebooks, one line each)

Disposition: **KEEP** / **PORT** (→ tutorials) / **DELETE** / **DELETE\*** (delete, but has an
optional bit noted above). Deletions happen after the two ports; nothing removed now.

### `tutorials/` (5) — the surviving set
| Notebook | Disposition |
|---|---|
| `00_quickstart` | KEEP |
| `track_optimization` | KEEP |
| `calibration_optimization` | KEEP — this **is** the calibration mode (global optics + per-PMT QE via Schur) |
| `data_vs_prediction` | KEEP |
| `track_gradients` | KEEP |

### `good_notebooks/` (28) — retire after porting the 2 below
| Notebook | Disposition |
|---|---|
| `00_quickstart` | DELETE (dup of tutorials) |
| `displays/event_displays` | **PORT → tutorials** (library-only; as-is) |
| `infrastructure/train_siren` | **→ docs runbook** (`docs/SIREN_TRAINING.md`), NOT a tutorial (it's a CLI cheat-sheet) |
| `calibration/calibrate_optics` | DELETE (identical to tutorials/calibration_optimization) |
| `calibration/detector_grad_qe_convergence_multi_source` | DELETE (per-PMT QE already covered by calibrate optics) |
| `calibration/grad_param_calibration_multi_init_no_qe` | DELETE (subset of calibration) |
| `calibration/laser_source_grad_analysis` | DELETE\* (calibration loss landscape) |
| `calibration/per_sensor_tau_analysis` | DELETE\* (timing-τ study → optional docs) |
| `calibration/wavelength_calibration` | DELETE\* (wavelength scenario → optional docs; has an f-string bug) |
| `reconstruction/two_start_reconstruction` | DELETE (dup of tutorials/track_optimization) |
| `reconstruction/recon_anatomy` | DELETE\* (unique seed/basin prose) |
| `reconstruction/visualize_3D_track_optimization` | DELETE\* (only JUNO-sphere example) |
| `reconstruction/track_optimization_visualization` | DELETE (broken: missing pkl path) |
| `reconstruction/optimization_vs_variables` | DELETE (broken: missing logs/pkls) |
| `gradients/loss_landscapes` | DELETE (already promoted to track_gradients) |
| `gradients/parameter_scans_1D` | DELETE (broken `lucid.utils` import) |
| `gradients/parameter_scans_1D_v2` | DELETE\* (multi-event resolution study) |
| `gradients/parameter_scans_1D_likelihood` | DELETE\* (Nphot budget study) |
| `gradients/grad_loss_and_opt_in_2D` | DELETE |
| `gradients/grad_loss_and_opt_in_2D_likelihood` | DELETE |
| `archive/*` (6: cylinder_2D_displays, event_hit_animation, geometry_and_events_3D_visualization, tracking_opt_development, tracking_opt_development_likelihood, tracking_opt_with_gif) | DELETE (superseded; some already broken) |

### `notebooks/` (24) — all DELETE (every one imports the removed `tools.*`)
Flat (17): `3D_predictions_vs_data_like_events`, `computational_performance_evaluation`,
`data_vs_prediction_full_3D_comparison`, `detector_3D_visualization`,
`detector_params_calibration`, `grad_param_calibration_BO_Leap`,
`gradient_and_relaxation_analysis`, `gradient_and_relaxation_analysis_data_like`,
`gradient_and_relaxation_analysis_noised`\*, `grid_detector_configuration_visualization`,
`position_grid_search_validation`, `single_ring_track_optimization`,
`single_ring_track_optimization_adam`, `siren_training`, `time_validations`,
`track_optimization`, `track_optimization_video`.
`data_vs_sim/` (5): `data_track_reconstruction`, `generate_example_datasets`,
`generate_muon_data_events`, `time_distribution_comparison`\*, `track_parameter_1d_scans`.
`validation/` (2): `grid_visualization`, `test_isotropic_fwd`.
(\* = a stale-but-unique idea noted in Optional extras; safe to delete regardless.)

### Production notebooks (6) — KEEP where they are
| Notebook | Disposition |
|---|---|
| `lucid/production/notebooks/read_production_output` | KEEP + **source for new `tutorials/work_with_a_dataset`** (fix `edep`→`step` comment) |
| `lucid/production/notebooks/2D_event_visualization` | KEEP but **needs real cleanup** (dead torch + inline display helpers; or demote to `figures/` script) |
| `lucid/production/notebooks/3D_event_visualization` | KEEP but **needs real cleanup** (same) |
| `lucid/production/photon_shotgun/notebooks/01_waveform_single_event` | KEEP as-is |
| `lucid/production/photon_shotgun/notebooks/02_per_photon_hit_map` | KEEP as-is |
| `lucid/production/photon_shotgun/notebooks/03_detection_rate_scan` | KEEP as-is |

**Totals (approx):** tutorials **7** (5 existing + `event_displays` port + new `detector_and_material_gallery`)
+ production kept-in-place (`read_production_output` + 3 `photon_shotgun` + 2 viz pending cleanup)
+ `train_siren` and `work_with_a_dataset` → docs. Deleted: 24 `notebooks/` + ~26 `good_notebooks/`.
New non-notebook additions: `examples/hello_telescope.py` (string + cascade spine) — see `05`.
