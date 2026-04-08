# LUCiD Refactor — Remaining Open Questions

Answered questions have been moved to design decisions in `CODEBASE_ANALYSIS.md` section 7.
These remaining questions are not blockers — they can be resolved during implementation.

---

## Physics & Simulation

1. **The `epsilon = 1e-4` offset after surface hits** scales with detector size per the comment. Should this be made configurable per-detector in the geometry config, or is it fine as a constant?

2. **Rayleigh vs Mie scattering:** Is Rayleigh-phase-function sufficient for all target detector media, or is Mie scattering support needed for future use cases?

3. **Dual reflection validation:** The wall=diffuse, sensor=specular model is a modeling choice. Is validation against Geant4/data planned before publication?

---

## Likelihood Losses

4. **`TAU_TIME = 0.15`** is hardcoded in the combined loss. How sensitive is reconstruction to this value? Should it be configurable or also learned?

5. **The 3-term combined loss formula** gates gradients asymmetrically via `stop_gradient`. What is the derivation/intuition behind this specific formulation?

6. **tau_vtx extrapolation:** The parametrization was fitted on Nrays 50k-250k, E 500-1500 MeV. Is extrapolation outside this range a concern?

7. **No-hit likelihood:** Should non-hit sensors contribute a "no-hit" likelihood term based on expected photon counts, or is masking them out correct?

8. **Vertex loss uses `stop_gradient(position)`** — only gives gradients to `t0`, not position. Is this the intended final behavior?

---

## Geometry

9. **`ConnectionTable_SK5.root`** — where does this file live? Committed, downloaded, or S3DF-only?

10. **`z_boundary = 18.0 m`** for SuperK barrel/cap classification — is this the correct official value?

11. **SuperK uses cylinder propagator.** Do real PMT positions ever fall outside cylinder bounds?

12. **Future geometries:** Any plans beyond Cylinder, Sphere, Box, SuperK?

---

## SIREN

13. **Model versioning:** How do you track which SIREN model version was used for a given reconstruction?

14. **SIREN output:** What does the model predict — photon density per solid angle, total count, or something else? What are input/output units?

---

## Data & Config

15. **`SK_physics_config.json`** (10,773 lines for uniform qe_corrections). Should this use a compact representation?

16. **Google Drive data hosting** — reliable long-term, or should data move to Zenodo/S3DF?

17. **ROOT file I/O** — core requirement, or could it be abstracted for users without ROOT data?

---

## Implementation Notes

18. **`get_isotropic_rays` Fibonacci spiral** — the comment says it "avoids float32 precision issues." What was the specific issue with random sampling?

19. **`solve_rayleigh_inverse_cdf` (Cardano's formula)** — has it been tested for numerical stability across all `u in [0,1]`?

20. **Notebook migration:** 11 notebooks still use old tuple API. These should be migrated as part of the refactor or immediately after.
