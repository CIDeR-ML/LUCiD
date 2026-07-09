"""Shared run profile for the LUCiD tutorials.

Every tutorial imports ``PROFILE`` from here instead of hard-coding photon
counts, grid resolution, and iteration budgets. This lets the same notebook
run on a **laptop GPU** (the default — sized to fit ~2.5 GB of VRAM) or at full
fidelity on a **big GPU / cluster**, by scaling the cost knobs only — the
detector, physics, and narrative are identical in both modes.

Select the mode with an environment variable (default ``laptop``)::

    LUCID_TUTORIAL_MODE=full  jupyter lab      # full fidelity (big GPU / cluster)
    jupyter lab                                # laptop-GPU mode (VRAM-safe)

or from Python before importing, e.g. ``os.environ['LUCID_TUTORIAL_MODE']='full'``.

Two families of cost knobs
--------------------------
* **Per-evaluation cost** — ``n_photons*`` and the emission ``grid``: how heavy a
  single forward simulation is.
* **Number of evaluations** — ``scan_1d``/``scan_2d`` (loss/gradient landscape
  sweeps), ``recon_niters``/``recon_nkeys`` (Gauss-Newton track reconstruction),
  ``fit_steps`` (calibration optimizer), ``grad_iters`` (simple gradient-descent
  demos). On a CPU these dominate: a 31x31 loss sweep is 961 forward+backward
  passes, and the reconstruction recipe defaults to 8 starts x 150 iterations.
  Laptop mode shrinks *both* families.

Correctness floor
-----------------
``K_ice`` is a **floor, not a knob** — it stays at 15 in both modes. Ice scatters
strongly; a smaller K silently drops a large fraction of the scatter weight (a
correctness bug, not a speed-up). Water/WbLS forwards use ``K_water``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    mode: str

    # --- per-evaluation cost: photon budgets ---
    n_photons: int          # a single forward sim (event display, track recon/gradients)
    n_photons_cal: int      # calibration: large enough for the Fisher/CRB story
    n_photons_string: int   # neutrino-telescope / ice (scatters strongly)
    n_photons_gallery: int  # the multi-geometry display gallery (many small sims)

    # --- per-evaluation cost: SIREN emission grid (n_cap x n_angular x n_height) ---
    n_cap: int
    n_angular: int
    n_height: int

    # --- scatter iterations ---
    K_water: int            # water / WbLS forward
    K_ice: int = 15         # ICE FLOOR — do not scale (see module docstring)

    # --- number of evaluations (each is a full-detector value_and_grad, ~seconds on CPU) ---
    scan_1d: int = 41       # points per 1-D loss/gradient sweep
    scan_2d: int = 31       # points per axis in a 2-D loss/gradient sweep (cost ~ scan_2d**2)
    recon_niters: int = 150 # Gauss-Newton reconstruction iterations (matches DEFAULT_RECIPE)
    recon_nkeys: int = 8    # reconstruction multi-start keys (matches DEFAULT_RECIPE)
    fit_steps: int = 150    # calibration optimizer steps
    grad_iters: int = 120   # simple gradient-descent demo loops
    nkeys: int = 4          # random keys for light averaging (energy-charge scans, etc.)
    # seed-search resolution for track reconstruction (hierarchical grid/energy scans)
    seed_levels: int = 6    # position grid-search refinement levels
    seed_ndiv: int = 5      # divisions per axis per level (cost ~ seed_ndiv**3 per level)
    seed_energy_pts: int = 12  # points in the energy pre-scan

    @property
    def grid(self) -> dict:
        """Kwargs for ``setup_event_simulator(..., **P.grid)``."""
        return dict(n_cap=self.n_cap, n_angular=self.n_angular, n_height=self.n_height)

    @property
    def is_laptop(self) -> bool:
        return self.mode == "laptop"

    def describe(self) -> None:
        note = "laptop GPU, ~2.5 GB VRAM" if self.is_laptop else "full fidelity; big GPU / cluster"
        print(
            f"tutorial profile: mode={self.mode}  ({note})\n"
            f"  photons: sim={self.n_photons:,} cal={self.n_photons_cal:,} string={self.n_photons_string:,}"
            f"  grid={self.n_cap}x{self.n_angular}x{self.n_height}  K_water={self.K_water} K_ice={self.K_ice}\n"
            f"  sweeps: 1d={self.scan_1d} 2d={self.scan_2d}  recon: {self.recon_nkeys}start x {self.recon_niters}it"
            f"  cal_fit={self.fit_steps} steps\n"
            f"  (set LUCID_TUTORIAL_MODE=full for full fidelity)"
        )


_PROFILES = {
    # Laptop: CPU-friendly. Shrinks per-eval cost AND eval counts so every
    # notebook finishes in a few minutes on a laptop.
    "laptop": Profile(
        mode="laptop",
        n_photons=80_000,        # recon Fisher intermediate ~ n_photons; keeps the two-detector track recon in-VRAM
        n_photons_cal=150_000,
        n_photons_string=150_000,
        n_photons_gallery=80_000,
        n_cap=80, n_angular=120, n_height=80,
        K_water=6, K_ice=15,
        scan_1d=25, scan_2d=15,
        recon_niters=80, recon_nkeys=2,
        fit_steps=100, grad_iters=60, nkeys=3,
        seed_levels=4, seed_ndiv=4, seed_energy_pts=8,
    ),
    # Full: the settings behind the published figures. GPU recommended.
    # recon_niters/recon_nkeys match lucid.fitting.sweep.DEFAULT_RECIPE.
    "full": Profile(
        mode="full",
        n_photons=250_000,
        n_photons_cal=1_000_000,
        n_photons_string=500_000,
        n_photons_gallery=60_000,
        n_cap=100, n_angular=150, n_height=100,
        K_water=8, K_ice=15,
        scan_1d=41, scan_2d=31,
        recon_niters=150, recon_nkeys=8,
        fit_steps=150, grad_iters=120, nkeys=4,
        seed_levels=6, seed_ndiv=5, seed_energy_pts=12,
    ),
}


def get_profile(mode: str | None = None) -> Profile:
    """Return the run profile for ``mode`` (or ``$LUCID_TUTORIAL_MODE``, default ``laptop``)."""
    mode = (mode or os.environ.get("LUCID_TUTORIAL_MODE", "laptop")).lower()
    if mode not in _PROFILES:
        raise ValueError(f"unknown LUCID_TUTORIAL_MODE={mode!r}; choose 'laptop' or 'full'")
    return _PROFILES[mode]


PROFILE = get_profile()
