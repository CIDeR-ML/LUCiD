"""Calibration campaign constants — the single source of truth for the calibration figures.

This is the calibration counterpart to ``studies.py``: it defines the detector, the truth
model, the source layout and the fit recipe **once**, and every calibration figure builds
from it. Nothing else may define a recipe value.

The detector medium is the SK-calibration-paper wavelength-dependent water model
(``config/materials/water.json``), the SAME model the tracking section uses
(``wavelength_mode=True``). Calibration follows SK: monochromatic lasers at several
wavelengths, recovering the EFFECTIVE optical parameters at each wavelength
(L_s(λ)=1/scatter_coeff, L_a(λ)=1/absorption_coeff, QE(λ)), fit as scalars. The recovered
points trace out the water.json curves.
"""
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.wavelength.medium import make_medium, load_qe_curve

_ROOT = str(Path(__file__).resolve().parents[3])          # <repo>/

GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
QE_PATH = os.path.join(_ROOT, 'config/pmt/SK_QE.json')

# The SAME wavelength-dependent water model tracking uses (materials/water.json).
_WLG = jnp.linspace(280, 650, 371)
_MED = make_medium('water', wavelength_grid=_WLG)
_QE = load_qe_curve(QE_PATH)

# Specular/diffuse-mixture reflection truth (literature-anchored, ~400 nm):
#   SK inner-wall black PET sheet: total R~0.05, ~55% specular (Abe et al. NIM A737 (2014)=arXiv:1307.0162
#     Fig.32; WCSim UNIFIED groundfrontpainted ~0.50 specular+0.20 backscatter+0.30 diffuse).
#   PMT face from water (glass+bialkali photocathode): total R~0.25, specular-dominated (Motta & Schonert
#     NIM A539 (2005) 217; WCSim rgcff=0.32 / SKDETSIM ~0.21). f_s<1 keeps it a genuine mixture (micro-roughness).
WALL_R_MIX, SENSOR_R_MIX = 0.05, 0.25
WALL_FSPEC, SENSOR_FSPEC = 0.55, 0.90    # specular fraction (1-fspec diffuse)
WAVELENGTHS = [337, 375, 405, 445, 473]  # SK water-calibration laser wavelengths (arXiv:1307.0162)
REP_WL = 405                             # representative wavelength

_det = generate_detector(GEOM)
NS = len(_det.all_points)
TOP = _det.H / 2 - 0.1

# Realistic per-PMT gain/QE spread baked into the wavelength-scan truth: a fixed ~5% log-normal
# (physical PMT manufacturing/aging spread), gauged to mean(log k)=0 so it doesn't collide with
# the base-QE amplitude. The fit recovers ALL NS gains alongside the per-λ optics and the
# reflection mixture by PROFILING them in closed form (`profile_gains`, k = SQ/SM under this
# same mean(log k)=0 gauge).
QE_SPREAD = 0.05
_tk = np.exp(np.random.default_rng(12345).normal(0.0, QE_SPREAD, NS))
TRUTH_K = _tk / np.exp(np.mean(np.log(_tk)))


# ----------------------------------------------------------------------------------------
# The published fit recipe: the keyword arguments of `calib_run.run_seed`, defined once here
# and recorded inside every run's .npz.
# ----------------------------------------------------------------------------------------
# The estimator itself is the library's: Neyman residual (the only chi-square member linear in
# the noisy forward M, hence the only one with an unbiased fixed point when M is redrawn every
# step) with the per-PMT gains profiled in closed form. Why each recipe value:
#   basis=rlogit   fit [log R, logit f]; recovers R_s^diff, which the direct spec/diff log basis
#                  leaves stuck through anti-correlation.
#   mu=0.1         load-bearing. mu=0.3 over-damps the soft reflection directions (Rs_diff reaches
#                  +11.2% with a sign flip); mu=0.03 is mixed.
#   lam=0.01       measured inert (lam=0 is indistinguishable), retained because Marquardt would
#                  matter if the curvature spread changed.
#   steps=1200     buys DIFFUSE reflection convergence and nothing else: every other parameter is
#                  settled by ~300. With the leg-bounded deposit (C-005, F-012) the diffuse pair
#                  reaches truth only by ~1200 and then wanders at its noise floor (wall ~2%, sensor
#                  ~6%; 600 left them at 10% / 22%, 2400 adds nothing; F-028 log 2026-11-05).
#   polyak=50      tail-average window. The readout optimum is nearer 200 (2.87% vs 3.43% mean
#                  worst-parameter error); a readout only, recomputable from the saved `hist`.
#   intensity=1e8  sets the photon/charge scale at zero compute cost (1e6 would under-photon the
#                  sources 100x). n_ph/k/grid drive runtime and are kept lean: the fit is
#                  self-consistent with truth at the same settings.
#   grid           the photon-grid sizing make_grid() picks for this detector, stated so the
#                  run does not depend on that default.
#   truth_seed=0   the pseudo-data's seed, the same for every fit seed (keys: lucid.fitting.calib).
CALIB = dict(
    steps=1200, n_ph=1_000_000, k=12, grid=dict(n_cap=71, n_angular=223, n_height=76),
    intensity=1e8,
    btruth=8, n_forward_draws=1, jacobian_draws=8, refresh=20, polyak=50,
    lam=0.01, mu=0.1, max_step=0.5, q_floor_frac=0.01,
    gauge='log', basis='rlogit', mie_scatter_length=1e6, g=0.9,
    pert=0.4, pert_ls=2.0, pert_refl=1.2,
    truth_seed=0,
)
CALIB_SEEDS = (0, 1, 2)
POLYAK_PLOT_WINDOW = 150     # the window the FIGURE smooths with (see calib_plots.py)

# The 2D loss-geometry figure. The surface and the trajectories are built from this one dict, so
# they share K / N_PH / NK_* by construction; lam/mu are the joint fit's, so the Gauss-Newton
# panel shows the step the fit takes. At this setting no sensor has zero charge, so the surface is
# linear in `intensity`: it moves only the colour-bar offset.
LANDSCAPE = dict(
    ngrid=51, hwf=0.55, k=8, n_ph=2_000_000, grid=dict(n_cap=100, n_angular=150, n_height=100),
    intensity=1e8, nk_truth=8, nk_sim=4, lam=CALIB['lam'], mu=CALIB['mu'],
)
LANDSCAPE_SHARDS = 10
# Required, not optional: JAX's default 0.75 leaves ~8.4 GiB of an 11.26 GiB card against a
# 9.51 GiB peak, and every shard OOMs.
LANDSCAPE_ENV = {'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95'}


def effective_truth(wl):
    """Effective scalar optical params at wavelength `wl` from the water.json model."""
    sc = float(jnp.interp(float(wl), _WLG, _MED.scatter_coeff))
    ac = float(jnp.interp(float(wl), _WLG, _MED.absorption_coeff))
    return dict(scatter_length=1.0 / sc, absorption_length=1.0 / ac, qe=float(_QE(float(wl))))


def _cal_sim(n_ph, k, grid):
    """The calibration forward: hard-step overlap, expected first-arrival hits, scalar mode."""
    return setup_event_simulator(GEOM, n_ph, source='light_source', temperature=None, K=k,
                                 wavelength_mode=False,
                                 reflection_model='scalar_mix', **grid)


def _laser_sources(intensity):
    """Canonical calibration layout: ONE laser + 7 isotropic sources.

    Isotropic emission is a deterministic Fibonacci LATTICE that ignores the PRNG key: low
    variance, and what the forward model should use.
    """
    s = 10.0
    iso_pos = [[0, 0, 0], [s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]]
    inten = float(intensity)
    return ([laser_source(position=[0, 0, TOP], direction=[0, 0, -1], intensity=inten)]
            + [isotropic_source(position=p, intensity=inten) for p in iso_pos])
