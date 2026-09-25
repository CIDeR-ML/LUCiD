"""Calibration campaign constants — the single source of truth for the calibration figures.

This is the calibration counterpart to ``studies.py``: it defines the detector, the truth
model, the source layout and the fit recipe **once**, and every calibration figure builds
from it. Nothing else may define a recipe value.

Ported from ``paper_figures/calibration_figures.py`` (2026-08-12). Two changes were forced
by the move and are the only intentional differences:

* the repo root is now ``parents[3]`` (the module sits at ``analysis/paper/utils/``),
* outputs route through ``utils.paths`` instead of a directory beside the script.

Consistency (referee IV.1 + II.C.3): the detector medium is the SK-calibration-paper
wavelength-dependent water model (``config/materials/water.json``) — the SAME model the
tracking section uses (``wavelength_mode=True``). Calibration is done the way real SK does
it: monochromatic lasers at several wavelengths, recovering the EFFECTIVE optical parameters
at each wavelength (L_s(λ)=1/scatter_coeff, L_a(λ)=1/absorption_coeff, QE(λ)), fit as scalars.
The recovered points trace out the water.json curves.

Not ported: the six ``compute_*``/``plot_*`` figure pairs, the 7-parameter Mie variants
(``effective_truth7``, ``_dp7``), ``_wall_sources``, ``_truth_sim``, ``_load_truth`` and
``_crb_honesty``. Those belong to calibration figures outside the current scope and are
still in ``paper_figures/calibration_figures.py`` and in the 2026-08-12 backup. Also not
ported: ``compute_conv`` / ``_mgpu_joint_fit`` / ``conv_combine`` — the older in-repo joint
fit, superseded by ``calib_run.py`` (see CALIBRATION_STUDIES.md).
"""
import os
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_ROOT = str(Path(__file__).resolve().parents[3])          # <repo>/
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector                        # noqa: E402
from lucid.simulation import setup_event_simulator                  # noqa: E402
from lucid.sources import laser_source, isotropic_source, isotropic_source_random  # noqa: E402
from lucid.detector_params import DetectorParams                    # noqa: E402
from lucid.wavelength.medium import make_medium, load_qe_curve      # noqa: E402

from analysis.paper.utils import paths                              # noqa: E402

OUT = str(paths.OUTPUT_ROOT / 'calibration')
os.makedirs(OUT, exist_ok=True)

GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
QE_PATH = os.path.join(_ROOT, 'config/pmt/SK_QE.json')

# The SAME wavelength-dependent water model tracking uses (materials/water.json).
_WLG = jnp.linspace(280, 650, 371)
_MED = make_medium('water', wavelength_grid=_WLG)
_QE = load_qe_curve(QE_PATH)

WALL_R, SENSOR_R = 0.2, 0.2      # wavelength-independent reflectivities (scalar figures)
# Realistic specular/diffuse-mixture reflection truth (literature-anchored, ~400 nm):
#   SK inner-wall black PET sheet: total R~0.05, ~55% specular (Abe et al. NIM A737 (2014)=arXiv:1307.0162
#     Fig.32; WCSim UNIFIED groundfrontpainted ~0.50 specular+0.20 backscatter+0.30 diffuse).
#   PMT face from water (glass+bialkali photocathode): total R~0.25, specular-dominated (Motta & Schonert
#     NIM A539 (2005) 217; WCSim rgcff=0.32 / SKDETSIM ~0.21). f_s<1 keeps it a genuine mixture (micro-roughness).
WALL_R_MIX, SENSOR_R_MIX = 0.05, 0.25
WALL_FSPEC, SENSOR_FSPEC = 0.55, 0.90    # specular fraction (1-fspec diffuse)
WAVELENGTHS = [337, 375, 405, 445, 473]  # SK water-calibration laser wavelengths (arXiv:1307.0162)
REP_WL = 405                             # representative wavelength

# INTENSITY is the essential fix (was 1e6 -> 100x under-photoned): sets the photon/charge
# scale and hence the CRB, at zero compute cost. Rays/K/grid drive RUNTIME, so keep them lean:
# K=8 is the >99.8%-intensity convergence criterion, and the fit is self-consistent with truth
# at the same settings, so 1e6 rays is plenty once the intensity (shot-noise) is realistic.
N_PH = int(float(os.environ.get('N_PH', '1e6')))
K = int(os.environ.get('K', '8'))
INTEN = float(os.environ.get('INTEN', '1e8'))
GRID = dict(n_cap=int(os.environ.get('NCAP', '100')),
            n_angular=int(os.environ.get('NANG', '150')),
            n_height=int(os.environ.get('NHGT', '100')))

_det = generate_detector(GEOM)
NS = len(_det.all_points)
TOP, BOT, R = _det.H / 2 - 0.1, -_det.H / 2 + 0.1, _det.r
PTS = jnp.asarray(_det.all_points)

# Realistic per-PMT gain/QE spread baked into the wavelength-scan truth: a fixed ~5% log-normal
# (physical PMT manufacturing/aging spread), gauged to mean(log k)=0 so it doesn't collide with
# the base-QE amplitude. The fit recovers ALL NS gains simultaneously by PROFILING
# them in closed form (`profile_gains`, k = SQ/SM under this same mean(log k)=0 gauge) --
# NOT through a Schur or `bake_k` block: `bake_k` is retired and CalibrationProblem needs no
# Schur block in the fitter. The gains are still all recovered; the mechanism named was stale
# alongside the per-λ optics + reflection mixture -- the honest, fully-loaded calibration.
QE_SPREAD = 0.05
_tk = np.exp(np.random.default_rng(12345).normal(0.0, QE_SPREAD, NS))
TRUTH_K = _tk / np.exp(np.mean(np.log(_tk)))


# ----------------------------------------------------------------------------------------
# The published fit recipe. Defined ONCE here; calib_run.recipe_to_kwargs translates it
# into the typed arguments the fit takes.
# ----------------------------------------------------------------------------------------
# Provenance: this is the knob set of the campaign's `steps600.sh` driver (not in this
# repository), behind the paper's convergence figure. It is also recorded inside every run's own .npz, so a
# saved run is self-describing and can be checked against this table.
#
# Why each choice — see CALIBRATION_STUDIES.md for the measurements behind them:
#   LOSS=neyman     the only member of the chi-square family LINEAR in the noisy forward
#                   model M, hence the only one with an unbiased fixed point. M is redrawn
#                   every step, so any nonlinearity (log M, sqrt M, M in the weight)
#                   permanently displaces the fixed point.
#   REFL=rlogit     fit [log R, logit f]; recovers R_s^diff that the direct spec/diff log
#                   basis leaves stuck through anti-correlation.
#   GAINS=profiled  the ~NS per-PMT gains are nuisance, profiled in closed form each step.
#   JKEY_SEED=1     decorrelates the Jacobian key stream per seed. With the default 0 the
#                   stream is IDENTICAL in every seed, so Jacobian-noise displacement shows
#                   up as an apparent bias the ensemble s.e.m. cannot see.
#   SOLVER=solve    plain damped solve (reconstruction's structure). The legacy eigen-floor
#                   was measured never to engage: 0 of 19 directions, margin 3.7-15x.
#   MU=0.1          load-bearing. MU=0.3 over-damps the soft reflection directions
#                   (Rs_diff reaches +11.2% with a sign flip); MU=0.03 is mixed.
#   LAM=0.01        measured INERT (LAM=0 is indistinguishable from baseline), retained
#                   because Marquardt would matter if the curvature spread changed.
#   STEPS=600       buys DIFFUSE reflection convergence and nothing else: every other
#                   parameter is settled by ~300, while the diffuse pair is still at 3.0 pp
#                   seed spread at 400 and only reaches ~1.2 pp by 500-600.
#   POLYAK=50       tail-average window. NOTE: the readout optimum is nearer W=200 (2.87%
#                   vs 3.43% mean worst-parameter error). This is a READOUT ONLY and can be
#                   recomputed from the saved `hist` without re-running.
#
# COMPLETENESS IS THE POINT, not tidiness. The convergence figure no longer builds an
# environment at all — `calib_run.recipe_to_kwargs` translates this dict into typed arguments —
# but the recipe is still the single description of the published run, and the 2-D figure DOES
# still build a subprocess environment from LANDSCAPE_RECIPE below.
#
# This named 22 of the 38 knobs the run read across the campaign's engine and this module; the other 16
# were inherited from whatever happened to be exported. HCORR, METRIC, TAU and GAUGE change the
# ESTIMATOR rather than a cost, and INTEN sets the photon budget of every source — an exported
# `HCORR=1` or `INTEN=1e6` would have silently published a different run. Each is now set to the
# reading site's own default, so completing it is inert on a clean shell.
CALIB_RECIPE = {
    # --- estimator ---
    'SAMPLE_TRUTH': '0', 'COMPUTE_CRB': '0',
    'REFL': 'rlogit', 'LOSS': 'neyman', 'GAINS': 'profiled', 'GAUGE': 'log',
    'METRIC': 'neyman', 'TAU': '0', 'QFLOOR_FRAC': '0.01',
    'HCORR': '0', 'HCORR_MIN': '0.03', 'HFREEZE': '0',
    'SOLVER': 'solve',
    # --- start and truth ---
    'PERT': '0.4', 'PERT_LS': '2.0', 'PERT_REFL': '1.2',
    'TRUTH_RANDOM_ISO': '0', 'FIXED_FWD_SEED': '',
    # TRUTH_NPH tracks N_PH: the published run generates its truth at the same photon count it
    # fits at, so the two must move together.
    'TRUTH_NPH': '1e6',
    # --- damping and step ---
    'LAM': '0.01', 'MU': '0.1', 'STEP_MAX': '0.5', 'FLOOR': '1e-5',
    # --- cost ---
    # INTEN is read by THIS module, not by the engine, so a scan of the engine alone misses it.
    # It sets the photon budget of every calibration source: an exported 1e6 under-photons the
    # published run by 100x, silently. NCAP/NANG/NHGT build GRID, which reaches the simulator only
    # when GRID_MANUAL=1, but are pinned for the same reason.
    'INTEN': '1e8', 'NCAP': '100', 'NANG': '150', 'NHGT': '100',
    'K': '12', 'N_PH': '1e6', 'BTRUTH': '8', 'NB_RES': '1', 'NBH': '8',
    'STEPS': '600', 'REFRESH': '20', 'POLYAK': '50', 'JKEY_SEED': '1',
    'DIFF_HIGH': '0', 'GRID_MANUAL': '0',
}
CALIB_SEEDS = (0, 1, 2)
POLYAK_PLOT_WINDOW = 150     # the window the FIGURE smooths with (see calib_plots.py)

# The 2D loss-geometry figure. Surface and trajectories MUST share K / N_PH / NK_*.
LANDSCAPE_RECIPE = {
    'NGRID': '51', 'HWF': '0.55', 'K': '8', 'N_PH': '2e6',
    'NK_TRUTH': '8', 'NK_SIM': '4',
    # Read by THIS module, not by the 2-D scripts, and therefore missed by a scan of them. The
    # leak is worse here than on the convergence path: `_cal_sim` passes **GRID UNCONDITIONALLY,
    # with no GRID_MANUAL gate, so NCAP/NANG/NHGT reach the simulator; and the 2-D scripts call
    # `_laser_sources()` with no intensity, so INTEN sets the photon budget of the source that
    # builds the surface. An exported INTEN=1e6 under-photons the published landscape 100-fold,
    # which moves the surface height, the Neyman weight 1/Q, the noise floor and the realized
    # minimum the figure marks. Values are the reading site's own defaults, so pinning them is
    # inert on a clean shell.
    'INTEN': '1e8', 'NCAP': '100', 'NANG': '150', 'NHGT': '100',
}
LANDSCAPE_SHARDS = 10
# Required, not optional: JAX's default 0.75 leaves ~8.4 GiB of an 11.26 GiB card against a
# 9.51 GiB peak, and every shard OOMs.
LANDSCAPE_ENV = {'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95'}


def effective_truth(wl):
    """Effective scalar optical params at wavelength `wl` from the water.json model."""
    sc = float(jnp.interp(float(wl), _WLG, _MED.scatter_coeff))
    ac = float(jnp.interp(float(wl), _WLG, _MED.absorption_coeff))
    return dict(scatter_length=1.0 / sc, absorption_length=1.0 / ac,
                wall_reflection_rate=WALL_R, sensor_reflection_rate=SENSOR_R,
                qe=float(_QE(float(wl))))


def _dp(truth, qe_corr=None):
    # mie disabled (folded into the effective total scatter_length); scalar wavelength mode.
    return DetectorParams.from_flat(
        scatter_length=truth['scatter_length'], mie_scatter_length=1e6, g=0.9,
        wall_reflection_rate=truth['wall_reflection_rate'],
        sensor_reflection_rate=truth['sensor_reflection_rate'],
        absorption_length=truth['absorption_length'], qe=truth['qe'],
        qe_corrections=jnp.ones(NS) if qe_corr is None else jnp.asarray(qe_corr))


def _cal_sim(nph=None, dp_default=None, reflection_model='scalar_mix',
             deposit_leg_bound=False):
    # `deposit_leg_bound` reaches the deposit regardless of `temperature=None`: the bound is on
    # the ray PARAMETER, not on the overlap width, so the hard-step calibration path is exposed
    # to it exactly like the soft reconstruction path. Default False keeps the published
    # calibration bit-identical.
    kw = dict(temperature=None, K=K, is_calibration=True, hit_mode='aggregated',
              wavelength_mode=False, reflection_model=reflection_model,
              deposit_leg_bound=deposit_leg_bound, **GRID)
    if dp_default is not None:
        kw['default_detector_params'] = dp_default
    return setup_event_simulator(GEOM, nph or N_PH, **kw)


def _laser_sources(random_iso=False, intensity=None):
    """Canonical calibration layout: ONE laser + 7 isotropic sources.

    random_iso=False (default): isotropic emission is a deterministic Fibonacci LATTICE that
      IGNORES the PRNG key -- low variance, and what the forward model should use.
    random_iso=True: isotropic emission is key-dependent. Use for TRUTH generation, so that
      truth and forward differ in emission as well as transport. With the lattice, the direct
      component (~half the charge at K=8) is identical between truth and forward for every key,
      so its residual is zero by construction rather than by fit quality.
    """
    iso = isotropic_source_random if random_iso else isotropic_source
    s = 10.0
    iso_pos = [[0, 0, 0], [s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]]
    # `intensity` is injectable because a driver that runs IN-PROCESS cannot override a
    # module-level environment read the way a subprocess environment can: by the time it could
    # set INTEN, this module is already imported. Defaults to INTEN, so the subprocess engine is
    # unchanged.
    inten = INTEN if intensity is None else float(intensity)
    return ([laser_source(position=[0, 0, TOP], direction=[0, 0, -1], intensity=inten)]
            + [iso(position=p, intensity=inten) for p in iso_pos])
