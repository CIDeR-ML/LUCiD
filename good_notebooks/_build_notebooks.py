"""Build notebooks 1 (bias study) and 2 (optimization) from cell specs.

Run once to (re)generate the .ipynb files. Each notebook is a thin
wrapper over the scripts in good_notebooks/. Heavy logic stays in the
.py files; the notebooks drive them cell-by-cell.

Usage:
    python _build_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent


def code(src: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in src.rstrip('\n').split('\n')],
    }


def md(src: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in src.rstrip('\n').split('\n')],
    }


META = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.x',
    },
}


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        'cells': cells,
        'metadata': META,
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    path.write_text(json.dumps(nb, indent=1))


# ---------------------------------------------------------------------------
# Notebook 1 — bias_study_factored_sum.ipynb
# ---------------------------------------------------------------------------
NB1_CELLS = [
    md("""# Bias study with the factored-sum Poisson-process loss

Goal: characterize the loss landscape near truth across many events.

Pipeline (each cell calls into the scripts under `good_notebooks/`):

1. **Setup** — detector + data/pred simulators, knobs at the top.
2. **Single-event scan** — visualize loss + gradient around truth for all 7 params (one event).
3. **Multi-event bias sweep (secant)** — gradient zero-crossing offset across 50 (seed, entry) pairs.
4. **Mode comparison** — `joint` vs `factored_sum` side-by-side.
5. **Source decomposition** — total ΣN_data vs ΣN_pred, photon-source vs propagation split.
6. **Calibration check** — needed `tot_n_photons_normalization` vs current fit.

Scripts the notebook drives: `parameter_scans_1D_gaussian.py`, `multi_event_bias_joint.py`,
`sim_total_compare.py`, `photon_source_compare.py`, `distribution_compare.py`,
`siren_vs_photonsim_emission.py`, `check_total_norm.py`."""),

    md("## 1. Setup"),

    code("""import sys, time, importlib
sys.path.append('..')
import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from parameter_scans_1D_gaussian import (
    DEFAULT_JSON_FILENAME, PHYSICS_CONFIG, K_PRED, K_DATA,
    NPHOT_PRED, NPHOT_DATA, SIGMA_TTS_NS,
    make_loss_fn, make_loss_and_grad, perform_scan, _MODE_PARAMS,
)
import multi_event_bias_joint as meb
import sim_total_compare as stc
import check_total_norm as ctn
import distribution_compare as dc
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator

print('Modules loaded.')"""),

    code("""# Knobs
MODE = 'factored_sum'
HIT_THRESHOLD = 2.0
LAMBDA_BG = 1e-3
WAVELENGTH_MODE = False
N_EVENTS = 50
SEED_BASE = 44
START_ENTRY = 0
N_SCAN = 41

# Build simulators
detector = generate_detector(DEFAULT_JSON_FILENAME)
num_detectors = len(detector.all_points)

data_sim = setup_event_simulator(
    DEFAULT_JSON_FILENAME, NPHOT_DATA,
    temperature=None, K=K_DATA,
    is_data=True, is_calibration=False, apply_smearing=True,
    wavelength_mode=WAVELENGTH_MODE,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)

pred_sim = setup_event_simulator(
    DEFAULT_JSON_FILENAME, NPHOT_PRED, 0.10,
    max_sensors_per_cell=4, K=K_PRED,
    is_data=False, hit_mode='per_photon',
    wavelength_mode=WAVELENGTH_MODE,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)

loss_fn = make_loss_fn(pred_sim, num_detectors)
loss_and_grad = make_loss_and_grad(loss_fn)
grad_fn = meb.make_grad_fn(loss_fn)
print(f'Setup done. mode={MODE}, thr={HIT_THRESHOLD}, '
      f'lambda_bg={LAMBDA_BG}, wavelength_mode={WAVELENGTH_MODE}')"""),

    md("""## 2. Single-event scan — all 7 params

Pick one (seed, entry), scan each parameter 41 points around its true value,
plot loss + gradient. Useful sanity check before running the multi-event sweep."""),

    code("""ENTRY_SHOW = 0
SEED_SHOW = 44

(true_track, true_data, x, y, z, theta, phi, energy) = meb.build_event_for_entry(
    detector, data_sim, num_detectors, entry_idx=ENTRY_SHOW, seed=SEED_SHOW)
true_param_values = [x, y, z, 0.0, theta, phi, energy]
n_hit = int(np.sum(np.asarray(true_data[0]) > 0))
print(f'event seed={SEED_SHOW}, entry={ENTRY_SHOW}: '
      f'pos=({x:+.2f},{y:+.2f},{z:+.2f}), '
      f'θ={theta:+.3f}, φ={phi:+.3f}, E={energy:.0f}, n_hit={n_hit}')

# JIT warmup
hit_counts, _hit_times_true, hit_times = true_data
cw, tw, gm, jt, fac = _MODE_PARAMS[MODE]
t0 = time.time()
_ = loss_and_grad(jnp.array(true_param_values), hit_times, hit_counts,
                   jax.random.PRNGKey(42),
                   jnp.float32(cw), jnp.float32(tw),
                   jnp.float32(HIT_THRESHOLD), jnp.float32(LAMBDA_BG),
                   jnp.float32(gm), jnp.float32(jt), jnp.float32(fac))
print(f'  loss_and_grad warmup: {time.time() - t0:.1f}s')"""),

    code("""# Scan and plot
scan_specs = meb.SCAN_SPECS  # [(name, idx, range), ...]
results = []
for name, idx, scan_rng in scan_specs:
    r = perform_scan(loss_and_grad, true_param_values, true_data,
                      name, idx, scan_rng, n_points=N_SCAN,
                      mode=MODE, hit_threshold=HIT_THRESHOLD,
                      lambda_bg=LAMBDA_BG)
    i_min = int(np.argmin(r['losses']))
    zero_off, slope = meb.find_grad_zero_crossing(r['values'], r['gradients'], r['true'])
    rng_loss = float(r['losses'].max() - r['losses'].min())
    results.append((name, r, zero_off, slope, rng_loss))
    print(f'  {name:>6}: zero-grad offset = {zero_off:+.4f}, '
          f'argmin offset = {r["values"][i_min] - r["true"]:+.4f}, '
          f'range = {rng_loss:.1f}')

n = len(results)
fig, axes = plt.subplots(n, 2, figsize=(13, 2.4 * n))
for row, (name, r, zero_off, slope, rng_loss) in enumerate(results):
    ax_l, ax_g = axes[row, 0], axes[row, 1]
    ax_l.plot(r['values'] - r['true'], r['losses'] - r['losses'].min(),
              color='navy', marker='o', markersize=3, lw=1.4)
    ax_l.axvline(0.0, color='red', ls='--', lw=1.0, label='true')
    ax_l.axvline(zero_off, color='gray', ls=':', lw=1.0,
                  label=f'zero-grad ({zero_off:+.3f})')
    ax_l.set_ylabel('NLL − min'); ax_l.set_xlabel(f'{name} − true')
    ax_l.grid(alpha=0.3); ax_l.legend(fontsize=8); ax_l.set_title(f'{name}  range={rng_loss:.1f}')

    ax_g.plot(r['values'] - r['true'], r['gradients'],
               color='darkgreen', marker='o', markersize=3, lw=1.4)
    ax_g.axhline(0.0, color='gray', ls=':', lw=1.0)
    ax_g.axvline(0.0, color='red', ls='--', lw=1.0)
    ax_g.axvline(zero_off, color='gray', ls=':', lw=1.0)
    ax_g.set_ylabel(f'∂NLL/∂{name}'); ax_g.set_xlabel(f'{name} − true')
    ax_g.grid(alpha=0.3); ax_g.set_title(f'{name} gradient')

fig.suptitle(f'Single event: seed={SEED_SHOW}, entry={ENTRY_SHOW}, '
             f'mode={MODE}, thr={HIT_THRESHOLD}, λ_bg={LAMBDA_BG}, '
             f'wavelength_mode={WAVELENGTH_MODE}', fontsize=10)
plt.tight_layout()
plt.show()"""),

    md("""## 3. Multi-event bias sweep (secant)

Runs the secant solver for each parameter across 50 (seed, entry) pairs. Per
parameter we record the gradient zero-crossing offset, then aggregate across
events. The histogram shows the distribution of biases per param."""),

    code("""# Warmup grad_fn
t0 = time.time()
e_warm = jnp.zeros(7, dtype=jnp.float32).at[0].set(1.0)
cw, tw, gm, jt, fac = _MODE_PARAMS[MODE]
_ = grad_fn(jnp.float32(0.0),
              jnp.array(true_param_values, dtype=jnp.float32),
              e_warm, hit_times, hit_counts, jax.random.PRNGKey(42),
              jnp.float32(cw), jnp.float32(tw),
              jnp.float32(HIT_THRESHOLD), jnp.float32(LAMBDA_BG),
              jnp.float32(gm), jnp.float32(jt), jnp.float32(fac))
print(f'  grad_fn warmup: {time.time() - t0:.1f}s')

per_param = {name: [] for name, _, _ in scan_specs}
raw_events = []
for ev in range(N_EVENTS):
    entry_idx = START_ENTRY + ev
    seed = SEED_BASE + ev
    (true_track, true_data, x, y, z, theta, phi, energy) = meb.build_event_for_entry(
        detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
    true_pv = [x, y, z, 0.0, theta, phi, energy]
    base_params = jnp.array(true_pv, dtype=jnp.float32)
    ev_out = meb.run_one_event_secant(
        grad_fn, base_params, true_data,
        hit_threshold=HIT_THRESHOLD, lambda_bg=LAMBDA_BG, mode=MODE)
    for name, _, _ in scan_specs:
        per_param[name].append(ev_out[name])
    raw_events.append({'event_idx': ev, 'entry_idx': entry_idx, 'seed': seed,
                         'true_param_values': true_pv, 'scans': ev_out})
    if (ev + 1) % 10 == 0 or ev == 0:
        print(f'  event {ev + 1}/{N_EVENTS}')

meb.print_summary_table(per_param, MODE, HIT_THRESHOLD, LAMBDA_BG, N_EVENTS)"""),

    code("""# Histograms of zero-grad bias per parameter
per_param_zeros = {name: [e['grad_zero_offset'] for e in per_param[name]]
                     for name, _, _ in scan_specs}
out_dir = Path('figures/multi_event_bias')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'nb_bias_hist_{MODE}_thr{HIT_THRESHOLD:.1f}_n{N_EVENTS}.png'
meb.plot_bias_histograms(per_param_zeros, MODE, HIT_THRESHOLD,
                          LAMBDA_BG, N_EVENTS, out_path)
print(f'saved: {out_path}')
# Inline display
from IPython.display import Image
Image(filename=str(out_path))"""),

    md("""## 4. Mode comparison: `joint` vs `factored_sum`

Re-run the same N events under both modes for direct comparison. `joint` applies
hit_threshold to every term so it loses Poisson-zero info from unhit sensors;
`factored_sum` sums charge over all sensors. The E bias typically flips sign
between the two."""),

    code("""def run_mode(mode_name):
    per_param_m = {name: [] for name, _, _ in scan_specs}
    for ev in range(N_EVENTS):
        entry_idx = START_ENTRY + ev
        seed = SEED_BASE + ev
        (_, true_data, x, y, z, theta, phi, energy) = meb.build_event_for_entry(
            detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
        bp = jnp.array([x, y, z, 0.0, theta, phi, energy], dtype=jnp.float32)
        ev_out = meb.run_one_event_secant(
            grad_fn, bp, true_data,
            hit_threshold=HIT_THRESHOLD, lambda_bg=LAMBDA_BG, mode=mode_name)
        for name, _, _ in scan_specs:
            per_param_m[name].append(ev_out[name])
    return per_param_m

print('Running joint ...')
pp_joint = run_mode('joint')
print('Running factored_sum ...')
pp_fs = run_mode('factored_sum')

print()
print(f'{"param":>6} | {"joint mean ± SE":>20} | {"factored_sum mean ± SE":>26}')
print('-' * 60)
for name, _, _ in scan_specs:
    a = np.asarray([e['grad_zero_offset'] for e in pp_joint[name]], dtype=float)
    b = np.asarray([e['grad_zero_offset'] for e in pp_fs[name]], dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    sa = float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a) > 1 else 0.0
    sb = float(np.std(b, ddof=1)/np.sqrt(len(b))) if len(b) > 1 else 0.0
    print(f'{name:>6} | {a.mean():+10.4f} ± {sa:7.4f}    '
          f'| {b.mean():+10.4f} ± {sb:7.4f}')"""),

    md("""## 5. Source decomposition

For each event, run pred_sim and data_sim at true parameters and compare totals.
The ratio ΣN_pred / ΣN_data tells you the data/pred calibration offset that the
E bias is chasing."""),

    code("""# Use sim_total_compare's helpers
print(f'Using wavelength_mode={WAVELENGTH_MODE} for both sims (matches setup).')
ratios, sums_data, sums_pred = [], [], []
pred_key = jax.random.PRNGKey(42)
for ev in range(N_EVENTS):
    entry_idx = START_ENTRY + ev
    seed = SEED_BASE + ev
    (true_track, true_data, x, y, z, theta, phi, energy) = meb.build_event_for_entry(
        detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
    sum_d = float(np.sum(np.asarray(true_data[0])))
    out_p = pred_sim(true_track, pred_key)
    sum_p = float(np.sum(np.asarray(out_p[3])))
    ratios.append(sum_p / sum_d)
    sums_data.append(sum_d)
    sums_pred.append(sum_p)
ratios = np.asarray(ratios)
sums_data = np.asarray(sums_data)
sums_pred = np.asarray(sums_pred)
print(f'ΣN_data    : {sums_data.mean():.1f} ± {sums_data.std(ddof=1):.1f}')
print(f'ΣN_pred    : {sums_pred.mean():.1f} ± {sums_pred.std(ddof=1):.1f}')
print(f'ratio      : {ratios.mean():.4f} ± {ratios.std(ddof=1):.4f}')
implied_e_bias = 1050.0 * (1.0 / ratios - 1.0)
print(f'implied E bias = {implied_e_bias.mean():+.1f} ± '
      f'{implied_e_bias.std(ddof=1)/np.sqrt(N_EVENTS):.1f} MeV (SE of mean)')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(ratios, bins=20, color='C0', alpha=0.7, edgecolor='black')
ax[0].axvline(1.0, color='red', ls='--')
ax[0].axvline(ratios.mean(), color='darkorange', ls='-')
ax[0].set_xlabel('ΣN_pred / ΣN_data')
ax[0].set_ylabel('events')
ax[0].set_title('Pred/Data total-charge ratio per event')
ax[0].grid(alpha=0.3)
ax[1].scatter(sums_data, sums_pred, alpha=0.6)
mn, mx = min(sums_data.min(), sums_pred.min()), max(sums_data.max(), sums_pred.max())
ax[1].plot([mn, mx], [mn, mx], 'k--', lw=0.7, label='y = x')
ax[1].set_xlabel('ΣN_data'); ax[1].set_ylabel('ΣN_pred')
ax[1].set_title('Per-event ΣN_data vs ΣN_pred')
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

    md("""## 6. Calibration: needed `tot_n_photons_normalization`

Compute the needed `total_norm` at E=1050 separately for "match emission" and
"match detection". If they differ, distribution shape (not just total) matters
beyond a global gain calibration."""),

    code("""importlib.reload(ctn)
# This calls the same script function but we'll do it inline for the notebook
from lucid.utils import unpack_photonsim_params
from lucid.siren.training.inference import SIRENPredictor
from lucid.siren.core import create_photonsim_siren_grid
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.generate import read_photon_data_from_photonsim
from parameter_scans_1D_gaussian import DATA_FILE

params = unpack_photonsim_params('muon', 'water')
a, b, c = params['tot_n_photons_normalization']
sa, sb, sc = params['num_seeds']
ENERGY_REF = 1050.0
fit_total_norm = a * ENERGY_REF**b + c

predictor = SIRENPredictor(params['siren_model_path'])
grid_data = create_photonsim_siren_grid(predictor)
model_params = predictor.params
# SIREN mean weight at E_ref (deterministic for fixed key)
_, _, ws = photonsim_differentiable_get_rays(
    jnp.zeros(3), jnp.array([0., 0., 1.]), ENERGY_REF, NPHOT_PRED,
    grid_data, model_params, jax.random.PRNGKey(42), sa, sb, sc)
siren_mw = float(np.mean(np.asarray(ws)))

# Per-event needed_norm to match EMISSION and DETECTION
needed_emit, needed_det = [], []
for ev in range(N_EVENTS):
    entry_idx = START_ENTRY + ev
    seed = SEED_BASE + ev
    pd = read_photon_data_from_photonsim(DATA_FILE, entry_idx)
    n_phsim = int(len(pd['photon_origins']))
    needed_emit.append(n_phsim / siren_mw)
    needed_det.append(fit_total_norm * sums_data[ev] / sums_pred[ev])
needed_emit = np.asarray(needed_emit)
needed_det  = np.asarray(needed_det)

print(f'fit  total_norm(1050)              = {fit_total_norm:>10.2f}')
print(f'needed total_norm — match emission = {needed_emit.mean():>10.2f} ± '
      f'{needed_emit.std(ddof=1):.2f}  ({100*(fit_total_norm/needed_emit.mean()-1):+.2f} % off)')
print(f'needed total_norm — match detect.  = {needed_det.mean():>10.2f} ± '
      f'{needed_det.std(ddof=1):.2f}  ({100*(fit_total_norm/needed_det.mean()-1):+.2f} % off)')
print(f'gap (distribution-shape effect)    = {100*(needed_emit.mean()/needed_det.mean()-1):+.2f} %')"""),

    md("""## Notes

- Heavy logic lives in scripts: `parameter_scans_1D_gaussian.py`, `multi_event_bias_joint.py`,
  `sim_total_compare.py`, `distribution_compare.py`, `check_total_norm.py`.
- To compare a different loss mode, change `MODE` at the top.
- To toggle wavelength-dependent optics, change `WAVELENGTH_MODE` and rebuild simulators.
- For richer per-event detail (full scan-grid arrays per param), use `--solver scan`
  in the CLI version of `multi_event_bias_joint.py`."""),
]


# ---------------------------------------------------------------------------
# Notebook 2 — tracking_opt_factored_sum.ipynb (minimal driver)
# ---------------------------------------------------------------------------
NB2_CELLS = [
    md("""# Track-reconstruction optimization with factored-sum loss

5-stage pipeline mirroring `tracking_opt_development_likelihood.ipynb`, with
Stage 4 (Adam refinement) using the **factored-sum** loss instead of the older
3-term `√(c·t·v)` combiner. Vertex_loss + dynamic τ_vtx are dropped — not needed
with the joint formulation.

Stages 0–3 are kept as-is (cheap pre-conditioning):

- Stage 0 — energy scan at origin (`energy_loss`)
- Stage 1 — hierarchical (position, t0) grid search (geometric `origin_time_loss`)
- Stage 2 — cone direction search (`poisson_nll` on charges)
- Stage 3 — energy scan refinement (`energy_loss`)
- **Stage 4 — Adam refinement using `factored_sum`** (the change)

This notebook is intentionally lean. Cell 1 sets knobs; cell 2 imports the
helpers; cell 3 runs N events; cell 4 prints aggregate residuals. For the full
visualization/grid-search/save machinery, see the original
`tracking_opt_development_likelihood.ipynb`."""),

    md("## Setup"),

    code("""import sys, time
sys.path.append('..')
import jax, jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path

from parameter_scans_1D_gaussian import (
    DEFAULT_JSON_FILENAME, PHYSICS_CONFIG, K_PRED, K_DATA,
    NPHOT_PRED, NPHOT_DATA, SIGMA_TTS_NS,
    make_loss_fn, _MODE_PARAMS,
)
import multi_event_bias_joint as meb
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.optimization.grid_search import (
    load_optimization_config, get_detector_bounds,
    hierarchical_position_grid_search,
)
from lucid.optimization.utils.functions import (
    cartesian_to_spherical, spherical_to_cartesian,
)

# Knobs
MODE = 'factored_sum'
HIT_THRESHOLD = 2.0
LAMBDA_BG = 1e-3
WAVELENGTH_MODE = False
N_EVENTS = 10
SEED_BASE = 44
START_ENTRY = 0
MAX_ITERATIONS = 400
TOLERANCE = 1e-6
print(f'mode={MODE}, thr={HIT_THRESHOLD}, lambda_bg={LAMBDA_BG}, '
      f'wavelength_mode={WAVELENGTH_MODE}, N_EVENTS={N_EVENTS}')"""),

    code("""# Build sims
detector = generate_detector(DEFAULT_JSON_FILENAME)
detector_points = jnp.array(detector.all_points)
num_detectors = len(detector_points)
detector_bounds = get_detector_bounds(detector)

data_sim = setup_event_simulator(
    DEFAULT_JSON_FILENAME, NPHOT_DATA,
    temperature=None, K=K_DATA,
    is_data=True, is_calibration=False, apply_smearing=True,
    wavelength_mode=WAVELENGTH_MODE,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)
pred_sim = setup_event_simulator(
    DEFAULT_JSON_FILENAME, NPHOT_PRED, 0.10,
    max_sensors_per_cell=4, K=K_PRED,
    is_data=False, hit_mode='per_photon',
    wavelength_mode=WAVELENGTH_MODE,
    physics_config=PHYSICS_CONFIG, default_detector_params=True)

loss_fn = make_loss_fn(pred_sim, num_detectors)

# Stage 4 loss: factored_sum via flag setup
cw, tw, gm, jt, fac = _MODE_PARAMS[MODE]
cw_j, tw_j, gm_j, jt_j, fac_j = (jnp.float32(v) for v in (cw, tw, gm, jt, fac))
thr_j = jnp.float32(HIT_THRESHOLD)
bg_j = jnp.float32(LAMBDA_BG)

@jax.jit
def loss_and_grad_s4(params, hit_times, hit_counts, key):
    def f(p):
        return loss_fn(p, hit_times, hit_counts, key,
                       cw_j, tw_j, thr_j, bg_j, gm_j, jt_j, fac_j)
    return jax.value_and_grad(f)(params)

print('Sims + Stage-4 loss ready.')"""),

    md("""## Stages 0–3 (pre-conditioning, simple losses)

These follow the reference notebook. Energy scan at origin, hierarchical
position+t0 grid, cone direction search, energy refinement. We only need
modest accuracy here — Stage 4 polishes."""),

    code("""# Light wrappers for the reference-notebook stages. We use Poisson + energy_loss
# heuristics for stages 0–3 (consistent with tracking_opt_development_likelihood).

def energy_loss(sim_counts, true_counts, eps=1e-8):
    return jnp.abs(jnp.log(jnp.sum(sim_counts) / (jnp.sum(true_counts) + eps)))

def poisson_nll_charge(true, pred, eps=1e-8):
    nll = pred - true * jnp.log(pred + eps) + jax.scipy.special.gammaln(true + 1.0)
    return jnp.sum(nll) / (jnp.sum(true) + eps)

def stage0(observed_counts, true_energy):
    theta_i = jnp.arccos(1/jnp.sqrt(3)); phi_i = jnp.pi/4.
    pos_i = jnp.array([0., 0., 0.])
    energy_guess = 1000 + np.random.uniform(-50, 50)
    energies = jnp.linspace(energy_guess - 700, energy_guess + 700, 10)
    scan_key = jax.random.PRNGKey(42)
    best_loss, best_e = float('inf'), energy_guess
    for e in energies:
        track = ParticleParams(energy=e, position=pos_i, theta=theta_i,
                                phi=phi_i, t0=jnp.array(0.0))
        _, _, _, q = pred_sim(track, scan_key)
        L = float(energy_loss(q, observed_counts))
        if L < best_loss: best_loss, best_e = L, e
    return float(best_e)

def stage2(opt_pos, opt_t0, energy_guess, observed_counts, true_direction,
            levels=3, divs=8, max_angle_deg=180., reduction=0.5):
    best_dir = np.array([0., 0., 1.]); best_th = 0.; best_ph = 0.; best_loss = float('inf')
    cone_key = jax.random.PRNGKey(42)
    cur_max = np.radians(max_angle_deg)
    for level in range(levels):
        n_t, n_p = divs, divs * 2
        if level == 0:
            for i in range(n_t):
                tv = np.pi * (i / max(n_t - 1, 1))
                for j in range(n_p):
                    pv = 2 * np.pi * (j / n_p)
                    track = ParticleParams(energy=energy_guess, position=opt_pos,
                                            theta=tv, phi=pv, t0=opt_t0)
                    _, _, _, q = pred_sim(track, cone_key)
                    L = float(poisson_nll_charge(observed_counts, q))
                    if L < best_loss:
                        best_loss = L
                        best_dir = np.array(spherical_to_cartesian(tv, pv))
                        best_th, best_ph = tv, pv
        cur_max *= reduction
    cos_a = np.clip(np.dot(best_dir, true_direction), -1., 1.)
    return float(best_th), float(best_ph), float(np.degrees(np.arccos(cos_a)))

def stage3(opt_pos, th, ph, t0_v, energy_guess, observed_counts, n_steps=10, delta=400):
    energies = jnp.linspace(energy_guess - delta, energy_guess + delta, n_steps)
    scan_key = jax.random.PRNGKey(42)
    best_loss, best_e = float('inf'), energy_guess
    for e in energies:
        track = ParticleParams(energy=e, position=opt_pos, theta=th, phi=ph, t0=t0_v)
        _, _, _, q = pred_sim(track, scan_key)
        L = float(energy_loss(q, observed_counts))
        if L < best_loss: best_loss, best_e = L, e
    return float(best_e)

print('Stage 0/2/3 helpers ready.')"""),

    md("""## Stage 4 — Adam refinement with factored-sum loss"""),

    code("""def stage4(initial_params, hit_times, hit_counts, true_pos, true_dir, TRUE_T0,
            true_energy, lr=0.2, b1=0.9, b2=0.999, eps_a=1e-8):
    opt = optax.adam(learning_rate=lr, b1=b1, b2=b2, eps=eps_a)
    opt_state = opt.init(initial_params)
    cur = jnp.array(initial_params)
    POS_LR, DIR_LR, T0_LR, ENE_LR = 0.4, 0.5, 0.05, 1.0
    opt_key = jax.random.PRNGKey(12345)
    DETR = detector_bounds['r']; DETH = detector_bounds['H']
    history = {'losses': [], 'pos_err': [], 'dir_err': [], 't0_err': [], 'E_err': []}
    for it in range(MAX_ITERATIONS):
        opt_key, _ = jax.random.split(opt_key)
        L, g = loss_and_grad_s4(cur, hit_times, hit_counts, opt_key)
        if jnp.any(jnp.isnan(g)):
            g = jnp.nan_to_num(g, nan=0.0)
        if jnp.linalg.norm(g) < TOLERANCE:
            break
        if it < 25:
            scales = jnp.array([0., 0., 0., 0., DIR_LR, DIR_LR, 0.])
        else:
            scales = jnp.array([POS_LR, POS_LR, POS_LR, T0_LR, DIR_LR, DIR_LR, ENE_LR])
        upd, opt_state = opt.update(g, opt_state, cur)
        cur = optax.apply_updates(cur, upd * scales)
        cur = jnp.array([
            jnp.clip(cur[0], -DETR*0.95, DETR*0.95),
            jnp.clip(cur[1], -DETR*0.95, DETR*0.95),
            jnp.clip(cur[2], -DETH/2*0.95, DETH/2*0.95),
            jnp.clip(cur[3], -20., 20.),
            cur[4], cur[5],
            jnp.clip(cur[6], 300., 2000.)])
        cur_dir = spherical_to_cartesian(cur[4], cur[5])
        cos_a = np.clip(np.dot(np.array(cur_dir), np.array(true_dir)), -1., 1.)
        history['losses'].append(float(L))
        history['pos_err'].append(float(jnp.linalg.norm(cur[:3] - true_pos)))
        history['dir_err'].append(float(np.degrees(np.arccos(cos_a))))
        history['t0_err'].append(float(abs(cur[3] - TRUE_T0)))
        history['E_err'].append(float(abs(cur[6] - true_energy)))
    return cur, history
print('Stage 4 ready.')"""),

    md("""## Run N events"""),

    code("""all_results = []
for ev in range(N_EVENTS):
    entry_idx = START_ENTRY + ev
    seed = SEED_BASE + ev
    print(f'\\n=== event {ev + 1}/{N_EVENTS}  (entry={entry_idx}, seed={seed}) ===')
    (true_track, true_data, x, y, z, theta, phi, energy) = meb.build_event_for_entry(
        detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
    TRUE_T0 = 0.0
    true_pos = np.array([x, y, z])
    true_dir = np.asarray(spherical_to_cartesian(theta, phi))
    n_hit = int(np.sum(np.asarray(true_data[0]) > 0))
    print(f'  true: pos=({x:+.2f},{y:+.2f},{z:+.2f}), E={energy:.0f}, n_hit={n_hit}')

    hit_counts, _hit_times_true, hit_times = true_data
    obs_counts = np.asarray(hit_counts)
    obs_times = hit_times

    # Stage 0
    e0 = stage0(obs_counts, energy)
    # Stage 1 (geometric grid)
    hit_mask = obs_counts > 0
    hit_positions = detector_points[hit_mask]
    obs_times_hit = obs_times[hit_mask]
    obs_counts_hit = hit_counts[hit_mask]
    s1 = hierarchical_position_grid_search(
        hit_positions, obs_times_hit, obs_counts_hit,
        true_pos, TRUE_T0, 0.0, detector_bounds,
        n_div=5, t0_n_div=5, levels=4, fraction=0.95,
        t0_min=-15., t0_max=15., min_L=0.05, verbosity=0)
    # Stage 2 (direction cone)
    th2, ph2, dir_err = stage2(s1['best_position'], s1['best_t0'], e0,
                                  obs_counts, true_dir)
    # Stage 3 (energy refine)
    e3 = stage3(s1['best_position'], th2, ph2, s1['best_t0'], e0, obs_counts)

    # Stage 4 (Adam, factored_sum)
    init = jnp.array([s1['best_position'][0], s1['best_position'][1],
                       s1['best_position'][2], s1['best_t0'], th2, ph2, e3])
    final, hist = stage4(init, hit_times, hit_counts,
                            true_pos, true_dir, TRUE_T0, energy)
    print(f'  final: pos_err={hist["pos_err"][-1]:.3f}m  '
          f'dir_err={hist["dir_err"][-1]:.2f}°  '
          f't0_err={hist["t0_err"][-1]:.3f}  '
          f'E_err={hist["E_err"][-1]:.1f}MeV  ({len(hist["losses"])} iters)')
    all_results.append({'event_idx': ev, 'entry_idx': entry_idx, 'seed': seed,
                          'final': np.asarray(final), 'history': hist,
                          'true_pos': true_pos, 'true_dir': true_dir,
                          'true_energy': float(energy)})"""),

    md("""## Aggregate residuals"""),

    code("""pos_errs = np.array([r['history']['pos_err'][-1] for r in all_results])
dir_errs = np.array([r['history']['dir_err'][-1] for r in all_results])
t0_errs  = np.array([r['history']['t0_err'][-1] for r in all_results])
E_errs   = np.array([r['history']['E_err'][-1] for r in all_results])
E_signed = np.array([r['final'][6] - r['true_energy'] for r in all_results])

def stats(a):
    return f'mean={np.mean(a):+.3f}, median={np.median(a):+.3f}, std={np.std(a, ddof=1):.3f}'
print(f'  pos error (m):   {stats(pos_errs)}')
print(f'  dir error (°):   {stats(dir_errs)}')
print(f'  t0 error (ns):   {stats(t0_errs)}')
print(f'  E error (MeV):   {stats(E_errs)}')
print(f'  E signed (MeV):  {stats(E_signed)}    # negative = under-estimated')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, arr, title in zip(axes.ravel(),
                                [pos_errs, dir_errs, t0_errs, E_signed],
                                ['pos error (m)', 'dir error (deg)',
                                 't0 error (ns)', 'E signed (MeV)']):
    ax.hist(arr, bins=15, color='C0', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', ls='--')
    ax.axvline(arr.mean(), color='darkorange', ls='-',
                label=f'mean={arr.mean():+.3f}')
    ax.set_title(title); ax.set_ylabel('events')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

    md("""## Notes

- Stage 4 uses the **factored-sum** loss (C_event + T_event), with all knobs
  (`MODE`, `HIT_THRESHOLD`, `LAMBDA_BG`, `WAVELENGTH_MODE`) exposed at the top.
- Stages 0–3 are kept simple/cheap as in the reference notebook.
- For comparison, change `MODE` to `'joint'` or `'gmean_joint'` to see how
  the converged residuals change.
- Per-event traces are in `all_results[i]['history']` (loss, pos_err, etc.
  per Adam iteration)."""),
]


# ---------------------------------------------------------------------------
# Notebook 3 — event_visualization.ipynb
# ---------------------------------------------------------------------------
NB3_CELLS = [
    md("""# Event displays + per-sensor likelihood vs observed time

Thin wrapper over `per_sensor_time_overlay.py` and `per_sensor_p_first_validation.py`.

For one ground-truth event:

1. **Event displays** — 2D unwrapped detector views (true / pred × charge / time).
2. **Per-sensor diagnostics** — three-panel plots for representative sensors:
   - λ(t) (Gaussian-convolved per-photon arrival rate)
   - cumulative Λ(t) with 1-PE reference
   - first-arrival density `p_first(t) = λ(t)·exp(-Λ(t))` with the observed t_obs overlaid and per-sensor NLL annotated.
3. **Empirical validation** — overlay the empirical t_obs histogram (across many fresh data-sim trials on the same true event) against the analytical `p_first(t)`.

Heavy logic lives in the two scripts. Each cell imports and calls in."""),

    md("## Setup"),

    code("""import sys, time
sys.path.append('..')
import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import per_sensor_time_overlay as pso
import per_sensor_p_first_validation as psv

# Knobs (override the script defaults here)
pso.ENTRY_IDX = 2
pso.SEED = 44
pso.N_PER_STRATUM = 3
psv.N_TRIALS = 200          # number of data-sim re-samples for validation
print(f'entry={pso.ENTRY_IDX}, seed={pso.SEED}, N_PER_STRATUM={pso.N_PER_STRATUM}, '
      f'σ_TTS={pso.SIGMA_TTS_NS} ns, N_TRIALS={psv.N_TRIALS}')"""),

    md("## Build simulators and one event"),

    code("""detector, num_detectors, data_sim, pred_sim = pso.build_simulators()
print(f'Detectors: {num_detectors}')

t0 = time.time()
(true_track, true_data, pred_data_pp,
 pos, direction, energy, photon_data) = pso.build_event(
    detector, data_sim, pred_sim, jax.random.PRNGKey(pso.SEED))
print(f'build_event: {time.time() - t0:.1f}s')

# Stay in JAX. The simulator returns device arrays; downstream JIT'd helpers
# accept them directly (no host transfer). We convert to numpy lazily only at
# matplotlib boundaries.
log_w_j, flat_times_j, flat_indices_j, total_charge_j = pred_data_pp
weights_j = jnp.exp(log_w_j)

# Just one host pull for the print + sensor stratification (small)
hit_counts0 = np.asarray(true_data[0])
n_hit = int(np.sum(hit_counts0 > 0))
print(f'event: E = {float(energy):.1f} MeV, hit sensors = {n_hit}/{num_detectors}')"""),

    md("""## 1. Build per-sensor analytical diagnostics (one JIT call)

`evaluate` is `@jax.jit`'d. We pass JAX arrays directly so the call has no
host→device copy. The returned `diag` is a dict of JAX arrays; we lazily pull
to numpy only when matplotlib needs them."""),

    code("""print('Computing analytical p_first diagnostics ...')
t0 = time.time()
evaluate = pso.make_diagnostics_evaluator(
    num_sensors=num_detectors, n_grid=200, sigma_tts=pso.SIGMA_TTS_NS)
diag = evaluate(flat_times_j, weights_j, flat_indices_j, true_data[1])
print(f'  done in {time.time() - t0:.2f}s (grid: {diag["t_grid"].shape[1]} samples/sensor)')"""),

    md("## 2. Event displays — 4 unwrapped 2D views"),

    code("""pso.EVENT_DISPLAY_DIR.mkdir(parents=True, exist_ok=True)
# matplotlib boundary: pull the per-sensor (NUM_DETECTORS-element) arrays
# we actually plot. The big per-photon arrays stay on-device.
pred_charges = np.asarray(total_charge_j)
mode_np   = np.asarray(diag['mode'])
t_lead_np = np.asarray(diag['t_lead'])
pred_times = np.where(np.isfinite(mode_np), mode_np, t_lead_np)
pred_times = np.nan_to_num(pred_times, nan=0.0)
pso.render_event_displays(true_data, pred_charges, pred_times)
display_paths = sorted(pso.EVENT_DISPLAY_DIR.glob('*.png'))
print(f'saved {len(display_paths)} displays in {pso.EVENT_DISPLAY_DIR}/')
from IPython.display import Image, display
for p in display_paths[-4:]:
    print(p.name)
    display(Image(filename=str(p)))"""),

    md("""## 3. Per-sensor 3-panel plots

Pick high / medium / low PE sensors and plot:
  (a) λ(t),  (b) cumulative Λ(t),  (c) p_first(t) = λ(t)·exp(-Λ(t)) with t_obs overlaid."""),

    code("""# Stratification works on per-sensor arrays (NUM_DETECTORS elements — small).
total_pe = np.asarray(total_charge_j)
high, medium, low = pso.stratified_sensor_selection(
    total_pe, hit_counts0, n_per_stratum=pso.N_PER_STRATUM, hit_threshold=1.0)
print(f'High PE   sensors: {high}')
print(f'Medium PE sensors: {medium}')
print(f'Low PE    sensors: {low}')

# matplotlib boundary for the per-photon arrays — needed for plot_single_sensor's
# per-sensor masking. (Plotting could be made fully JAX-native by refactoring
# plot_single_sensor to mask sensors in jax and only pull each sensor's small
# subset; left as future work in the script.)
flat_times   = np.asarray(flat_times_j)
flat_indices = np.asarray(flat_indices_j)
weights      = np.asarray(weights_j)

pso.PER_SENSOR_DIR.mkdir(parents=True, exist_ok=True)
pso.render_per_sensor_plots(
    flat_times=flat_times, weights=weights, flat_indices=flat_indices,
    total_pe=total_pe, true_data=true_data,
    stratum_sensors={'high': high, 'medium': medium, 'low': low},
    diag=diag,
)
sensor_paths = sorted(pso.PER_SENSOR_DIR.glob('*.png'))
print(f'saved {len(sensor_paths)} per-sensor plots in {pso.PER_SENSOR_DIR}/')
# Show a few inline (one per stratum)
shown_strata = set()
for p in sensor_paths:
    stratum = p.stem.split('_')[0]
    if stratum in shown_strata:
        continue
    shown_strata.add(stratum)
    print(p.name)
    display(Image(filename=str(p)))
    if len(shown_strata) >= 3:
        break"""),

    md("""## 4. Empirical-vs-analytical p_first validation

Run the data simulator MANY times on the same true event with fresh keys.
Overlay the empirical t_obs histogram (across trials, conditioned on hit) on
the analytical p_first(t) used by the loss. They should agree modulo MC noise.

`diag` was already computed in section 1 — we reuse it."""),

    code("""print(f'Running {psv.N_TRIALS} data trials ...')
t0 = time.time()
all_hit_counts, all_hit_times = psv.run_data_trials(
    data_sim, true_track, photon_data, jax.random.PRNGKey(pso.SEED + 1),
    psv.N_TRIALS)
print(f'  done: {time.time() - t0:.2f}s')"""),

    code("""# Plot validation for a representative sensor in each stratum
psv.OUT_DIR.mkdir(parents=True, exist_ok=True)
label_map = {'high': 'High PE', 'medium': 'Medium PE', 'low': 'Low PE'}
selected = {'high': high[:1], 'medium': medium[:1], 'low': low[:1]}
val_paths = []
for stratum_key, sensors in selected.items():
    for s in sensors:
        out = psv.OUT_DIR / f'{stratum_key}_sensor_{s}.png'
        psv.plot_validation(
            sensor_id=s, stratum_label=label_map[stratum_key],
            all_hit_counts=all_hit_counts, all_hit_times=all_hit_times,
            diag=diag, out_path=str(out))
        val_paths.append(out)
print(f'saved {len(val_paths)} validation plots in {psv.OUT_DIR}/')
for p in val_paths:
    print(p.name)
    display(Image(filename=str(p)))"""),

    md("""## Notes

- Knobs at the top: `pso.ENTRY_IDX`, `pso.SEED`, `pso.N_PER_STRATUM`, `psv.N_TRIALS`.
  For other settings (temperature, K, σ_TTS, etc.) edit
  `per_sensor_time_overlay.py` directly — those are module-level constants.
- All figures are saved to `figures/event_displays/`, `figures/per_sensor_time/`,
  and `figures/per_sensor_validation/`.
- The validation cell takes ~1–2 min for `N_TRIALS=200` because it runs the
  data simulator N_TRIALS times.

### JAX / numpy convention used in this notebook

- The per-photon arrays from `pred_sim` stay on-device as JAX arrays
  (`flat_times_j`, `flat_indices_j`, `weights_j`, `total_charge_j`).
- The JIT'd `evaluate(...)` consumes them directly — no host transfer; XLA can
  fuse the diagnostic computation with the simulator output if running on GPU.
- `np.asarray(...)` is only invoked at matplotlib boundaries, and ideally on the
  small per-sensor (~10k-element) arrays rather than the big per-photon
  (~1M-element) ones. The per-photon pull in section 3 is the one remaining
  large transfer; a full refactor of `plot_single_sensor` to mask sensors in
  JAX could eliminate it, but it isn't worth it for a one-event diagnostic."""),
]


def main():
    write_notebook(HERE / 'bias_scan_multi_event.ipynb', NB1_CELLS)
    write_notebook(HERE / 'tracking_opt_joint_loss.ipynb', NB2_CELLS)
    write_notebook(HERE / 'event_likelihood_diagnostics.ipynb', NB3_CELLS)
    print('wrote bias_scan_multi_event.ipynb (notebook 1)')
    print('wrote tracking_opt_joint_loss.ipynb (notebook 2)')
    print('wrote event_likelihood_diagnostics.ipynb (notebook 3)')


if __name__ == '__main__':
    main()