#!/usr/bin/env python3
"""Figure: JUNO-like WbLS sphere — Cherenkov-vs-scintillation asymmetry, data vs prediction.

The per-sensor quantity is the signed asymmetry
    a = (Q_cher - Q_scint) / (Q_cher + Q_scint)   in [-1, +1]
on a diverging blue->white->red scale (+1 pure Cherenkov, -1 pure scintillation, white where
they balance). Two plotly disc scenes side by side (data left, prediction right), shared colorbar.

  * DATA       : real PhotonSim ROOT photons (Cherenkov + scintillation expanded from the dE/dx
                 segments) injected into the is_data simulator, one emission process at a time.
  * PREDICTION : forward sim of the same track, Cherenkov-only and scintillation-only, combined.

The expensive per-process forward sims are cached, so layout/style can be iterated with
--plot-results alone. The S-yield figure (fig_juno_S_loss.py) reuses this same charge cache.

    python analysis/paper/fig_juno_cher_scint_fraction.py                 # simulate + cache + plot
    python analysis/paper/fig_juno_cher_scint_fraction.py --generate-data
    python analysis/paper/fig_juno_cher_scint_fraction.py --plot-results  # re-montage from cache

Run inside the container (needs plotly+kaleido and a GPU for the sims); see docs/QUICKSTART_S3DF.md.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'juno_cher_scint'
DET = 'JUNO_wbls'
GEOM = str(CONFIG / f'{DET}_geom_config.json')
PHYS = str(CONFIG / f'{DET}_physics_config.json')
DEFAULT_ROOT = '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/water/mu-/1000MeV_100events.root'
ENTRY = 0
N_PHOTONS = 2_000_000
ENERGY = 1000.0
SEED = 6
FRACTION_REL_THRESHOLD = 0.02
DIVERGING = [[0.0, '#2166ac'], [0.5, '#ffffff'], [1.0, '#b2182b']]
CB = dict(colorbar_label='', colorbar_thickness=36, colorbar_len=0.32,
          colorbar_tickfont_size=40, colorbar_tickfont_family='serif', colorbar_x=0.86)


def _cache_file():
    return paths.data_dir(FIGURE, 'local') / 'charges.npz'


# ---- emission-process patch (forward/prediction path) -----------------------------------
# The forward simulator resolves make_medium from lucid.simulation.simulator (which did
# `from lucid.wavelength.medium import make_medium`). Patch every module that binds the name
# so the built medium emits only the requested process(es). setup_event_simulator builds the
# medium once, so patch BEFORE each per-process setup call.
import lucid.wavelength.medium as _wl_medium            # noqa: E402
_ORIG_MAKE_MEDIUM = _wl_medium.make_medium
_DESIRED_PROCS = {'value': ('cherenkov', 'scintillation')}
_PATCH_MODULES = ['lucid.wavelength.medium', 'lucid.simulation.simulator',
                  'lucid.geometry.detector_geometry']


def _patched_make_medium(material, *a, **k):
    return _ORIG_MAKE_MEDIUM(material, *a, **k)._replace(
        emission_processes=_DESIRED_PROCS['value'])


def _set_emission_patch(on):
    import importlib
    fn = _patched_make_medium if on else _ORIG_MAKE_MEDIUM
    for m in _PATCH_MODULES:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, 'make_medium'):
                mod.make_medium = fn
        except Exception:
            pass


def _track_cartesian(theta, phi):
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams
    from lucid.utils import spherical_to_cartesian
    direction = np.asarray(spherical_to_cartesian(theta, phi), dtype=np.float64)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction); an = np.linalg.norm(axis)
    rot_axis = (np.array([1.0, 0.0, 0.0]) if an < 1e-8 else axis / an)
    rot_angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
    track = ParticleParams.from_cartesian(
        energy=jnp.array(ENERGY, dtype=jnp.float32),
        position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32), t0=0.0)
    return track, rot_axis, rot_angle


def _data_asymmetry(root, entry, theta, phi):
    """Inject real ROOT photons (per process) -> per-sensor (Q_cher, Q_scint)."""
    import jax
    import jax.numpy as jnp
    from lucid.simulation import setup_event_simulator
    from lucid.generate import read_event_data_from_photonsim
    from lucid.sources.scintillation_photons import scintillation_medium_params
    key = jax.random.PRNGKey(SEED)
    sim = setup_event_simulator(GEOM, N_PHOTONS, temperature=0.0, K=6, is_data=True,
                                is_calibration=False, detector_type='Sphere',
                                max_candidates_per_ray=4, physics_config=PHYS,
                                default_detector_params=True, hit_mode='aggregated')
    medium_params = scintillation_medium_params(sim.default_detector_params, sim.medium)
    track, rot_axis, rot_angle = _track_cartesian(theta, phi)

    def charges_for(procs):
        pd = dict(read_event_data_from_photonsim(
            root, entry, emission_processes=procs, medium_params=medium_params,
            rng=np.random.default_rng(SEED)))
        pd['rotation_axis'] = jnp.asarray(rot_axis, dtype=jnp.float32)
        pd['rotation_angle'] = jnp.asarray(rot_angle, dtype=jnp.float32)
        pd['apply_rotation'] = jnp.array(True)
        pd['apply_translation'] = jnp.array(False)
        pd['translation_vector'] = jnp.zeros(3, dtype=jnp.float32)
        q, _ = jax.lax.stop_gradient(sim(track, key, pd))
        return np.asarray(q)

    qc, qs = charges_for(('cherenkov',)), charges_for(('scintillation',))
    print(f'DATA  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}', flush=True)
    return qc, qs


def _pred_asymmetry(theta, phi):
    """Forward sim (per process via the emission patch) -> per-sensor (Q_cher, Q_scint)."""
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams
    from lucid.simulation import setup_event_simulator
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(
        energy=jnp.array(ENERGY, dtype=jnp.float32),
        position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        theta=jnp.array(theta, dtype=jnp.float32), phi=jnp.array(phi, dtype=jnp.float32),
        t0=jnp.array(0.0, dtype=jnp.float32))

    def charges_for(procs):
        _DESIRED_PROCS['value'] = procs
        _set_emission_patch(True)
        try:
            sim = setup_event_simulator(GEOM, 1_000_000, temperature=0.0, K=6, is_data=False,
                                        is_calibration=False, detector_type='Sphere',
                                        max_candidates_per_ray=4, physics_config=PHYS,
                                        default_detector_params=True, hit_mode='aggregated')
            q, _ = jax.lax.stop_gradient(sim(track, key))
        finally:
            _set_emission_patch(False)
        return np.asarray(q)

    qc, qs = charges_for(('cherenkov',)), charges_for(('scintillation',))
    print(f'PRED  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}', flush=True)
    return qc, qs


def generate_data(root, entry, theta, phi):
    qc_d, qs_d = _data_asymmetry(root, entry, theta, phi)
    qc_p, qs_p = _pred_asymmetry(theta, phi)
    cf = _cache_file(); cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cf, qc_d=qc_d, qs_d=qs_d, qc_p=qc_p, qs_p=qs_p)
    print(f'cached charges -> {cf}')


# ---- plotting ---------------------------------------------------------------------------
def _panel_fig(detector, qc, qs, show_colorbar, eye):
    total = qc + qs
    qmax = float(total.max())
    lit = np.where(total > FRACTION_REL_THRESHOLD * qmax)[0]
    asym = (qc[lit] - qs[lit]) / total[lit]
    print(f'  asym on {len(lit)} sensors, range [{asym.min():.2f}, {asym.max():.2f}]')
    cb = dict(CB, colorbar_x=0.985, colorbar_len=0.55)
    return detector.visualize_event_data_plotly_discs(
        lit, asym, np.zeros(len(lit)), show_all_sensors=True, log_scale=False,
        show_colorbar=show_colorbar, dark_theme=False, plot_time=False, colorscale=DIVERGING,
        surface_color='lightgray', inactive_color='lightgray', inactive_opacity=0.15,
        cmin=-1.0, cmax=1.0, colorbar_tickvals=[-1.0, 0.0, 1.0],
        colorbar_ticktext=['-1', '0', '1'], return_fig=True, **cb)


def plot_results(out, eye, panel_gap, cbar_gutter):
    from plotly.subplots import make_subplots
    from lucid.geometry import generate_detector
    cf = _cache_file()
    if not cf.exists():
        print(f'[skip] no charge cache at {cf} — run --generate-data first'); return
    z = np.load(cf)
    detector = generate_detector(GEOM)
    eye_d = dict(x=eye, y=eye, z=eye)
    f_data = _panel_fig(detector, z['qc_d'], z['qs_d'], show_colorbar=False, eye=eye_d)
    f_pred = _panel_fig(detector, z['qc_p'], z['qs_p'], show_colorbar=True, eye=eye_d)
    sub = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        horizontal_spacing=0.0)
    for tr in f_data.data:
        tr.update(scene='scene'); sub.add_trace(tr, row=1, col=1)
    for tr in f_pred.data:
        tr.update(scene='scene2'); sub.add_trace(tr, row=1, col=2)
    sd = f_data.layout.scene.to_plotly_json(); sd.pop('domain', None)
    sp = f_pred.layout.scene.to_plotly_json(); sp.pop('domain', None)
    sub.layout.scene.update(sd); sub.layout.scene2.update(sp)
    half = (1.0 - cbar_gutter - panel_gap) / 2.0
    sub.layout.scene.update(domain=dict(x=[0.0, half], y=[0.0, 1.0]), camera=dict(eye=eye_d))
    sub.layout.scene2.update(domain=dict(x=[half + panel_gap, 2 * half + panel_gap],
                                         y=[0.0, 1.0]), camera=dict(eye=eye_d))
    sub.update_layout(width=2200, height=1100, showlegend=False, paper_bgcolor='white',
                      plot_bgcolor='white', margin=dict(l=0, r=0, t=0, b=0))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'sphere_wbls_cher_scint_fraction'
    for ext in ('pdf', 'png'):
        sub.write_image(f'{base}.{ext}', scale=2)
    print(f'wrote {base}.pdf (+png)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--entry', type=int, default=ENTRY)
    ap.add_argument('--theta', type=float, default=float(np.pi / 4))
    ap.add_argument('--phi', type=float, default=float(np.pi / 6))
    ap.add_argument('--eye', type=float, default=0.95, help='camera zoom (smaller = sphere fills more)')
    ap.add_argument('--panel-gap', type=float, default=0.0)
    ap.add_argument('--cbar-gutter', type=float, default=0.07)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.root, a.entry, a.theta, a.phi)
    if a.plot_results or both:
        plot_results(Path(a.out) if a.out else paths.figure_dir(),
                     a.eye, a.panel_gap, a.cbar_gutter)


if __name__ == '__main__':
    main()
