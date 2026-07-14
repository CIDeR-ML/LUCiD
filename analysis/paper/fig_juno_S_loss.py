#!/usr/bin/env python3
"""Figure: JUNO-like WbLS sphere — differentiability of the scintillation yield S (ph/MeV).

The forward simulator is re-run at every scan point with a different S, and the loss gradient
dNLL/dS is taken by AUTOMATIC DIFFERENTIATION through the simulator (not by rescaling a
precomputed prediction). S enters as a runtime DetectorParams field, so one jax.value_and_grad
of the loss is JIT-compiled once and evaluated at each S.

Loss = per-sensor Poisson NLL of the total predicted charge at yield S vs the fixed data-like
event (the injected PhotonSim photons cached by fig_juno_cher_scint_fraction.py):
    mu_i(S) = forward_sim(track, S)_i   (both emission processes)
    n_i     = qc_data_i + qs_data_i
    NLL(S)  = sum_i [ mu_i - n_i * ln(mu_i) ]
The plotted gradient's zero crossing recovers the truth yield (S_hat ~ S0) — scintillation-yield
calibration by gradient descent.

    python analysis/paper/fig_juno_S_loss.py --generate-data   # the S scan (needs the charge cache)
    python analysis/paper/fig_juno_S_loss.py --plot-results    # figure from the cached scan
    python analysis/paper/fig_juno_S_loss.py                   # both

Needs the charge cache from fig_juno_cher_scint_fraction.py first. Run in the container (GPU).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'juno_S_loss'
DET = 'JUNO_wbls'
GEOM = str(CONFIG / f'{DET}_geom_config.json')
PHYS = str(CONFIG / f'{DET}_physics_config.json')
ENERGY = 1000.0
SEED = 6
EPS = 1e-6
CHARGES = paths.data_dir('juno_cher_scint', 'local') / 'charges.npz'   # from the display figure


def _scan_file():
    return paths.data_dir(FIGURE, 'local') / 'scan.npz'


def generate_data(theta, phi, nphot, npoints, srel_lo, srel_hi):
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams, load_physics_config
    from lucid.simulation import setup_event_simulator
    if not CHARGES.exists():
        raise SystemExit(f'[error] charge cache {CHARGES} missing — run '
                         f'fig_juno_cher_scint_fraction.py --generate-data first')

    # Forward prediction must emit BOTH processes so the total charge depends on S. Patch
    # make_medium in every module that binds it (see fig_juno_cher_scint_fraction for why).
    import importlib
    import lucid.wavelength.medium as _wl_medium
    _orig = _wl_medium.make_medium
    def _both(material, *a, **k):
        return _orig(material, *a, **k)._replace(emission_processes=('cherenkov', 'scintillation'))
    for m in ('lucid.wavelength.medium', 'lucid.simulation.simulator',
              'lucid.geometry.detector_geometry'):
        mod = importlib.import_module(m)
        if hasattr(mod, 'make_medium'):
            mod.make_medium = _both

    sim = setup_event_simulator(GEOM, nphot, temperature=0.0, K=6, is_data=False,
                                detector_type='Sphere', max_candidates_per_ray=4,
                                physics_config=PHYS, default_detector_params=False,
                                hit_mode='aggregated')
    dp0, _, _ = load_physics_config(PHYS)
    S0 = float(dp0.scintillation.S)
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(energy=jnp.array(ENERGY, jnp.float32), position=jnp.zeros(3, jnp.float32),
                           theta=jnp.array(theta, jnp.float32), phi=jnp.array(phi, jnp.float32),
                           t0=jnp.array(0.0, jnp.float32))

    z = np.load(CHARGES)
    mask = (z['qc_p'] + z['qs_p']) > 0.0                # predicted-active sensors
    midx = jnp.asarray(np.where(mask)[0])
    n = jnp.asarray((z['qc_d'] + z['qs_d'])[mask])      # observed total charge (data)
    print(f'likelihood over {int(mask.sum())} sensors; NPHOT={nphot}', flush=True)

    def nll(S):
        dp = dp0._replace(scintillation=dp0.scintillation._replace(S=S))
        charges, _ = sim(track, dp, key)                # re-simulate prediction at S
        mu = jnp.clip(charges[midx], EPS, None)
        return jnp.sum(mu - n * jnp.log(mu))

    vg = jax.jit(jax.value_and_grad(nll))               # JIT once
    S_scan = np.linspace(srel_lo * S0, srel_hi * S0, npoints)
    losses = np.empty(npoints); grads = np.empty(npoints)
    for i, S in enumerate(S_scan):
        l, g = vg(jnp.asarray(S, jnp.float32))
        losses[i] = float(l); grads[i] = float(g)
        print(f'  S={S:8.1f} ({S/S0:4.2f} S0)  NLL={losses[i]:.5e}  dNLL/dS={grads[i]:+.5e}',
              flush=True)

    sf = _scan_file(); sf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sf, S=S_scan, S0=S0, loss=losses, grad=grads)
    print(f'cached scan -> {sf}')


def plot_results(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = 13
    sf = _scan_file()
    if not sf.exists():
        print(f'[skip] no scan cache at {sf} — run --generate-data first'); return
    z = np.load(sf)
    S, S0, loss, grad = z['S'], float(z['S0']), z['loss'], z['grad']
    r = S / S0
    i = np.where(np.diff(np.sign(grad)) != 0)[0]
    r_min = None
    if len(i):
        j = i[0]
        r_min = r[j] - grad[j] * (r[j + 1] - r[j]) / (grad[j + 1] - grad[j])
    dloss = loss - loss.min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08),
                                   facecolor='white')
    ax1.plot(r, dloss, '-', color='cornflowerblue', lw=2.4, zorder=1)
    ax1.plot(r, dloss, 'o', color='navy', ms=6, zorder=3)
    ax1.set_ylabel(r'$\Delta \mathcal{L}$')
    ax2.axhline(0.0, color='0.5', lw=1.0)
    ax2.plot(r, grad, '-', color='cornflowerblue', lw=2.4, zorder=1)
    ax2.plot(r, grad, 'o', color='navy', ms=6, zorder=3)
    ax2.set_ylabel(r'$d\mathcal{L}/dS$')
    ax2.set_xlabel(r'Scintillation yield  $S / S_0$')
    if r_min is not None:
        for ax in (ax1, ax2):
            ax.axvline(r_min, color='0.5', ls=':', lw=1.2)
        ax2.plot(r_min, 0.0, '*', color='#c0392b', ms=15, zorder=4,
                 label=r'$\hat{S}=%.3f\,S_0$' % r_min)
        ax2.legend(frameon=False, loc='lower right')

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'sphere_wbls_S_loss_scan'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {base}.pdf (+png)' + (f'; S_hat = {r_min:.3f} S0' if r_min else
                                        '; no zero crossing in range'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--theta', type=float, default=float(np.pi / 4))
    ap.add_argument('--phi', type=float, default=float(np.pi / 6))
    ap.add_argument('--nphot', type=int, default=1_000_000)
    ap.add_argument('--npoints', type=int, default=11)
    ap.add_argument('--srel-lo', type=float, default=0.5)
    ap.add_argument('--srel-hi', type=float, default=1.5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.theta, a.phi, a.nphot, a.npoints, a.srel_lo, a.srel_hi)
    if a.plot_results or both:
        plot_results(Path(a.out) if a.out else paths.figure_dir())


if __name__ == '__main__':
    main()
