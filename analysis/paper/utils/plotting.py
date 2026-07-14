"""Thin wrappers over the ``analysis/paper/utils/ovv_plots`` plot modes.

The paper figures reuse the exact plotting code that made the study figures; these
helpers just assemble the argument namespace so a figure script stays a few lines.
"""
from pathlib import Path
from types import SimpleNamespace

from analysis.paper.utils import ovv_plots


def nrays_combined(mu_base, el_base, mass_mu, mass_el, out,
                   max_rays=250000, max_events=None):
    """Muon+electron resolution-vs-nrays overlay (ovv_plots.mode_nrays_combined).

    ``mu_base`` / ``el_base`` are the run-dir prefixes: the h5 for tag ``T`` is
    read from ``<base><T>/<base-name><T>.h5`` (ovv_plots globs the dir).
    """
    args = SimpleNamespace(base=str(mu_base), base_el=str(el_base),
                           mass=float(mass_mu), mass_el=float(mass_el),
                           out=str(out), max_rays=max_rays, max_events=max_events)
    ovv_plots.mode_nrays_combined(args)
    return Path(out) / 'tracking_performance_vs_nrays_combined.pdf'


def _h5_in(d):
    """The single .h5 inside a run dir (matches ovv_plots' dir-name / glob convention)."""
    d = Path(d)
    cand = d / f'{d.name}.h5'
    if cand.exists():
        return cand
    hits = sorted(d.glob('*.h5'))
    return hits[0] if len(hits) == 1 else None


def energy_composite(w1_dir, w075_dir, energies, crossover, out,
                     name_w1='escan', name_w075='escanw75', max_energy=1800):
    """Composite energy figure: w=1 below ``crossover``, w=0.75 at/above it.

    Stages symlink run-dirs (``efinal_<E>`` pointing at the w=1 or w=0.75 h5) and
    calls ovv_plots.mode_energy over them, so the published policy is expressed by
    which source each energy point is drawn from.
    """
    w1_dir, w075_dir, out = Path(w1_dir), Path(w075_dir), Path(out)
    stage = out / '_efinal_stage'
    stage.mkdir(parents=True, exist_ok=True)
    for E in energies:
        src = _h5_in((w075_dir if E >= crossover else w1_dir) /
                     f'{name_w075 if E >= crossover else name_w1}_{E}')
        if src is None:
            continue
        d = stage / f'efinal_{E}'; d.mkdir(exist_ok=True)
        link = d / f'efinal_{E}.h5'
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
    args = SimpleNamespace(base=str(stage / 'efinal_'), max_energy=max_energy, out=str(out))
    ovv_plots.mode_energy(args)
    return out / 'tracking_performance_vs_energy.pdf'


def sensors(base, out, max_sensors=18000):
    """Detector-performance-vs-num-sensors figure (ovv_plots.mode_sensors)."""
    args = SimpleNamespace(base=str(base), max_sensors=max_sensors, out=str(out))
    ovv_plots.mode_sensors(args)
    return Path(out) / 'detector_perf_vs_num_sensors.pdf'


def convergence(h5, out, label):
    """Per-event convergence figure (ovv_plots.mode_iters), renamed by ``label``."""
    out = Path(out)
    args = SimpleNamespace(h5=str(h5), out=str(out))
    ovv_plots.mode_iters(args)
    stem = out / 'convergence_all_metrics'
    dst = out / f'{label}_convergence'
    for ext in ('png', 'pdf'):
        src = stem.with_suffix(f'.{ext}')
        if src.exists():
            src.replace(dst.with_suffix(f'.{ext}'))
    return dst.with_suffix('.pdf')
