#!/usr/bin/env python3
"""JUNO-like (WbLS sphere): side-by-side Cherenkov-scintillation fraction for a
DATA-like and a PREDICTION-like event.

The per-sensor quantity is the signed Cherenkov-vs-scintillation asymmetry
    a = (Q_cher - Q_scint) / (Q_cher + Q_scint)   in [-1, +1]
rendered on a diverging blue->white->red scale (+1 pure Cherenkov, -1 pure
scintillation, white where they balance).

  * DATA-like       : the real PhotonSim ROOT photons (Cherenkov + scintillation
                      expanded from dE/dx segments) injected into the is_data
                      simulator, one process at a time.
  * PREDICTION-like : forward sim of the same track; Cherenkov-only and
                      scintillation-only run separately and combined.

Both panels share the same view and colour scale; they are rendered as two
plotly disc images and montaged horizontally (data left, prediction right) with
a single shared colorbar on the right panel.

Run inside the container with the kaleido env:
  APPTAINERENV_PYTHONUSERBASE=$LUCID_ENV_BASE apptainer exec ... \
      /opt/conda/bin/python3 /opt/ClaudePlayground/juno_cher_scint_fraction.py
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LUCID_DIR = Path(os.environ.get("LUCID_DIR", HERE.parents[1]))   # analysis/juno_wbls -> repo root
if not (LUCID_DIR / "lucid").is_dir():
    LUCID_DIR = Path("/opt/LUCiD")
sys.path.insert(0, str(LUCID_DIR))
CONFIG = LUCID_DIR / "config"

import jax
import jax.numpy as jnp

import lucid.geometry.detector_geometry as _detgeom
from lucid.geometry import generate_detector
from lucid.detector_params import ParticleParams
from lucid.simulation import setup_event_simulator
from lucid.utils import spherical_to_cartesian
from lucid.generate import read_event_data_from_photonsim
from lucid.sources.scintillation_photons import scintillation_medium_params

DET = "JUNO_wbls"
GEOM = str(CONFIG / f"{DET}_geom_config.json")
PHYS = str(CONFIG / f"{DET}_physics_config.json")
ROOT = "/sdf/data/neutrino/cjesus/CIDER/ROOT_files/water/mu-/1000MeV_100events.root"
ENTRY = 0
N_PHOTONS = 2_000_000
THETA = jnp.pi / 4
PHI = jnp.pi / 6
ENERGY = 1000.0
SEED = 6
FIG = HERE / "figures"
FRACTION_REL_THRESHOLD = 0.02
DIVERGING = [[0.0, "#2166ac"], [0.5, "#ffffff"], [1.0, "#b2182b"]]
CB = dict(colorbar_label="", colorbar_thickness=36, colorbar_len=0.32,
          colorbar_tickfont_size=40, colorbar_tickfont_family="serif",
          colorbar_x=0.86)

# --- emission-process monkeypatch (forward/prediction path only) -------------
_ORIG_MAKE_MEDIUM = _detgeom.make_medium
_DESIRED_PROCS = {"value": ("cherenkov", "scintillation")}


def _patched_make_medium(material, *a, **k):
    m = _ORIG_MAKE_MEDIUM(material, *a, **k)
    return m._replace(emission_processes=_DESIRED_PROCS["value"])


def _track_cartesian():
    direction = np.asarray(spherical_to_cartesian(THETA, PHI), dtype=np.float64)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction); an = np.linalg.norm(axis)
    rot_axis = (np.array([1.0, 0.0, 0.0]) if an < 1e-8 else axis / an)
    rot_angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
    track = ParticleParams.from_cartesian(
        energy=jnp.array(ENERGY, dtype=jnp.float32),
        position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32), t0=0.0)
    return track, rot_axis, rot_angle


def data_asymmetry():
    """Inject real ROOT photons (per process) -> per-sensor asymmetry."""
    key = jax.random.PRNGKey(SEED)
    sim = setup_event_simulator(
        GEOM, N_PHOTONS, temperature=0.0, K=6, is_data=True,
        is_calibration=False, detector_type="Sphere", max_candidates_per_ray=4,
        physics_config=PHYS, default_detector_params=True, hit_mode="aggregated")
    medium_params = scintillation_medium_params(sim.default_detector_params, sim.medium)
    track, rot_axis, rot_angle = _track_cartesian()

    def charges_for(procs):
        pd = dict(read_event_data_from_photonsim(
            ROOT, ENTRY, emission_processes=procs,
            medium_params=medium_params, rng=np.random.default_rng(SEED)))
        pd["rotation_axis"] = jnp.asarray(rot_axis, dtype=jnp.float32)
        pd["rotation_angle"] = jnp.asarray(rot_angle, dtype=jnp.float32)
        pd["apply_rotation"] = jnp.array(True)
        pd["apply_translation"] = jnp.array(False)
        pd["translation_vector"] = jnp.zeros(3, dtype=jnp.float32)
        q, _ = jax.lax.stop_gradient(sim(track, key, pd))
        return np.asarray(q)

    qc, qs = charges_for(("cherenkov",)), charges_for(("scintillation",))
    print(f"DATA  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}")
    return qc, qs


def pred_asymmetry():
    """Forward sim (per process via emission-process patch) -> asymmetry."""
    _detgeom.make_medium = _patched_make_medium
    try:
        key = jax.random.PRNGKey(SEED)
        track = ParticleParams(
            energy=jnp.array(ENERGY, dtype=jnp.float32),
            position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            theta=jnp.array(THETA, dtype=jnp.float32),
            phi=jnp.array(PHI, dtype=jnp.float32),
            t0=jnp.array(0.0, dtype=jnp.float32))

        def charges_for(procs):
            _DESIRED_PROCS["value"] = procs
            sim = setup_event_simulator(
                GEOM, 1_000_000, temperature=0.0, K=6, is_data=False,
                is_calibration=False, detector_type="Sphere",
                max_candidates_per_ray=4, physics_config=PHYS,
                default_detector_params=True, hit_mode="aggregated")
            q, _ = jax.lax.stop_gradient(sim(track, key))
            return np.asarray(q)

        qc, qs = charges_for(("cherenkov",)), charges_for(("scintillation",))
    finally:
        _detgeom.make_medium = _ORIG_MAKE_MEDIUM
    print(f"PRED  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}")
    return qc, qs


# camera zoom (smaller eye magnitude -> sphere fills more of its scene)
_EYE = float(os.environ.get("EYE", 0.95))
EYE = dict(x=_EYE, y=_EYE, z=_EYE)
# inter-panel gap and the colorbar gutter on the right, in paper fraction
PANEL_GAP = float(os.environ.get("PANEL_GAP", 0.0))
CBAR_GUTTER = float(os.environ.get("CBAR_GUTTER", 0.07))


def panel_fig(detector, qc, qs, show_colorbar):
    """Return a single-scene plotly figure of the Cher-scint asymmetry."""
    total = qc + qs
    qmax = float(total.max())
    lit = np.where(total > FRACTION_REL_THRESHOLD * qmax)[0]
    asym = (qc[lit] - qs[lit]) / total[lit]
    print(f"asym on {len(lit)} sensors range [{asym.min():.2f}, {asym.max():.2f}]")
    cb = dict(CB, colorbar_x=0.985, colorbar_len=0.55)
    return detector.visualize_event_data_plotly_discs(
        lit, asym, np.zeros(len(lit)),
        show_all_sensors=True, log_scale=False, show_colorbar=show_colorbar,
        dark_theme=False, plot_time=False, colorscale=DIVERGING,
        surface_color="lightgray", inactive_color="lightgray",
        inactive_opacity=0.15, cmin=-1.0, cmax=1.0,
        colorbar_tickvals=[-1.0, 0.0, 1.0], colorbar_ticktext=["-1", "0", "1"],
        return_fig=True, **cb)


def compose(f_data, f_pred, out_base):
    """Two scenes side by side, shared colorbar (on the prediction panel),
    exported as a tight vector PDF (+ PNG preview)."""
    from plotly.subplots import make_subplots
    sub = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "scene"}]],
                        horizontal_spacing=0.0)
    for tr in f_data.data:
        tr.update(scene="scene"); sub.add_trace(tr, row=1, col=1)
    for tr in f_pred.data:
        tr.update(scene="scene2"); sub.add_trace(tr, row=1, col=2)
    # copy each source scene (axis ranges, cube aspect, bg) into the subplot
    sd = f_data.layout.scene.to_plotly_json(); sd.pop("domain", None)
    sp = f_pred.layout.scene.to_plotly_json(); sp.pop("domain", None)
    sub.layout.scene.update(sd); sub.layout.scene2.update(sp)
    half = (1.0 - CBAR_GUTTER - PANEL_GAP) / 2.0
    sub.layout.scene.update(domain=dict(x=[0.0, half], y=[0.0, 1.0]),
                            camera=dict(eye=EYE))
    sub.layout.scene2.update(domain=dict(x=[half + PANEL_GAP, 2 * half + PANEL_GAP],
                                         y=[0.0, 1.0]), camera=dict(eye=EYE))
    sub.update_layout(width=2200, height=1100, showlegend=False,
                      paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=0, r=0, t=0, b=0))
    for ext in ("pdf", "png"):
        sub.write_image(f"{out_base}.{ext}", scale=2)
    print(f"saved {out_base}.pdf (+png)")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    # Cache the (deterministic) per-process charges so layout tweaks don't re-sim.
    cache = HERE / "data" / "juno_cher_scint_charges.npz"
    if cache.exists() and not os.environ.get("FORCE_SIM"):
        z = np.load(cache)
        qc_d, qs_d, qc_p, qs_p = z["qc_d"], z["qs_d"], z["qc_p"], z["qs_p"]
        print(f"loaded charges from {cache}")
    else:
        qc_d, qs_d = data_asymmetry()
        qc_p, qs_p = pred_asymmetry()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, qc_d=qc_d, qs_d=qs_d, qc_p=qc_p, qs_p=qs_p)
        print(f"saved charges to {cache}")
    detector = generate_detector(GEOM)
    f_data = panel_fig(detector, qc_d, qs_d, show_colorbar=False)   # left
    f_pred = panel_fig(detector, qc_p, qs_p, show_colorbar=True)    # right (shared bar)
    compose(f_data, f_pred, str(FIG / "juno_cher_scint_fraction"))


if __name__ == "__main__":
    main()
