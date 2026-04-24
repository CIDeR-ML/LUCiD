"""Diagnose the C-shape / partial-cap missing-photons artifact.

Runs three independent simulator paths for each of three mu- events at ~2 GeV
and renders PE + first-hit-time on the unwrapped detector surface:

  A) Existing v5 production output read straight from sensor h5 (reference).
  B) Whole-event data simulator (read_photon_data_from_photonsim path).
     All photons in one shot, no PAD, no per-particle split.
  C) Per-particle vmap (production code: generate_events_from_photonsim_particles).

If A & C show the artifact but B does not -> bug is in PAD / per-particle / vmap.
If all three show it -> bug is in _common_propagation / make_hits / sensor mapping.
If only A shows it -> bug is in the writer or PE_true aggregation (unlikely).

Run (inside the lucid_dev container with LUCiD bind-mounted at /opt/LUCiD):

    python /opt/LUCiD/scratch/diagnose_missing_photons.py
"""

import os
import sys
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist

sys.path.insert(0, "/opt/LUCiD")

from lucid.geometry import generate_detector
from lucid.sources.event_io import (
    read_photon_data_from_photonsim,
    read_particle_data_from_photonsim,
    get_max_photons_per_particle,
    generate_events_from_photonsim_particles,
    derive_event_keys,
)
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams

ROOT_FILE = "/tmp/v5-debug/three_muons.root"
V5_SENSOR_FILE = "/tmp/v5-s3df/01-hE/sensor/wc_sensor_0000.h5"
OUT_DIR = "/tmp/v5-debug/out"
GEOM_CONFIG = "/opt/LUCiD/config/SK_like_geom_config.json"
PHYSICS_CONFIG = "/opt/LUCiD/config/SK_like_physics_config.json"
N_EVENTS = 3
MASTER_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2D unwrap detector display (lifted from good_notebooks/cylinder_2D_displays.ipynb)
# Sparse-only; sensor_idx + PE + T in, figure out.
# ---------------------------------------------------------------------------

def _calc_min_distance(positions):
    dists = pdist(positions)
    return np.min(dists) if len(dists) > 0 else 1.0


def make_display_fn(json_filename):
    detector = generate_detector(json_filename)
    radius = detector.r
    height = detector.H
    sensor_positions = np.array(detector.all_points)
    sensor_cases = np.array([detector.ID_to_case[i] for i in range(len(sensor_positions))])
    n_sensors = len(sensor_positions)

    caps_offset = 1.05 * height / 2 + radius
    x = np.zeros(n_sensors)
    y = np.zeros(n_sensors)

    barrel = sensor_cases == 0
    theta = np.arctan2(sensor_positions[barrel, 1], sensor_positions[barrel, 0])
    theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
    x[barrel] = theta * radius * 2
    y[barrel] = sensor_positions[barrel, 2]

    top = sensor_cases == 1
    x[top] = sensor_positions[top, 0] + np.pi * radius
    y[top] = caps_offset + sensor_positions[top, 1]

    bot = sensor_cases == 2
    x[bot] = sensor_positions[bot, 0] + np.pi * radius
    y[bot] = -caps_offset - sensor_positions[bot, 1]

    xy = np.column_stack((x, y))
    min_d = _calc_min_distance(xy)
    diam = min_d
    xmin, xmax = x.min() - diam, x.max() + diam
    ymin, ymax = y.min() - diam, y.max() + diam

    def render(dense_values, ax, title, plot_time=False, perc_min=1.0, perc_max=99.0):
        pos = dense_values[dense_values > 0]
        if pos.size > 0:
            vmin = float(np.percentile(pos, perc_min))
            vmax = float(np.percentile(pos, perc_max))
        else:
            vmin, vmax = 0, 1
        cmap = plt.get_cmap("viridis_r" if plot_time else "viridis")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plot_vals = np.clip(dense_values, vmin, vmax)
        colors = sm.to_rgba(plot_vals)
        zero = dense_values <= 0
        colors[zero] = np.array([0.9, 0.9, 0.9, 1.0])

        ells = EllipseCollection(widths=diam, heights=diam, angles=0, units="x",
                                 facecolors=colors, offsets=xy,
                                 transOffset=ax.transData, edgecolors="none")
        ax.add_collection(ells)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        return sm

    return render, n_sensors, sensor_cases


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def photon_partition_check(root_path, entry):
    """Per-event consistency: total photons vs sum of per-particle photon-id sizes,
    range covered, orphan count."""
    import uproot
    with uproot.open(root_path) as rf:
        tree = rf["OpticalPhotons"]
        arr = tree.arrays(
            ["NParticles", "Particle_PhotonIDsSize", "Particle_PhotonIDsData", "PhotonPosX"],
            entry_start=entry, entry_stop=entry + 1, library="np")
    n_particles = int(arr["NParticles"][0])
    sizes = np.asarray(arr["Particle_PhotonIDsSize"][0], dtype=np.int64)
    data = np.asarray(arr["Particle_PhotonIDsData"][0], dtype=np.int64)
    n_photons_total = int(len(arr["PhotonPosX"][0]))
    n_assigned = int(sizes.sum())
    if len(data) > 0:
        idx_min = int(data.min())
        idx_max = int(data.max())
        unique_assigned = len(np.unique(data))
    else:
        idx_min = idx_max = unique_assigned = 0
    orphans = n_photons_total - unique_assigned
    return {
        "n_particles": n_particles,
        "n_photons_total": n_photons_total,
        "sum_PhotonIDsSize": n_assigned,
        "unique_indices_assigned": unique_assigned,
        "orphan_photons": orphans,
        "idx_range": (idx_min, idx_max),
        "per_particle_N": sizes.tolist(),
        "max_N": int(sizes.max()) if sizes.size else 0,
    }


def azimuthal_pe_hist(detector, pe_dense, n_bins=36):
    """PE summed over sensors in each azimuthal bin of the BARREL (ignoring caps)."""
    sensor_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])
    pos = np.array(detector.all_points)
    barrel = sensor_cases == 0
    theta = np.arctan2(pos[barrel, 1], pos[barrel, 0])  # (-pi, pi]
    pe = pe_dense[barrel]
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(theta, bins=bins, weights=pe)
    return hist, bins


def zone_pe(detector, pe_dense):
    sensor_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])
    barrel = pe_dense[sensor_cases == 0].sum()
    top = pe_dense[sensor_cases == 1].sum()
    bot = pe_dense[sensor_cases == 2].sum()
    return dict(barrel=float(barrel), top=float(top), bottom=float(bot))


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def run_path_A_whole_event(data_sim, entry_idx, sim_key):
    """Whole-event simulator call: all photons as one chunk, N = total."""
    photon_data = read_photon_data_from_photonsim(ROOT_FILE, entry_idx)
    n_total = int(len(photon_data["photon_origins"]))
    photon_data["N"] = n_total
    photon_data["apply_rotation"] = jnp.array(False)
    photon_data["rotation_axis"] = jnp.array([1.0, 0.0, 0.0])
    photon_data["rotation_angle"] = jnp.array(0.0)
    photon_data["apply_translation"] = jnp.array(False)
    photon_data["translation_vector"] = jnp.zeros(3)

    # Dummy particle params — in 'data' mode the simulator doesn't use the
    # track origin for propagation (photons come from the data); energy only
    # matters if downstream normalisation does, which it doesn't for is_data.
    track = ParticleParams.from_cartesian(
        energy=float(photon_data["energy"]),
        position=jnp.zeros(3),
        direction=jnp.array([0.0, 0.0, 1.0]),
        t0=0.0,
    )
    return data_sim(track, sim_key, photon_data), n_total


def run_path_C_via_production(apply_translation_flag, out_dir):
    """Run the real production pipeline on the new ROOT.

    Parameters
    ----------
    apply_translation_flag : bool
        Whether to apply per-event random vertex translation (the suspected
        regression trigger). False places all events at the origin.
    out_dir : str
        Destination directory for the v5 four-file batch.
    """
    import numpy as _np
    from lucid.geometry.detector_geometry import DetectorGeometry

    os.makedirs(out_dir, exist_ok=True)

    det_geom = DetectorGeometry.from_config(GEOM_CONFIG)
    sensor_positions = _np.asarray(det_geom.sensor_points, dtype=_np.float32)

    simulate_event = setup_event_simulator(
        GEOM_CONFIG, 0, K=12, is_data=True, temperature=0.0,
        apply_smearing=False, physics_config=PHYSICS_CONFIG,
        default_detector_params=True,
    )

    generate_events_from_photonsim_particles(
        event_simulator=simulate_event,
        root_file_path=ROOT_FILE,
        sensor_positions=sensor_positions,
        output_dir=out_dir,
        n_events=N_EVENTS,
        batch_size=N_EVENTS,
        master_seed=MASTER_SEED,
        job_id=1,
        apply_smearing=True,
        apply_rotation=False,
        apply_translation=apply_translation_flag,
        detector_config_path=GEOM_CONFIG,
        dataset_name=f"diag_three_muons_{'translated' if apply_translation_flag else 'origin'}",
        run_id=None,
        file_index_start=0,
        detector_type="cylinder",
        material="water",
        include_track_segments=True,
        primary_source="particles",
    )
    return out_dir


def read_vertex_from_labl(labl_h5, evt_idx):
    """Fetch the drawn vertex written into per_interaction/{vertex_x,y,z} for one event."""
    with h5py.File(labl_h5, "r") as f:
        g = f[f"event_{evt_idx:03d}"]["per_interaction"]
        vx = float(g["vertex_x"][0])
        vy = float(g["vertex_y"][0])
        vz = float(g["vertex_z"][0])
    return np.array([vx, vy, vz], dtype=np.float32)


# ---------------------------------------------------------------------------
# Load rendering artifacts
# ---------------------------------------------------------------------------

def load_sparse_v5(sensor_h5_path, event_key):
    with h5py.File(sensor_h5_path, "r") as f:
        g = f[event_key]
        idx = np.asarray(g["sensor_idx"][()], dtype=np.int64)
        pe = np.asarray(g["PE"][()], dtype=np.float32)
        t = np.asarray(g["T"][()], dtype=np.float32)
    return idx, pe, t


def sparse_to_dense(idx, vals, n_sensors):
    out = np.zeros(n_sensors, dtype=np.float32)
    out[idx] = vals
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"Diagnostic: C-shape / partial-caps missing-photons")
    print(f"ROOT:     {ROOT_FILE}")
    print(f"v5 ref:   {V5_SENSOR_FILE}")
    print(f"Geom:     {GEOM_CONFIG}")
    print("=" * 70)

    detector = generate_detector(GEOM_CONFIG)
    render_fn, n_sensors, sensor_cases = make_display_fn(GEOM_CONFIG)
    print(f"n_sensors = {n_sensors}")
    print(f"  barrel={int((sensor_cases==0).sum())} "
          f"top={int((sensor_cases==1).sum())} "
          f"bot={int((sensor_cases==2).sum())}")
    print()

    # ---- Path C, twice: origin + translated ----
    print("-" * 70)
    print("[C-origin]     Running production path with apply_translation=False…")
    prod_out_origin = run_path_C_via_production(False, "/tmp/v5-debug/prod_out_origin")
    print(f"  -> {prod_out_origin}")
    print()
    print("[C-translated] Running production path with apply_translation=True…")
    prod_out_trans = run_path_C_via_production(True, "/tmp/v5-debug/prod_out_translated")
    print(f"  -> {prod_out_trans}")
    print()

    origin_sensor = f"{prod_out_origin}/sensor/wc_sensor_0000.h5"
    trans_sensor = f"{prod_out_trans}/sensor/wc_sensor_0000.h5"
    trans_labl = f"{prod_out_trans}/labl/wc_labl_0000.h5"

    # ---- Per-event loop ----
    for evt_idx in range(N_EVENTS):
        print("-" * 70)
        print(f"EVENT {evt_idx}")
        print("-" * 70)

        # Partition check from the ROOT
        part = photon_partition_check(ROOT_FILE, evt_idx)
        print(f"  ROOT partition check:")
        print(f"    n_particles        = {part['n_particles']}")
        print(f"    n_photons_total    = {part['n_photons_total']:,}")
        print(f"    per-particle N     = {part['per_particle_N']}")

        # Vertex drawn for translated run
        vtx = read_vertex_from_labl(trans_labl, evt_idx)
        print(f"  translated vertex (m): ({vtx[0]:.2f}, {vtx[1]:.2f}, {vtx[2]:.2f})")
        print(f"    |r_xy| = {float(np.hypot(vtx[0], vtx[1])):.2f}  "
              f"(0.9*R = {0.9*16.9:.2f})   |z| = {abs(vtx[2]):.2f}  "
              f"(0.45*H = {0.45*36.2:.2f})")

        # Load origin + translated outputs
        idx_o, pe_o_s, t_o_s = load_sparse_v5(origin_sensor, f"event_{evt_idx:03d}")
        pe_o = sparse_to_dense(idx_o, pe_o_s, n_sensors)
        t_o = sparse_to_dense(idx_o, t_o_s, n_sensors)
        idx_t, pe_t_s, t_t_s = load_sparse_v5(trans_sensor, f"event_{evt_idx:03d}")
        pe_t = sparse_to_dense(idx_t, pe_t_s, n_sensors)
        t_t = sparse_to_dense(idx_t, t_t_s, n_sensors)

        print(f"  [origin]     total PE={pe_o.sum():.1f}  #sensors hit={(pe_o>0).sum()}  "
              f"zones={zone_pe(detector, pe_o)}")
        print(f"  [translated] total PE={pe_t.sum():.1f}  #sensors hit={(pe_t>0).sum()}  "
              f"zones={zone_pe(detector, pe_t)}")

        hist_o, _ = azimuthal_pe_hist(detector, pe_o)
        hist_t, _ = azimuthal_pe_hist(detector, pe_t)
        print(f"  Azimuthal PE (barrel, 36 bins, deg -180..180):")
        print(f"    [origin]     {np.array2string(hist_o, precision=0, separator=',')}")
        print(f"    [translated] {np.array2string(hist_t, precision=0, separator=',')}")
        print(f"    origin zero bins: {int((hist_o<=0).sum())}  "
              f"translated zero bins: {int((hist_t<=0).sum())}")

        # -- Render side-by-side --
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        sm0 = render_fn(pe_o, axes[0, 0], f"event {evt_idx}  [origin]     PE")
        sm1 = render_fn(t_o, axes[0, 1], f"event {evt_idx}  [origin]     first-hit T", plot_time=True)
        sm2 = render_fn(pe_t, axes[1, 0],
                        f"event {evt_idx}  [translated]  PE   vertex=({vtx[0]:.1f},{vtx[1]:.1f},{vtx[2]:.1f})")
        sm3 = render_fn(t_t, axes[1, 1], f"event {evt_idx}  [translated]  first-hit T", plot_time=True)
        for ax, sm in zip(axes.ravel(), (sm0, sm1, sm2, sm3)):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(sm, cax=cax)
        plt.tight_layout()
        out_png = f"{OUT_DIR}/event_{evt_idx:03d}_origin_vs_translated.png"
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {out_png}")
        print()

    # ---- Also render the three v5-existing events for reference ----
    print("-" * 70)
    print("[B] Existing v5 production output (reference) — /tmp/v5-s3df/01-hE/")
    for ev_idx in range(N_EVENTS):
        evk = f"event_{ev_idx:03d}"
        idx, pe_s, t_s = load_sparse_v5(V5_SENSOR_FILE, evk)
        pe = sparse_to_dense(idx, pe_s, n_sensors)
        t = sparse_to_dense(idx, t_s, n_sensors)
        print(f"  v5 {evk}: total PE={pe.sum():.1f}  #sensors hit={(pe>0).sum()}  "
              f"zones={zone_pe(detector, pe)}")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sm0 = render_fn(pe, axes[0], f"v5 {evk}  PE")
        sm1 = render_fn(t, axes[1], f"v5 {evk}  first-hit T", plot_time=True)
        for ax, sm in zip(axes, (sm0, sm1)):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(sm, cax=cax)
        plt.tight_layout()
        out_png = f"{OUT_DIR}/v5_ref_{ev_idx:03d}.png"
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {out_png}")

    print("=" * 70)
    print(f"All PNGs in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
