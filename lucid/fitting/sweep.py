"""Recon characterization sweep over (particle, energy, nph, event, true-pose).

Drives the production recon (``ReconModel`` + ``fit_track``) across a grid of physics + nuisance axes
and records per-fit truth/seed/result/error rows for resolution analysis. The key axis beyond a plain
event scan is **true pose**: ``n_poses`` independent random ``rand_tf`` placements per GEANT4 shower, so
the same physics event is reconstructed at many true vertex positions / directions — exposing the
position- and angle-dependence of the resolution, not just per-event photon noise.

Loop is ordered for compile reuse: the simulator is built ONCE per ``(particle, nph)`` (the expensive
JIT), then reused across all ``(energy, event, pose)`` fits. Work is sharded for multi-process runs by
splitting the ``(particle, nph)``-sorted work-list into contiguous blocks, so each worker rebuilds the
sim only a handful of times.

Plain Python API::

    from lucid.fitting.sweep import run_sweep, aggregate
    run_sweep("output/sweep1", particles=["muon"], energies=[1000], nphs=[150000],
              n_events=20, n_poses=4)                       # one (sub)shard
    aggregate("output/sweep1")                              # merge shards -> results.npy

Command line (``python -m``, no pip install / console entry needed)::

    python -m lucid.fitting.sweep --out output/sweep1 --particles muon \
        --energies 1000 --nphs 150000 --n-events 20 --n-poses 4 --niters 150
    python -m lucid.fitting.sweep --out output/sweep1 --aggregate
    # multi-GPU: one process per GPU, then aggregate
    for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g python -m lucid.fitting.sweep \
        --out output/sweep1 --shard $g --nshards 3 ... & done; wait
    python -m lucid.fitting.sweep --out output/sweep1 --aggregate

Output (per shard, dependency-free): ``shard_<i>.npy`` = an object array of flat dict rows; a single
``meta.json`` stamps the resolved arguments + recipe + git commit. ``aggregate()`` merges shards into
one ``results.npy`` (object array of rows) for groupby analysis. For multi-GPU runs, call ``run_sweep``
from N processes with ``shard=i, nshards=N`` (one ``CUDA_VISIBLE_DEVICES`` each), then ``aggregate``.
"""
import os, json, time, subprocess

# Recipe = the committed default (5312c93 + the time_weight knob). Override (partially) via run_sweep(recipe=...).
DEFAULT_RECIPE = dict(lr=4.0, lr_final=1.5, ridge_i=0.1, lam=0.01,
                      nkeys=8, niters=150, refresh=8, time_weight=1.0, trust=3.0)
POSE_STRIDE = 1000          # pose_seed = pose_seed_base + event*STRIDE + pose  (stable across configs)
DEFAULT_DATA_ROOT = "/sdf/group/neutrino/omara/LUCiD_dlcheck/data/water/{particle}/{energy}MeV_100events.root"
DEFAULT_GEOM = "config/SK_like_geom_config.json"
DEFAULT_PHYS = "config/SK_like_physics_config.json"
DEFAULT_BAND = (274.91, 673.83)


def _enumerate_work(cfg):
    """Full work-list of (particle, nph, energy, event, pose) dicts, SORTED by (particle, nph) first so
    contiguous shards stay within a few simulator builds."""
    events = cfg.get("events") or list(range(cfg["n_events"]))
    work = [dict(particle=p, nph=int(n), energy=int(e), event=int(ev), pose=int(po))
            for p in cfg["particles"] for n in cfg["nphs"]
            for e in cfg["energies"] for ev in events for po in range(cfg["n_poses"])]
    work.sort(key=lambda w: (w["particle"], w["nph"], w["energy"], w["event"], w["pose"]))
    return work


def _git_commit(root):
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------------------
# Forward / data helpers (kept faithful to the validated campaign workers)
# --------------------------------------------------------------------------------------------------
def _make_engine(cfg):
    """Build the import-time-heavy pieces once: detector, default DetectorParams, and a per-(particle,nph)
    simulator factory. Returns a dict of reusable handles + a ``build(particle, nph)`` closure."""
    import numpy as np, jax.numpy as jnp
    import lucid.simulation.simulator as SIM
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import load_detector_params
    from lucid.fitting import ReconModel
    from lucid.optimization.grid_search import get_detector_bounds

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    geom = cfg["geom"] if os.path.isabs(cfg["geom"]) else os.path.join(root, cfg["geom"])
    phys = cfg["phys"] if os.path.isabs(cfg["phys"]) else os.path.join(root, cfg["phys"])

    _orig = SIM.unpack_siren_params                                   # importance seed sampling: keeps
    def _patched(p, m):                                               # the full angular tail (threshold
        c = dict(_orig(p, m))                                         # unused). Explicit so recon is
        c["ray_sampling"] = {**c.get("ray_sampling", {}), "seed_mode": "importance"}  # robust to the code default.
        return c
    SIM.unpack_siren_params = _patched

    det = generate_detector(geom); ND = len(det.all_points)
    POS = np.asarray(det.all_points); bounds = get_detector_bounds(det)
    dp = load_detector_params(phys, num_sensors=ND)
    dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(float(cfg["tts"]))))
    K, NBUF, band = int(cfg["K"]), int(cfg["nbuf"]), tuple(cfg["cherenkov_band"])

    def build(particle, nph):
        data_sim = setup_event_simulator(geom, NBUF, temperature=None, K=K, is_data=True, hit_mode="realistic",
            physics_config=phys, default_detector_params=dp, particle=particle, wavelength_mode=True,
            charge_resolution=None, max_candidates_per_ray=4)
        pred = setup_event_simulator(geom, int(nph), temperature=0.1, K=K, hit_mode="per_photon",
            physics_config=phys, default_detector_params=True, particle=particle, wavelength_mode=True,
            pos_grad_threshold=K, n_grad_iters=K, cherenkov_emission_band=band, max_candidates_per_ray=4)
        # energy_from_scale (mode 'simtotal'): the charge energy-gradient is routed through the
        # SIM's total charge, not the shape — removes the profile-bias that a small emission-shape
        # misspecification otherwise puts on energy via the soft (E↔geometry) degeneracy.
        model = ReconModel(pred, ND, sigma=float(cfg["tts"]), delta=1.0,
                           time_weight=float(cfg["recipe"].get("time_weight", 1.0)),
                           energy_from_scale=True, energy_scale_mode="simtotal")
        return data_sim, pred, model
    return dict(geom=geom, phys=phys, ND=ND, POS=POS, bounds=bounds, K=K, NBUF=NBUF, build=build, det=det, root=root)


def _read_event(root_path, ev):
    import numpy as np, jax.numpy as jnp, uproot
    f = uproot.open(root_path); raw = f["OpticalPhotonsRaw"]; eid = raw["EventID"].array(library="np")
    idx = np.where(eid == ev)[0]; lo, hi = int(idx.min()), int(idx.max()) + 1
    br = ["PhotonPosX", "PhotonPosY", "PhotonPosZ", "PhotonDirX", "PhotonDirY", "PhotonDirZ", "PhotonTime", "PhotonWavelength"]
    d = raw.arrays(br, entry_start=lo, entry_stop=hi, library="np")
    cat = lambda b: np.concatenate([np.asarray(x, np.float64) for x in d[b]])
    pos = np.column_stack((cat("PhotonPosX") / 10, cat("PhotonPosY") / 10, cat("PhotonPosZ") / 10))
    dr = np.column_stack((cat("PhotonDirX"), cat("PhotonDirY"), cat("PhotonDirZ")))
    E = float(f["OpticalPhotons"]["PrimaryEnergy"].array(library="np")[ev])
    return dict(photon_origins=jnp.asarray(pos), photon_directions=jnp.asarray(dr),
                photon_times=jnp.asarray(cat("PhotonTime")), energy=E, wavelengths=jnp.asarray(cat("PhotonWavelength")))


def _rotax(u, deg):
    import numpy as np
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def _rand_tf(raw, pose_seed, fidr, fidz):
    """A random true pose: isotropic direction + uniform-in-cylinder vertex, seeded by pose_seed."""
    import numpy as np
    rng = np.random.default_rng(pose_seed)
    beta = np.degrees(np.arccos(rng.uniform(-1, 1))); al = rng.uniform(0, 2 * np.pi)
    axis = np.array([-np.sin(al), np.cos(al), 0.]); rr = fidr * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-fidz, fidz)]) * 100.
    raw = dict(raw); O = np.asarray(raw["photon_origins"]).astype(float); D = np.asarray(raw["photon_directions"]).astype(float)
    R = _rotax(axis, beta); c = O.mean(0)
    raw["photon_origins"] = (O - c) @ R.T + c + sh; raw["photon_directions"] = D @ R.T
    return raw, ((np.zeros(3) - c) @ R.T + c + sh) / 100., np.array([0., 0., 1.]) @ R.T


def _pad(raw, nbuf):
    import numpy as np, jax.numpy as jnp
    n = int(raw["photon_origins"].shape[0]); reps = int(np.ceil(nbuf / n))
    tl = lambda a: jnp.asarray(np.tile(np.asarray(a), (reps,) + (1,) * (np.asarray(a).ndim - 1))[:nbuf])
    return dict(photon_origins=tl(raw["photon_origins"]), photon_directions=tl(raw["photon_directions"]),
                photon_times=tl(raw["photon_times"]), N=jnp.asarray(n), apply_rotation=False,
                rotation_axis=jnp.array([1., 0., 0.]), rotation_angle=jnp.array(0.), apply_translation=False,
                translation_vector=jnp.zeros(3), wavelengths=tl(raw["wavelengths"]))


def _basis(d):
    import numpy as np
    d = d / np.linalg.norm(d); a = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1); return d, e1, e2


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------
def run_sweep(out_dir, *, particles=("muon", "electron"), energies=(500, 1000, 1500),
              nphs=(150000,), n_events=100, n_poses=1, events=None, recipe=None,
              data_root=DEFAULT_DATA_ROOT, geom=DEFAULT_GEOM, phys=DEFAULT_PHYS,
              tts=2.5, fidr=12.0, fidz=12.0, K=8, nbuf=600_000, cherenkov_band=DEFAULT_BAND,
              pose_seed_base=100003, data_key_base=7000, chk=5, shard=0, nshards=1):
    """Run shard ``shard`` of ``nshards`` over the (particle, energy, nph, event, true-pose) grid and save
    ``out_dir/shard_<shard>.npy`` (object array of flat dict rows). Returns the rows.

    Axes
    ----
    particles, energies, nphs : the physics + photon-budget grid.
    n_events / events : number of GEANT4 showers per (particle,energy), or an explicit event-id list.
    n_poses : random true placements (``rand_tf``) per shower — the "different true initializations".

    Knobs
    -----
    recipe : optimizer recipe dict; merged over ``DEFAULT_RECIPE`` (so partial overrides work).
    data_root : ROOT path template with ``{particle}``/``{energy}``. tts/fidr/fidz/K/nbuf/cherenkov_band :
    forward + fiducial settings. pose_seed_base/data_key_base : reproducibility seeds. chk : convergence
    checkpoint stride. shard/nshards : split the work-list across processes (one GPU each).
    """
    cfg = dict(particles=list(particles), energies=[int(e) for e in energies], nphs=[int(n) for n in nphs],
               n_events=int(n_events), n_poses=int(n_poses), events=list(events) if events else None,
               recipe={**DEFAULT_RECIPE, **(recipe or {})}, data_root=data_root, geom=geom, phys=phys,
               tts=float(tts), fidr=float(fidr), fidz=float(fidz), K=int(K), nbuf=int(nbuf),
               cherenkov_band=list(cherenkov_band), pose_seed_base=int(pose_seed_base),
               data_key_base=int(data_key_base), chk=int(chk))
    return _run_shard(cfg, int(shard), int(nshards), out_dir)


def _run_shard(cfg, shard, nshards, out_dir):
    import numpy as np, jax, jax.numpy as jnp
    from lucid.fitting import fit_track, vec9_dir, track_from_vec9
    from lucid.fitting.recon import vec9_from_track
    from lucid.optimization.grid_search import hierarchical_position_grid_search
    from lucid.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization

    os.makedirs(out_dir, exist_ok=True)
    rec = cfg["recipe"]; eng = _make_engine(cfg); POS, bounds, ND, NBUF = eng["POS"], eng["bounds"], eng["ND"], eng["NBUF"]
    POSf = jnp.asarray(POS)
    work = _enumerate_work(cfg)
    lo = (len(work) * shard) // nshards; hi = (len(work) * (shard + 1)) // nshards   # contiguous block
    mine = work[lo:hi]
    if shard == 0:
        meta = dict(args=cfg, git=_git_commit(eng["root"]), n_work=len(work), nshards=nshards,
                    schema="see lucid.fitting.sweep")
        with open(os.path.join(out_dir, "meta.json"), "w") as fh: json.dump(meta, fh, indent=2)
    print(f"[shard {shard}/{nshards}] {len(mine)} fits (work {lo}:{hi} of {len(work)})", flush=True)

    rows = []; cur = (None, None); data_sim = pred = model = None; rkey = jax.random.PRNGKey
    for i, w in enumerate(mine):
        if (w["particle"], w["nph"]) != cur:                                  # (re)build sim on group change
            data_sim, pred, model = eng["build"](w["particle"], w["nph"]); cur = (w["particle"], w["nph"])
        rp = cfg["data_root"].format(particle=w["particle"], energy=w["energy"])
        raw0 = _read_event(rp, w["event"])
        pose_seed = cfg["pose_seed_base"] + w["event"] * POSE_STRIDE + w["pose"]
        raw, vtx, dirt = _rand_tf(raw0, pose_seed, cfg["fidr"], cfg["fidz"])
        pol = float(np.arccos(np.clip(dirt[2], -1, 1))); az = float(np.arctan2(dirt[1], dirt[0])); E = float(raw["energy"])
        th9 = np.array([E, vtx[0], vtx[1], vtx[2], np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
        dkey = rkey(cfg["data_key_base"] + w["event"] * POSE_STRIDE + w["pose"])
        c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), dkey, _pad(raw, NBUF)))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.); ocf, otf = jnp.asarray(oc), jnp.asarray(ot)
        try:
            e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1 / jnp.sqrt(3)), jnp.pi / 4, 0.,
                                          POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)["best_energy"]
            p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0., 0., bounds,
                                                   n_div=5, t0_n_div=5, levels=6, verbosity=0)
            c2 = hierarchical_direction_search_cone(pred, jnp.asarray(p1["best_position"]), float(p1["best_t0"]),
                                                    POSf, otf, ocf, (ocf, otf), e0, 3, 8, 90., 0.5, 0)
            dg = np.array([np.sin(c2["best_theta"]) * np.cos(c2["best_phi"]),
                           np.sin(c2["best_theta"]) * np.sin(c2["best_phi"]), np.cos(c2["best_theta"])])
            seed = vec9_from_track(e0, np.asarray(p1["best_position"]), dg, t0=float(p1["best_t0"]))
            ts = time.time()
            res, H = fit_track(model, oc, ot, seed, nkeys=rec["nkeys"], niters=rec["niters"], lr=rec["lr"],
                               lr_final=rec["lr_final"], ridge_i=rec["ridge_i"], lam=rec["lam"],
                               refresh=rec["refresh"], refresh_final=rec.get("refresh_final"),
                               refresh_switch=rec.get("refresh_switch", 0.5), fisher_mode="ad", hist=True,
                               trust=rec.get("trust"))
            wall = time.time() - ts
            traj = np.asarray(H["traj"]); dl, e1, e2 = _basis(dirt)
            def lt(x):
                dv = (x[1:4] - th9[1:4]) * 100; return float(abs(dv @ dl)), float(np.hypot(dv @ e1, dv @ e2))
            vlong, vtran = lt(res); vtot = float(np.hypot(vlong, vtran))
            chk = list(range(0, rec["niters"] + 1, cfg["chk"]))
            vtraj = np.array([np.linalg.norm((traj[min(j, len(traj) - 1)] - th9)[1:4]) * 100 for j in chk])
            thr = max(vtot * 1.5, vtot + 5.); conv = next((chk[k] for k in range(len(chk)) if np.all(vtraj[k:] <= thr)), rec["niters"])
            row = dict(particle=w["particle"], energy=w["energy"], nph=w["nph"], event=w["event"], pose=w["pose"],
                       pose_seed=pose_seed, E_true=E, vx_true=float(vtx[0]), vy_true=float(vtx[1]), vz_true=float(vtx[2]),
                       dx_true=float(dirt[0]), dy_true=float(dirt[1]), dz_true=float(dirt[2]),
                       R_xy=float(np.hypot(vtx[0], vtx[1])), absz=float(abs(vtx[2])), cos_zen=float(dirt[2]),
                       seed_vtx_err=float(np.linalg.norm((seed - th9)[1:4]) * 100), seed_E=float(e0),
                       seed_dir_err=float(np.degrees(np.arccos(np.clip(vec9_dir(seed) @ dirt, -1, 1)))),
                       E_fit=float(res[0]), t0_fit=float(res[8]),
                       vtx_err=vtot, vtx_long=vlong, vtx_trans=vtran,
                       dir_deg=float(np.degrees(np.arccos(np.clip(vec9_dir(res) @ dirt, -1, 1)))),
                       dE=float(res[0] - E), dt0=float(res[8]), conv_iter=int(conv), wall=float(wall),
                       n_hit=int((oc > 0).sum()), total_charge=float(oc.sum()),
                       # full fitted/seed 9-vecs + signed truth frame: keeps the row re-scorable
                       # (vtx_long above is |·|; the signed value needs the vectors)
                       fit9=[float(x) for x in res], seed9=[float(x) for x in seed],
                       truth9=[float(x) for x in th9], dir_true=[float(x) for x in dirt])
        except Exception as e:                                               # never lose a shard to one bad fit
            row = dict(particle=w["particle"], energy=w["energy"], nph=w["nph"], event=w["event"], pose=w["pose"],
                       error=str(e)[:300])
        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(mine):
            np.save(os.path.join(out_dir, f"shard_{shard:03d}.npy"), np.array(rows, dtype=object), allow_pickle=True)
            print(f"[shard {shard}] {i+1}/{len(mine)} done", flush=True)
    np.save(os.path.join(out_dir, f"shard_{shard:03d}.npy"), np.array(rows, dtype=object), allow_pickle=True)
    print(f"[shard {shard}] DONE ({len(rows)} rows)", flush=True)
    return rows


def aggregate(out_dir):
    """Merge shard_*.npy -> results.npy (object array of rows); dedup by (particle,energy,nph,event,pose)."""
    import glob, numpy as np
    seen = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "shard_*.npy"))):
        for r in np.load(f, allow_pickle=True):
            seen[(r.get("particle"), r.get("energy"), r.get("nph"), r.get("event"), r.get("pose"))] = r
    rows = list(seen.values())
    np.save(os.path.join(out_dir, "results.npy"), np.array(rows, dtype=object), allow_pickle=True)
    ok = [r for r in rows if "error" not in r]
    print(f"aggregated {len(rows)} rows ({len(rows)-len(ok)} errored) -> {out_dir}/results.npy", flush=True)
    return rows


# --------------------------------------------------------------------------------------------------
# Lightweight CLI: `python -m lucid.fitting.sweep ...` (NOT a pip console entry — no install needed).
# Thin wrapper over run_sweep/aggregate; every sweep arg is a flag, recipe knobs override DEFAULT_RECIPE.
# --------------------------------------------------------------------------------------------------
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m lucid.fitting.sweep",
                                description="Recon characterization sweep (run via python -m; no install).")
    p.add_argument("--out", help="output dir")
    p.add_argument("--aggregate", action="store_true", help="just merge shards in --out and exit")
    p.add_argument("--particles", nargs="+", default=["muon", "electron"])
    p.add_argument("--energies", nargs="+", type=int, default=[500, 1000, 1500])
    p.add_argument("--nphs", nargs="+", type=int, default=[150000])
    p.add_argument("--n-events", type=int, default=100)
    p.add_argument("--n-poses", type=int, default=1)
    p.add_argument("--events", nargs="+", type=int, default=None, help="explicit event-id list (overrides --n-events)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--data-root"); p.add_argument("--tts", type=float)
    # recipe overrides (applied only if given) — keep these in sync with DEFAULT_RECIPE
    p.add_argument("--lr", type=float); p.add_argument("--lr-final", type=float); p.add_argument("--ridge", type=float)
    p.add_argument("--nkeys", type=int); p.add_argument("--niters", type=int); p.add_argument("--refresh", type=int)
    p.add_argument("--refresh-final", type=int, help="annealed refresh: switch to this (smaller=fresher) late")
    p.add_argument("--refresh-switch", type=float, help="fraction of niters at which to switch refresh (default 0.5)")
    p.add_argument("--time-weight", type=float)
    a = p.parse_args(argv)
    if not a.out:
        p.error("--out is required")
    if a.aggregate:
        aggregate(a.out); return
    recipe = {k: v for k, v in dict(lr=a.lr, lr_final=a.lr_final, ridge_i=a.ridge, nkeys=a.nkeys,
                                    niters=a.niters, refresh=a.refresh, refresh_final=a.refresh_final,
                                    refresh_switch=a.refresh_switch, time_weight=a.time_weight).items() if v is not None}
    kw = dict(particles=a.particles, energies=a.energies, nphs=a.nphs, n_events=a.n_events, n_poses=a.n_poses,
              events=a.events, shard=a.shard, nshards=a.nshards, recipe=recipe)
    if a.data_root:
        kw["data_root"] = a.data_root
    if a.tts is not None:
        kw["tts"] = a.tts
    run_sweep(a.out, **kw)


if __name__ == "__main__":
    main()
