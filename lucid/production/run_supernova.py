"""Run sntools to produce a gRooTracker file of supernova-burst particles.

Called from run_job.py before the PhotonSim step when
``config['primary_source'] == 'supernova'``. Produces
``<output_dir>/gntp_job_<NNNNNN>.gtrac.root`` — the *same* rooTracker file
GENIE writes — so PhotonSim consumes it through the existing
``/gun/genieInput`` path and no PhotonSim C++ change is needed.

sntools (github.com/SNEWS2/sntools) Monte-Carlo–samples neutrino
interactions in a water Cherenkov detector from a supernova flux × cross
section. We drive it once per ``(model, ordering)`` subcase:

* the **mass ordering** is sntools' ``--transformation`` — NMO → the
  built-in ``AdiabaticMSW_NMO``, IMO → ``AdiabaticMSW_IMO`` (no SNEWPY
  download needed for the MSW transformations);
* the **model** is an sntools flux ``(flux_file, format)``. For the physics
  default ``SNEWPY-<Model>`` (e.g. ``SNEWPY-Nakazato_2013``), ``flux_file`` names
  a model file under the SNEWPY cache (``~/.cache/astropy/snewpy/models/<Model>/``)
  which must be **pre-downloaded once** — compute nodes may lack network::

      python -c "import astropy.units as u; from snewpy.models.ccsn import Nakazato_2013; \
          Nakazato_2013(progenitor_mass=20*u.Msun, revival_time=100*u.ms, metallicity=0.02, eos='shen')"

  Non-SNEWPY formats (``gamma``, ``nakazato`` ASCII) take a repo-relative path.

Each sntools event becomes one rooTracker entry: the incoming neutrino is
written as a ``StdHepStatus==0`` track (PhotonSim records it as the true
``neutrino_pdg`` / ``neutrino_energy``) and every outgoing final-state
particle as ``StdHepStatus==1`` (a G4 primary). sntools' vertex/time are
dropped — PhotonSim fires from the origin and LUCiD assigns the fiducial
vertex, exactly as for GENIE.

sntools is invoked as ``sys.executable -m sntools.genevts`` so a CFS
``PYTHONUSERBASE`` install (see user_paths.sh SN_ENV_BASE) is picked up via
the container Python without a PATH dance.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Rest masses in MeV for the final-state particles the water channels emit.
# Used to recover |p| = sqrt(E^2 - m^2) from sntools' total-energy tracks.
PARTICLE_MASS_MeV = {
    11: 0.510999, -11: 0.510999,   # e-, e+
    22: 0.0,                        # gamma
    2112: 939.565420,               # neutron
    2212: 938.272088,               # proton
}
NEUTRINO_PDGS = {12, -12, 14, -14, 16, -16}

# Ordering label → sntools transformation. The AdiabaticMSW variants are
# built into sntools (no SNEWPY dependency); pass-through is allowed for any
# explicit sntools transformation name (e.g. "SNEWPY-...").
ORDERING_TO_TRANSFORMATION = {
    "NMO": "AdiabaticMSW_NMO",
    "IMO": "AdiabaticMSW_IMO",
    "NO": "AdiabaticMSW_NMO",
    "IO": "AdiabaticMSW_IMO",
    "normal": "AdiabaticMSW_NMO",
    "inverted": "AdiabaticMSW_IMO",
    "noosc": "NoTransformation",
    "none": "NoTransformation",
}

# All water interaction channels sntools supports (its default when --channel
# is omitted). Kept explicit so a config can request the full set by name.
WATER_CHANNELS = ("ibd", "es", "o16e", "o16eb")

KMAX_PARTICLES = 4096  # must match PhotonSim RooTrackerReader::kMaxParticles

# sntools' per-event NUANCE channel code (the `$ nuance <code>` field) → name.
# Codes are sntools' interaction_channels/*.py Event(...) values. The rooTracker
# (StdHep) has no channel slot, so we carry the code in a sidecar and stamp it
# into the labl truth (per_interaction/interaction_channel + /channel).
CHANNEL_CODE_TO_NAME = {
    -1001001: "ibd",                 # inverse beta decay
    98: "es", -98: "es",             # neutrino-electron elastic scattering
    1008016: "o16e",                 # nu_e + 16O CC
    -1008016: "o16eb",               # nu_e-bar + 16O CC
    2001001: "ps", -2001001: "ps",   # proton scattering (LS)
    1006012: "c12e", -1006012: "c12eb",
    2006012: "c12nc", -2006012: "c12nc",
    0: "unknown",
}


class SupernovaError(RuntimeError):
    """Raised when the sntools run or the rooTracker conversion fails."""


# Repo root (…/LUCiD), used to resolve repo-relative flux_file paths so a
# config can ship a flux under the checkout and work both on the host and
# inside the container (where the checkout binds to /opt/LUCiD).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_flux_file(flux_file: str) -> Path:
    """Resolve a flux path: absolute, cwd-relative, or repo-relative."""
    for cand in (Path(flux_file), _REPO_ROOT / flux_file):
        if cand.is_file():
            # Absolute — sntools runs with cwd=output_dir, so a relative path
            # (resolved here against the caller's cwd) would not survive.
            return cand.resolve()
    raise SupernovaError(
        f"flux_file not found: {flux_file!r} (tried as-is and relative to {_REPO_ROOT})")


def _transformation_for(ordering: str) -> str:
    """Map an ordering label to an sntools --transformation name."""
    if ordering in ORDERING_TO_TRANSFORMATION:
        return ORDERING_TO_TRANSFORMATION[ordering]
    # Allow passing an explicit sntools transformation name verbatim.
    if ordering.startswith(("AdiabaticMSW", "NoTransformation", "SNEWPY-")):
        return ordering
    raise SupernovaError(
        f"Unknown ordering {ordering!r}. Use one of {sorted(ORDERING_TO_TRANSFORMATION)} "
        f"or an explicit sntools transformation name."
    )


def _resolve_model(config_sn: dict, model_name: Optional[str]) -> dict:
    """Select the model entry to run from the supernova block's ``models`` list.

    Each entry is ``{"name": <slug>, "format": <sntools format>,
    "flux_file": <path>}``. When ``model_name`` is given (the fan-out passes
    one per subcase) it must match an entry's ``name``; when omitted and there
    is exactly one model, that one is used.
    """
    models = config_sn.get("models")
    if not models:
        raise SupernovaError("supernova block missing non-empty 'models' list")
    if model_name is None:
        if len(models) == 1:
            return models[0]
        raise SupernovaError(
            f"multiple models present ({[m.get('name') for m in models]}); "
            f"a specific --sn-model must be selected")
    for m in models:
        if m.get("name") == model_name:
            return m
    raise SupernovaError(
        f"model {model_name!r} not found in supernova.models "
        f"({[m.get('name') for m in models]})")


def _build_sntools_cmd(
    *, flux_file: str, fmt: str, detector: str, transformation: str,
    distance: float, channel: Optional[str], starttime: Optional[float],
    endtime: Optional[float], seed: int, out_kin: Path,
) -> list[str]:
    """Assemble the ``python -m sntools.genevts ...`` command for one run."""
    cmd = [
        sys.executable, "-m", "sntools.genevts", str(flux_file),
        "--format", fmt,
        "--detector", detector,
        "--transformation", transformation,
        "--distance", f"{distance:g}",
        "--mcformat", "NUANCE",
        "--output", str(out_kin),
        "--randomseed", str(int(seed)),
    ]
    if channel and channel != "all":
        cmd += ["--channel", channel]
    if starttime is not None:
        cmd += ["--starttime", f"{starttime:g}"]
    if endtime is not None:
        cmd += ["--endtime", f"{endtime:g}"]
    return cmd


def _parse_nuance(path: Path) -> list[dict]:
    """Parse an sntools NUANCE ``.kin`` file into a list of events.

    Each event is ``{"incoming": [...], "outgoing": [...]}`` where every
    particle is ``(pdg, E_MeV, dirx, diry, dirz)``. Tracks are classified by
    their trailing status column: ``-1`` incoming (the neutrino), ``0``
    outgoing (final-state).
    """
    events: list[dict] = []
    cur: Optional[dict] = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("$"):
                continue
            tok = line.split()
            kind = tok[1]
            if kind == "begin":
                cur = {"incoming": [], "outgoing": [], "code": 0, "time_ms": 0.0}
            elif kind == "nuance" and cur is not None:
                # `$ nuance <code>` — the interaction channel code.
                cur["code"] = int(tok[2])
            elif kind == "vertex" and cur is not None:
                # `$ vertex x y z t` — t is the interaction time in ms
                # (sntools event.time, post-bounce). The burst time we preserve.
                cur["time_ms"] = float(tok[5])
            elif kind == "track" and cur is not None:
                pdg = int(tok[2])
                e = float(tok[3])
                dx, dy, dz = float(tok[4]), float(tok[5]), float(tok[6])
                status = int(tok[7])
                part = (pdg, e, dx, dy, dz)
                if status == -1:
                    cur["incoming"].append(part)
                elif status == 0:
                    cur["outgoing"].append(part)
            elif kind == "end" and cur is not None:
                events.append(cur)
                cur = None
    return events


def _p4_gev(pdg: int, e_mev: float, dx: float, dy: float, dz: float) -> tuple:
    """Return ``(px, py, pz, E)`` in GeV from an NUANCE total-energy track.

    NUANCE stores total energy; rooTracker P4 wants 3-momentum + energy, so
    recover ``|p| = sqrt(E^2 - m^2)`` (massless → |p| = E). Directions are
    already unit vectors.
    """
    mass = PARTICLE_MASS_MeV.get(pdg, 0.0)
    p2 = e_mev * e_mev - mass * mass
    p = p2 ** 0.5 if p2 > 0.0 else 0.0
    gev = 1.0e-3
    return (dx * p * gev, dy * p * gev, dz * p * gev, e_mev * gev)


def _rotation_from_z(direction) -> Optional["object"]:
    """3x3 rotation matrix mapping +z (sntools' fixed SN axis) onto the unit
    vector ``direction``. Returns None when ``direction`` is +z (no rotation).

    Used to point the whole supernova at a chosen sky direction while preserving
    every interaction's internal kinematics — a single global rotation, not a
    per-interaction one. Same for all jobs of a dataset, so the true SN direction
    is well-defined for pointing studies.
    """
    import numpy as np
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n == 0:
        raise SupernovaError("supernova.direction must be a non-zero vector")
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(d, z):
        return None
    if np.allclose(d, -z):
        return np.diag([1.0, -1.0, -1.0])            # 180 deg about x
    v = np.cross(z, d)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, d))
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _write_rootracker(events: list[dict], out_path: Path,
                      cap: Optional[int] = None, rotation=None) -> int:
    """Write events to a gRooTracker TTree; return the entry count.

    Reproduces GENIE's rooTracker layout — a ``TTree`` named ``gRooTracker``
    with fixed-size leaves ``StdHepPdg[NP]``, ``StdHepStatus[NP]``,
    ``StdHepP4[NP][4]`` and a per-entry valid count ``StdHepN`` — so
    PhotonSim's C++ ``SetBranchAddress`` (fixed ``[kMaxParticles][4]`` buffer,
    loop bounded by ``StdHepN``) reads it unchanged. ``NP`` is the actual max
    particle count across events (≤ kMaxParticles), keeping files small.

    We use uproot (``mktree``/``extend``) rather than PyROOT because the
    container's PyROOT is built against a different Python and won't import;
    uproot writes a genuine TTree (its ``file[name] = dict`` shorthand would
    emit an RNTuple, which PhotonSim cannot read — so mktree is required).

    Each event → one entry: incoming neutrino as ``StdHepStatus==0`` (recorded
    by PhotonSim as the true neutrino), outgoing particles as ``StdHepStatus==1``
    (G4 primaries). ``cap`` truncates to the first N events (validation runs).
    """
    import numpy as np
    import uproot

    if cap is not None and cap >= 0:
        events = events[:cap]

    # Build per-event rows first so we can size the fixed leaf to the real max.
    rows_per_event: list[list[tuple]] = []
    max_np = 1
    for evt in events:
        rows: list[tuple] = []
        def _dir(dx, dy, dz):
            if rotation is None:
                return dx, dy, dz
            r = rotation @ np.array([dx, dy, dz], dtype=np.float64)
            return float(r[0]), float(r[1]), float(r[2])
        # Incoming neutrino(s): status 0 → PhotonSim records true nu pdg/energy.
        for (ppdg, e, dx, dy, dz) in evt["incoming"]:
            if ppdg in NEUTRINO_PDGS:
                rows.append((ppdg, 0, _p4_gev(ppdg, e, *_dir(dx, dy, dz))))
        # Outgoing final-state particles: status 1 → injected as G4 primaries.
        for (ppdg, e, dx, dy, dz) in evt["outgoing"]:
            rows.append((ppdg, 1, _p4_gev(ppdg, e, *_dir(dx, dy, dz))))
        if len(rows) > KMAX_PARTICLES:
            rows = rows[:KMAX_PARTICLES]
        rows_per_event.append(rows)
        max_np = max(max_np, len(rows))

    nev = len(rows_per_event)
    if nev == 0:
        raise SupernovaError("no events to write to rooTracker (empty after cap)")
    np_fixed = min(max_np, KMAX_PARTICLES)

    n_arr = np.zeros(nev, dtype=np.int32)
    pdg = np.zeros((nev, np_fixed), dtype=np.int32)
    status = np.zeros((nev, np_fixed), dtype=np.int32)
    p4 = np.zeros((nev, np_fixed, 4), dtype=np.float64)
    for i, rows in enumerate(rows_per_event):
        n_arr[i] = len(rows)
        for k, (ppdg, st, (px, py, pz, energy)) in enumerate(rows):
            pdg[i, k] = int(ppdg)
            status[i, k] = int(st)
            p4[i, k, 0] = px
            p4[i, k, 1] = py
            p4[i, k, 2] = pz
            p4[i, k, 3] = energy

    with uproot.recreate(str(out_path)) as f:
        f.mktree("gRooTracker", {
            "StdHepN": np.int32,
            "StdHepPdg": ("int32", (np_fixed,)),
            "StdHepStatus": ("int32", (np_fixed,)),
            "StdHepP4": ("float64", (np_fixed, 4)),
        })
        f["gRooTracker"].extend({
            "StdHepN": n_arr, "StdHepPdg": pdg,
            "StdHepStatus": status, "StdHepP4": p4,
        })
    return nev


def run_supernova(
    *,
    config: dict,
    output_dir: Path,
    job_id: int,
    model_name: Optional[str] = None,
    ordering: Optional[str] = None,
    seed: Optional[int] = None,
    cap_events: Optional[int] = None,
) -> tuple[Path, int]:
    """Run sntools + convert to a rooTracker file.

    Returns ``(gtrac_path, n_entries)`` where ``n_entries`` is the number of
    G4 events PhotonSim will emit (one per sntools event). The caller passes
    ``n_entries`` to generate_macro so ``/run/beamOn`` matches, exactly as
    run_genie does.

    ``model_name`` / ``ordering`` select the subcase (the fan-out passes one
    ``(model, ordering)`` per job); ``cap_events`` truncates the burst for
    validation runs.
    """
    if "supernova" not in config:
        raise SupernovaError(
            "config missing 'supernova' block (primary_source=supernova requires it)")
    sn = config["supernova"]

    model = _resolve_model(sn, model_name)
    fmt = model.get("format")
    flux_file = model.get("flux_file")
    if not fmt or not flux_file:
        raise SupernovaError(
            f"model {model.get('name')!r} needs both 'format' and 'flux_file'")
    # SNEWPY-* formats: flux_file is a model file under the SNEWPY cache
    # (<model_path>/<Model>/<file>). sntools abspath()s the path relative to CWD,
    # so we must hand it the absolute cache path (the model must be pre-downloaded;
    # see the module docstring). Other formats (gamma, nakazato ASCII) are
    # repo-relative files.
    if str(fmt).startswith("SNEWPY-"):
        from snewpy import model_path as _snewpy_cache
        _model = str(fmt)[len("SNEWPY-"):]
        cand = Path(_snewpy_cache) / _model / flux_file
        if not cand.is_file():
            raise SupernovaError(
                f"SNEWPY model file not found in cache: {cand}. Pre-download it "
                f"once (see run_supernova module docstring) — compute nodes may "
                f"lack network.")
        flux_file = str(cand)
    else:
        flux_file = str(_resolve_flux_file(flux_file))

    ordering = ordering or sn.get("ordering") or "NMO"
    transformation = _transformation_for(ordering)

    # sntools --detector sets the fiducial volume and therefore the realistic
    # event *count*; it should match the scale of the LUCiD geometry being
    # simulated (e.g. "SuperK" for SK_like). Distance/time-window further
    # scale the burst.
    detector = sn.get("detector", "SuperK")
    distance = float(sn.get("distance_kpc", 10.0))
    starttime = sn.get("starttime_ms")
    endtime = sn.get("endtime_ms")

    # Channels: "all" (default) runs every water channel in one sntools call;
    # a list runs one call per channel and merges the events.
    channels = sn.get("channels", "all")
    if channels in ("all", None) or set(channels) == set(WATER_CHANNELS):
        channel_runs: list[Optional[str]] = [None]
    elif isinstance(channels, str):
        channel_runs = [channels]
    else:
        channel_runs = list(channels)

    if seed is None:
        seed = 100000 + job_id

    job_padded = f"{job_id:06d}"
    gtrac_file = output_dir / f"gntp_job_{job_padded}.gtrac.root"

    all_events: list[dict] = []
    for i, channel in enumerate(channel_runs):
        kin_path = output_dir / f"sn_job_{job_padded}_{(channel or 'all')}.kin"
        cmd = _build_sntools_cmd(
            flux_file=flux_file, fmt=fmt, detector=detector,
            transformation=transformation, distance=distance,
            channel=channel, starttime=starttime, endtime=endtime,
            # Distinct seed per channel run so they don't correlate.
            seed=int(seed) + i, out_kin=kin_path,
        )
        print("=== sntools: generating supernova events ===", flush=True)
        print(f"    model={model.get('name')} ordering={ordering} "
              f"transformation={transformation} channel={channel or 'all'}", flush=True)
        print("    " + " ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(output_dir))
        if rc != 0:
            raise SupernovaError(f"sntools failed (exit {rc}) for channel {channel or 'all'}")
        if not kin_path.is_file():
            raise SupernovaError(f"sntools did not produce {kin_path}")
        evts = _parse_nuance(kin_path)
        print(f"    sntools ok: {len(evts)} events ({time.time() - t0:.1f}s)", flush=True)
        all_events.extend(evts)
        # NUANCE intermediate is not needed downstream; drop it unless asked.
        if not config.get("keep_sntools_kin", False):
            try:
                kin_path.unlink()
            except OSError:
                pass

    if not all_events:
        raise SupernovaError(
            "sntools produced 0 events — check distance/time window/flux "
            "(a too-distant SN or too-narrow window yields no interactions)")

    # Order the whole burst by interaction time before capping, so a cap keeps
    # the burst onset (earliest interactions) in time order and the all-at-once
    # generator sees a monotone timeline. sntools sorts within a single run, but
    # per-channel runs are concatenated unsorted.
    all_events.sort(key=lambda e: e.get("time_ms", 0.0))
    capped = all_events if cap_events is None else all_events[:cap_events]
    # SN direction: sntools fixes the incoming neutrino to +z, so a single global
    # rotation (same for all jobs) points the whole burst at supernova.direction
    # while preserving every interaction's internal kinematics — no per-vertex
    # rotation. Default +z (identity). Recorded in a sidecar so the true direction
    # is available downstream for pointing studies.
    sn_direction = [float(x) for x in (sn.get("direction") or [0.0, 0.0, 1.0])]
    rotation = _rotation_from_z(sn_direction)
    n_entries = _write_rootracker(capped, gtrac_file, cap=None, rotation=rotation)

    # Sidecars, each aligned 1:1 with rooTracker entries (and therefore LUCiD's
    # event order). The StdHep rooTracker carries neither the channel nor the
    # burst time, so we write them alongside:
    #   .channels.json — sntools channel code (the generator stamps it into
    #                    per_interaction/interaction_channel)
    #   .times.json    — interaction time in ms (the all-at-once generator turns
    #                    each into a per-interaction t0 = global_t0 + time_ms*1e6)
    import json
    sidecar = output_dir / f"gntp_job_{job_padded}.channels.json"
    sidecar.write_text(json.dumps([int(e["code"]) for e in capped]))
    times_sidecar = output_dir / f"gntp_job_{job_padded}.times.json"
    times_sidecar.write_text(json.dumps([float(e.get("time_ms", 0.0)) for e in capped]))
    # True SN direction (unit vector) — same for the whole burst; the generator
    # records it in the labl truth for direction-pointing studies.
    _dnorm = sum(x * x for x in sn_direction) ** 0.5 or 1.0
    dir_sidecar = output_dir / f"gntp_job_{job_padded}.direction.json"
    dir_sidecar.write_text(json.dumps([x / _dnorm for x in sn_direction]))

    if cap_events is not None:
        print(f"    capped to {n_entries} of {len(all_events)} events (validation)", flush=True)
    from collections import Counter
    chan_hist = Counter(CHANNEL_CODE_TO_NAME.get(int(e["code"]), "unknown") for e in capped)
    print(f"    channels: {dict(chan_hist)}", flush=True)
    print(f"    wrote {n_entries} rooTracker entries → {gtrac_file}", flush=True)
    return gtrac_file, n_entries
