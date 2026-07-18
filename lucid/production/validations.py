"""Dataset validation suite for produced LUCiD datasets.

Runs five checks against one dataset directory (a ``config_NN`` tree with
``sensor/ hits/ step/ labl/`` shards):

  integrity     all shards readable, modalities self-consistent, one git commit
  charge        reco digit charge vs true p.e. against the digitizer SPE model
  time          digit-time residual vs true first-arrival against the TTS model
  independence  per-PID primary-energy uniformity, duplicate rates vs the
                float32 birthday null, event-fingerprint duplicates
  symmetry      per-sensor mean charge: azimuthal residual, Fourier modes,
                dead sensors

The electronics model used for the charge/time overlays is imported from
``lucid.simulation.digitizer`` (selected by the dataset's ``digitizer_model``
attr), so the reference always tracks the code.

Usage::

    python -m lucid.production.validations --dataset <.../test/config_06> \
        [--out DIR] [--response-shards 12] [--skip charge,time,...]

Writes ``val_{charge,time,symmetry,energy}.png`` plus a PASS/FAIL summary to
stdout; exits non-zero if any enabled check fails.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

MODALITIES = ("sensor", "hits", "step", "labl")


# --------------------------------------------------------------------- loading
def _shards(dataset: Path):
    files = sorted(glob.glob(str(dataset / "sensor" / "wc_sensor_*.h5")),
                   key=lambda p: int(re.search(r"_(\d+)\.h5", p).group(1)))
    return [int(re.search(r"_(\d+)\.h5", p).group(1)) for p in files]


def _path(dataset: Path, mod: str, fi: int) -> str:
    return str(dataset / mod / f"wc_{mod}_{fi:04d}.h5")


def check_integrity(dataset: Path):
    """All shards readable, n_events attr == event groups, modalities agree,
    and the file_index sequence is gap-free (a missing shard means a job died
    before its first write and left no file to flag)."""
    import h5py
    present = _shards(dataset)
    gaps = sorted(set(range(max(present) + 1)) - set(present)) if present else []
    corrupt, nev, commits = [], [], set()
    for fi in present:
        per, ok = {}, True
        for m in MODALITIES:
            try:
                with h5py.File(_path(dataset, m, fi), "r") as f:
                    ne = int(f["config"].attrs.get("n_events", -1))
                    groups = sum(1 for k in f.keys() if k.startswith("event_"))
                    per[m] = (ne, groups)
                    commits.add(str(f["config"].attrs.get("git_commit", "?"))[:12])
            except Exception:
                ok = False
        if not ok:
            corrupt.append((fi, "unreadable"))
        elif len({v[0] for v in per.values()}) != 1 or any(a != b for a, b in per.values()):
            corrupt.append((fi, "inconsistent"))
        else:
            nev.append(per["sensor"][0])
    passed = not corrupt and not gaps and len(commits) == 1
    detail = (f"{len(nev)} shards, {sum(nev)} events, commits={sorted(commits)}"
              + (f", BAD={corrupt}" if corrupt else "")
              + (f", MISSING file_indices={gaps}" if gaps else ""))
    return passed, detail, {"n_events": nev, "corrupt": corrupt, "missing": gaps}


def load_response_samples(dataset: Path, n_shards: int):
    """Per-digit (true_pe, reco_pe, time_residual), Cherenkov-only digits.

    Unreadable shards are skipped: integrity already reports them, and the
    response checks should still characterise the good data rather than die
    on the first bad file.
    """
    import h5py
    tp, reco, resid = [], [], []
    taken = 0
    for fi in _shards(dataset):
        if taken >= n_shards:
            break
        try:
            s = h5py.File(_path(dataset, "sensor", fi), "r")
            h = h5py.File(_path(dataset, "hits", fi), "r")
        except Exception:
            continue
        taken += 1
        with s, h:
            for ev in s.keys():
                if not ev.startswith("event_"):
                    continue
                sPE = s[ev]["PE"][:]
                sT = s[ev]["T"][:]
                hPE = h[ev]["PE"][:]
                hT = h[ev]["T"][:]
                hDi = h[ev]["digit_idx"][:]
                hEP = h[ev]["emission_process"][:]
                nd = len(sPE)
                true = np.zeros(nd)
                dark = np.zeros(nd)
                tt = np.full(nd, np.inf)
                v = hDi >= 0
                np.add.at(true, hDi[v], hPE[v])
                np.add.at(dark, hDi[v], (hEP[v] == 2).astype(float))
                np.minimum.at(tt, hDi[v], hT[v])
                clean = (dark == 0) & (sPE > 0) & np.isfinite(tt)
                tp.append(np.rint(true[clean]).astype(np.int16))
                reco.append(sPE[clean].astype(np.float32))
                resid.append((sT[clean] - tt[clean]).astype(np.float32))
    return np.concatenate(tp), np.concatenate(reco), np.concatenate(resid)


def load_sensor_sums(dataset: Path):
    """Per-sensor summed charge over all shards + event count + positions."""
    import h5py
    sum_pe, pos, nev = None, None, 0
    for fi in _shards(dataset):
        try:
            sf = h5py.File(_path(dataset, "sensor", fi), "r")
        except Exception:
            continue                      # unreadable shard: integrity reports it
        with sf as s:
            if pos is None:
                pos = s["config"]["sensor_positions"][:]
                sum_pe = np.zeros(pos.shape[0], np.float64)
            for ev in s.keys():
                if not ev.startswith("event_"):
                    continue
                np.add.at(sum_pe, s[ev]["sensor_idx"][:].astype(np.int64),
                          s[ev]["PE"][:].astype(np.float64))
                nev += 1
    return sum_pe, pos, nev


def load_primaries(dataset: Path):
    """(pdg, energy) of every primary, plus per-event fingerprints."""
    import h5py
    pdgs, ens, fingerprints = [], [], []
    for fi in _shards(dataset):
        try:
            lf = h5py.File(_path(dataset, "labl", fi), "r")
        except Exception:
            continue                      # unreadable shard: integrity reports it
        with lf as f:
            for ev in f.keys():
                if not ev.startswith("event_"):
                    continue
                pi = f[ev]["per_interaction"]
                p = np.asarray(pi["primary_pdgs_data"][:])
                e = np.asarray(pi["primary_energies_data"][:], np.float32)
                pdgs.append(p)
                ens.append(e)
                fingerprints.append((tuple(int(x) for x in p),
                                     tuple(float(x) for x in e)))
    return np.concatenate(pdgs), np.concatenate(ens), fingerprints


def _digitizer_model(dataset: Path):
    import h5py
    from lucid.simulation.digitizer import resolve_model_config
    fi = _shards(dataset)[0]
    with h5py.File(_path(dataset, "sensor", fi), "r") as s:
        name = str(s["config"].attrs.get("digitizer_model", "basic"))
    return name, resolve_model_config(name)


# ------------------------------------------------------------------ charge / Q
def check_charge(dataset: Path, out: Path, n_shards: int, tol=0.02):
    """Reco charge vs true p.e.: unbiased mean + shape match to the SPE model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lucid.simulation.digitizer import _sample_spe_charge

    name, model = _digitizer_model(dataset)
    spe = model.get("spe")
    tp, reco, _ = load_response_samples(dataset, n_shards)
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
    fig.suptitle(f"charge response vs SPE model ({name})", fontweight="bold")
    fails = []
    means = {}
    for i, N in enumerate([1, 2, 3, 4, 5]):
        m = tp == N
        a = ax.flat[i]
        if m.sum() < 1000:
            a.set_visible(False)
            continue
        mu = float(reco[m].mean())
        means[N] = mu
        if abs(mu - N) / N > tol:
            fails.append(f"mean(reco|{N}pe)={mu:.3f}")
        bins = np.linspace(max(0, N - 2.5), N + 3.5, 70)
        a.hist(reco[m], bins=bins, density=True, alpha=0.3, color="#2b6cb0")
        a.hist(reco[m], bins=bins, density=True, histtype="step", color="#2b6cb0",
               label=f"data (n={m.sum()})")
        if spe:
            mod = _sample_spe_charge(np.full(300000, float(N)), spe, rng)
            a.hist(mod, bins=bins, density=True, histtype="step", ls="--",
                   color="#dd6b20", label="SPE model")
        a.axvline(N, color="gray", ls=":")
        a.set_title(f"true={N} p.e. (mean reco {mu:.2f})")
        a.legend(fontsize=8)
    ax.flat[5].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "val_charge.png", dpi=130)
    plt.close(fig)
    detail = " ".join(f"{k}pe:{v:.3f}" for k, v in means.items())
    return not fails, detail + (f"  FAIL:{fails}" if fails else ""), {}


# -------------------------------------------------------------------- time / T
def check_time(dataset: Path, out: Path, n_shards: int, tol_ns=0.1):
    """Digit-time residual vs the digitizer TTS model (moments + shape).

    The model residual is built as quantize(true_phase + jitter) − true_phase
    so the TDC comb washes out exactly as it does in data (the residual is
    quantized-reco minus continuous truth).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lucid.simulation.digitizer import _sample_time_jitter

    name, model = _digitizer_model(dataset)
    tp, _, resid = load_response_samples(dataset, n_shards)
    rng = np.random.default_rng(11)
    tdc = float(model.get("tdc_ns", 0.0)) or 1e-9

    def model_resid(N, n=300000):
        u = rng.uniform(0.0, tdc, n)
        return _sample_time_jitter(u, np.full(n, float(N)), model, rng) - u

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
    fig.suptitle(f"time residual vs TTS model ({name})", fontweight="bold")
    fails, rows = [], []
    for i, N in enumerate([1, 2, 3, 4, 5]):
        m = tp == N
        a = ax.flat[i]
        if m.sum() < 1000:
            a.set_visible(False)
            continue
        mod = model_resid(N)
        dm, dr = float(resid[m].mean()), float(resid[m].std())
        mm, mr = float(mod.mean()), float(mod.std())
        rows.append(f"{N}pe:d({dm:.2f},{dr:.2f})m({mm:.2f},{mr:.2f})")
        if abs(dm - mm) > tol_ns or abs(dr - mr) > tol_ns:
            fails.append(f"{N}pe moments off")
        bins = np.linspace(-4, 9, 80)
        a.hist(resid[m], bins=bins, density=True, alpha=0.3, color="#2b6cb0")
        a.hist(resid[m], bins=bins, density=True, histtype="step", color="#2b6cb0",
               label=f"data (n={m.sum()})")
        a.hist(mod, bins=bins, density=True, histtype="step", ls="--",
               color="#dd6b20", label="model")
        a.axvline(0, color="gray", ls=":")
        a.set_title(f"true={N} p.e. (rms {dr:.2f} ns)")
        a.legend(fontsize=8)
    ax.flat[5].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "val_time.png", dpi=130)
    plt.close(fig)
    return not fails, " ".join(rows) + (f"  FAIL:{fails}" if fails else ""), {}


# --------------------------------------------------------- label independence
def check_independence(dataset: Path, out: Path, null_trials=100, nsigma=5.0):
    """Primary energies per PID: uniform, and duplicate rates at the float32
    birthday null. Event-level fingerprint duplicates are reported and failed
    only when far above the per-PID null (a genuine seed collision would also
    reproduce directions — worth checking by hand on any flagged pair)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdgs, ens, fingerprints = load_primaries(dataset)
    rng = np.random.default_rng(3)
    fails, rows = [], []
    uniq = {}
    null_mu_tot = null_var_tot = 0.0
    for pid in np.unique(pdgs):
        a = ens[pdgs == pid].astype(np.float32)
        n = len(a)
        if n < 1000:
            continue
        exact = n - len(np.unique(a))
        lo, hi = float(a.min()), float(a.max())
        null = [len(x := (lo + (hi - lo) * rng.random(n)).astype(np.float32))
                - len(np.unique(x)) for _ in range(null_trials)]
        mu, sd = float(np.mean(null)), float(np.std(null)) or 1.0
        null_mu_tot += mu
        null_var_tot += sd * sd
        rows.append(f"pid{pid}:dup{exact}(null {mu:.0f}±{sd:.0f})")
        if exact > mu + nsigma * sd:
            fails.append(f"pid {pid}: {exact} exact dups vs null {mu:.1f}±{sd:.1f}")
        uniq[int(pid)] = a
    # Event fingerprints can only collide when every (pdg, energy) matches, so
    # the summed per-PID float32-birthday null upper-bounds the expectation
    # (exact for single-particle datasets). A genuine seed collision sits far
    # above it — and would also reproduce directions, worth checking by hand.
    n_dupe = len(fingerprints) - len(set(fingerprints))
    thr = null_mu_tot + nsigma * (null_var_tot ** 0.5)
    rows.append(f"event-fingerprint dups={n_dupe}/{len(fingerprints)} "
                f"(null {null_mu_tot:.0f}, thr {thr:.0f})")
    if n_dupe > thr:
        fails.append(f"{n_dupe} duplicate event fingerprints vs null bound {thr:.0f}")

    cols = int(np.ceil(len(uniq) / 2)) or 1
    fig, ax = plt.subplots(2, cols, figsize=(3.2 * cols, 6), squeeze=False)
    for a, (pid, arr) in zip(ax.flat, sorted(uniq.items())):
        a.hist(arr, bins=80, color="#2b6cb0")
        a.set_title(f"pdg {pid} (n={len(arr)})", fontsize=9)
    for a in ax.flat[len(uniq):]:
        a.axis("off")
    fig.suptitle("primary energy by PID", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "val_energy.png", dpi=130)
    plt.close(fig)
    return not fails, " ".join(rows) + (f"  FAIL:{fails}" if fails else ""), {}


# ------------------------------------------------------------------- symmetry
def check_symmetry(dataset: Path, out: Path, max_mode_pct=0.5):
    """Azimuthal symmetry of per-sensor mean charge (cylinder barrel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sum_pe, pos, nev = load_sensor_sums(dataset)
    mean_q = sum_pe / max(nev, 1)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    phi = np.degrees(np.arctan2(y, x))
    cap = np.abs(z) > 0.97 * np.abs(z).max()
    barrel = ~cap
    zb, qb, phib = z[barrel], mean_q[barrel], phi[barrel]
    zbins = np.linspace(zb.min(), zb.max(), 41)
    zi = np.clip(np.digitize(zb, zbins) - 1, 0, len(zbins) - 2)
    ring = np.array([qb[zi == k].mean() if (zi == k).any() else np.nan
                     for k in range(len(zbins) - 1)])
    ratio = qb / ring[zi]
    pb = np.linspace(-180, 180, 73)
    pc = 0.5 * (pb[1:] + pb[:-1])
    pj = np.clip(np.digitize(phib, pb) - 1, 0, len(pb) - 2)
    prof = np.array([ratio[pj == k].mean() for k in range(len(pb) - 1)])
    ang = np.radians(pc)
    sig = prof - prof.mean()
    amps = {m: 200.0 / len(sig) * abs(np.sum(sig * np.exp(-1j * m * ang)))
            for m in range(1, 9)}
    dead = int((mean_q == 0).sum())

    fails = [f"m={m} amp {a:.2f}%" for m, a in amps.items() if a > max_mode_pct]
    if dead:
        fails.append(f"{dead} dead sensors")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.9])
    fig.suptitle("per-sensor mean charge symmetry", fontweight="bold")
    a = fig.add_subplot(gs[0, :])
    sc = a.scatter(phib, zb, c=qb, s=6, cmap="viridis")
    a.set_title("barrel (unrolled)")
    fig.colorbar(sc, ax=a, label="⟨Q⟩/event")
    a = fig.add_subplot(gs[1, :])
    vlim = min(max(abs(1 - ratio.min()), abs(ratio.max() - 1)), 0.15)
    sc = a.scatter(phib, zb, c=ratio, s=6, cmap="RdBu_r", vmin=1 - vlim, vmax=1 + vlim)
    a.set_title("azimuthal residual")
    fig.colorbar(sc, ax=a, label="ratio")
    a = fig.add_subplot(gs[2, 0])
    a.plot(pc, prof, color="#2b6cb0")
    a.axhline(1, color="gray", ls=":")
    a.set_title("φ-profile")
    a = fig.add_subplot(gs[2, 1])
    a.bar(list(amps), list(amps.values()), color="#2b6cb0")
    a.axhline(max_mode_pct, color="#dd6b20", ls="--")
    a.set_title("azimuthal Fourier amplitudes [%]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "val_symmetry.png", dpi=130)
    plt.close(fig)
    detail = (f"events={nev} " + " ".join(f"m{m}:{a:.2f}%" for m, a in amps.items())
              + f" dead={dead}")
    return not fails, detail + (f"  FAIL:{fails}" if fails else ""), {}


# ----------------------------------------------------------------------- main
CHECKS = {
    "integrity": lambda ds, out, args: check_integrity(ds),
    "charge": lambda ds, out, args: check_charge(ds, out, args.response_shards),
    "time": lambda ds, out, args: check_time(ds, out, args.response_shards),
    "independence": lambda ds, out, args: check_independence(ds, out),
    "symmetry": lambda ds, out, args: check_symmetry(ds, out),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True,
                   help="dataset dir containing sensor/ hits/ step/ labl/")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir for figures (default: <dataset>/validations)")
    p.add_argument("--response-shards", type=int, default=12,
                   help="shards to sample for the charge/time checks")
    p.add_argument("--skip", type=str, default="",
                   help="comma-separated checks to skip")
    args = p.parse_args(argv)

    out = args.out or (args.dataset / "validations")
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    all_ok = True
    for name, fn in CHECKS.items():
        if name in skip:
            print(f"[skip] {name}")
            continue
        try:
            ok, detail, _ = fn(args.dataset, out, args)
        except Exception as e:  # a crashed check is a failed check
            ok, detail = False, f"CRASHED: {e!r}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    print(f"=> {'ALL PASS' if all_ok else 'FAILURES PRESENT'}  (figures in {out})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
