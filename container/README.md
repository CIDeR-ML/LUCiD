# Container build

Two ways to build the unified GENIE → PhotonSim → LUCiD image:

- **Apptainer / Singularity** (Linux, HPC-friendly, supports
  `--fakeroot` sandbox iteration). Uses the two `.def` files in this
  directory. See below.
- **Docker** (macOS, Windows, generic Linux laptops). Uses
  `Dockerfile`. See [../docs/QUICKSTART_DOCKER.md](../docs/QUICKSTART_DOCKER.md).

Both build paths share the same `scripts/*.sh` — edit once, affect
both.

## Layout

- `lucid-base.def` — one-time ~60 min build: apt + conda (GEANT4 11.3, Python stack) + Pythia6 source + ROOT 6.28.12 source (with `-Dpythia6=ON`).
- `lucid.def` — ~15 min top layer bootstrapped from `lucid-base.sif`: GENIE 3.06.02 source + PhotonSim + LUCiD pip install + GENIE xsec splines baked in.
- `Dockerfile` — single-stage Docker build mirroring both `.def` files via the same scripts. Targets `linux/amd64`.
- `scripts/` — individual `build-*.sh` and helpers. Each script is idempotent; the def files and Dockerfile just invoke them in order.
- `scripts/fetch-xsec-splines.sh` — pulls GENIE xsec splines from the public FNAL scisoft mirror. Called by both `lucid.def` %post and the Dockerfile so every image is self-contained (no cvmfs, no first-run download).

## Apptainer build

### One-time base (~60 min)

```bash
cd container
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/apptainer-build}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/apptainer-cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer build --fakeroot "${LUCID_IMAGE_DIR:-$PWD}/lucid-base.sif" lucid-base.def
```

Set `APPTAINER_TMPDIR` / `APPTAINER_CACHEDIR` to somewhere with enough
free space (several GB during build). On S3DF, point them at a path
under `/sdf/data/<group>/...` rather than your `$HOME`.

### Top layer (~15 min, rebuild whenever GENIE/PhotonSim/LUCiD change)

```bash
apptainer build --fakeroot "${LUCID_IMAGE_DIR:-$PWD}/lucid.sif" lucid.def
```

`lucid.def` bootstraps from a `localimage:` path — open the file and
edit `From:` if your base `.sif` lives elsewhere.

## Iteration workflow (faster than rebuilding `.sif` each time)

Convert base `.sif` to a writable **sandbox** once; then run individual
`build-*.sh` scripts inside via `apptainer exec --writable --fakeroot`.
Each run takes seconds-to-minutes depending on the step.

```bash
SANDBOX=${LUCID_SANDBOX_DIR:-/tmp/lucid-sandbox}
IMG_DIR=${LUCID_IMAGE_DIR:-$PWD}
BUILD_CACHE=${LUCID_BUILD_CACHE:-/tmp/lucid-build-cache}
mkdir -p "$BUILD_CACHE"

# One-time: unpack base into a writable sandbox (~5 min).
apptainer build --sandbox "$SANDBOX" "$IMG_DIR/lucid-base.sif"

# Iterate: rerun a single build step
BIND_FLAGS=(
    --writable --fakeroot --no-mount tmp
    -B "$BUILD_CACHE:/build-cache"
    -B "$PWD/../../PhotonSim:/opt/PhotonSim"
    -B "$PWD/..:/opt/LUCiD"
)
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-genie.sh
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-photonsim.sh
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-lucid.sh

# Sanity check before committing to final .sif
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/preflight.sh

# When everything looks good, produce the final .sif
apptainer build --fakeroot "$IMG_DIR/lucid.sif" "$SANDBOX"
```

The `--no-mount tmp` flag stops Apptainer from bind-mounting the host's
`/tmp` into the container, which keeps leftover host state from
poisoning the build.

## Cache

Pythia6 tarball and the ROOT source checkout live in
`$LUCID_BUILD_CACHE` (default `/tmp/lucid-build-cache`, bind-mounted to
`/build-cache` inside the sandbox). The `build-*.sh` scripts reuse them
if present. Safe to delete to force a re-download.

The Docker build doesn't use a bind-mounted cache — each layer caches
inside the image itself, and Docker's own layer cache avoids re-runs
when inputs haven't changed.
