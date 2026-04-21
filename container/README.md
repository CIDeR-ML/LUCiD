# Container build

Two-layer Apptainer image for the GENIE → PhotonSim → LUCiD chain.

## Layout

- `lucid-base.def` — one-time ~60 min build: apt + conda (GEANT4 11.3, Python stack) + Pythia6 source + ROOT 6.28.12 source (with `-Dpythia6=ON`).
- `lucid.def` — ~15 min top layer bootstrapped from `lucid-base.sif`: GENIE 3.06.02 source + PhotonSim + LUCiD pip install.
- `scripts/` — individual `build-*.sh` and helpers. Each script is idempotent; the def files just invoke them in order.

## Building

### One-time base (~60 min)

```bash
cd container
export APPTAINER_TMPDIR=/sdf/data/neutrino/cjesus/tmp/apptainer-build
export APPTAINER_CACHEDIR=/sdf/data/neutrino/cjesus/tmp/apptainer-cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer build --fakeroot /sdf/data/neutrino/cjesus/software/images/lucid-base.sif lucid-base.def
```

### Top layer (~15 min, rebuild whenever GENIE/PhotonSim/LUCiD change)

```bash
apptainer build --fakeroot /sdf/data/neutrino/cjesus/software/images/lucid.sif lucid.def
```

## Iteration workflow (faster than rebuilding `.sif` each time)

Convert base `.sif` to a writable **sandbox** once; then run individual
`build-*.sh` scripts inside via `apptainer exec --writable --fakeroot`.
Each run takes seconds-to-minutes depending on the step.

```bash
SANDBOX=/sdf/data/neutrino/cjesus/software/sandboxes/lucid-work

# One-time: unpack base into a writable sandbox (~5 min).
apptainer build --sandbox "$SANDBOX" /sdf/data/neutrino/cjesus/software/images/lucid-base.sif

# Iterate: rerun a single build step
BIND_FLAGS=(
    --writable --fakeroot --no-mount tmp
    -B /sdf/data/neutrino/cjesus/tmp/build-cache:/build-cache
    -B "$PWD/../../PhotonSim:/opt/PhotonSim"
    -B "$PWD/..:/opt/LUCiD"
)
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-genie.sh
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-photonsim.sh
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/build-lucid.sh

# Sanity check before committing to final .sif
apptainer exec "${BIND_FLAGS[@]}" "$SANDBOX" bash /opt/container-scripts/preflight.sh

# When everything looks good, produce the final .sif
apptainer build --fakeroot /sdf/data/neutrino/cjesus/software/images/lucid.sif "$SANDBOX"
```

The `--no-mount tmp` flag stops Apptainer from bind-mounting the host's
`/tmp` into the container, which keeps leftover host state from
poisoning the build.

## Cache

Pythia6 and ROOT source tarballs live in
`/sdf/data/neutrino/cjesus/tmp/build-cache/`. The `build-*.sh` scripts
reuse them if present. Safe to delete to force a re-download.
