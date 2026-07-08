# Docker Quickstart — Produce events on macOS or generic Linux

This runbook targets **macOS (Apple Silicon or Intel)** and any Linux
machine without Apptainer/Singularity. The published image bundles
GEANT4 11.3, ROOT 6.30 with Pythia6, GENIE 3.04, PhotonSim, and LUCiD;
cross-section splines for the `AR23_20i_00_000`, `G18_10a_02_11b`, and
`G21_11a_00_000` tunes ship with the underlying NUISANCE base, so runs
work offline.

For S3DF/SLURM see [QUICKSTART_S3DF.md](QUICKSTART_S3DF.md); for a
host-native install (no container) see
[QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md).

## Prerequisites

- **Docker Desktop** (macOS/Windows) or `docker` engine (Linux).
- **~10 GB free disk** (image is ~4 GB).

### Apple Silicon (M1/M2/M3/M4): enable Rosetta 2

The image is `linux/amd64` — conda-forge does not ship `geant4` for
`linux-aarch64`. Rosetta 2 runs amd64 images at roughly 70–80% of
native speed, which is fine for test runs and development.

1. Install Rosetta 2: `softwareupdate --install-rosetta --agree-to-license`.
2. In Docker Desktop → Settings → General, enable
   **"Use Rosetta for x86/amd64 emulation on Apple Silicon"**.

## 1. Pull the image

```bash
docker pull --platform linux/amd64 ghcr.io/cider-ml/lucid:latest
docker tag ghcr.io/cider-ml/lucid:latest lucid:latest
```

No build required. First pull is ~4 GB.

## 2. Run a test job

```bash
mkdir -p /tmp/lucid-out
docker run --rm --platform linux/amd64 \
    -v /tmp/lucid-out:/out \
    lucid:latest \
    lucid-run-job \
        --config /opt/LUCiD/lucid/production/configs/GeV/01_mu.json \
        --output-dir /out --job-id 1 --test
```

Expect ~30 s. Output under `/tmp/lucid-out/`:

```
sensor/wc_sensor_0000.h5
hits/wc_hits_0000.h5
step/wc_step_0000.h5
labl/wc_labl_0000.h5
```

For the GENIE chain (gevgen → gntpc → PhotonSim → LUCiD), the
in-repo configs pin tune `G18_10a_02_11b`, whose splines ship in
the NUISANCE base image — so they run inside the container without
extra setup:

```bash
docker run --rm --platform linux/amd64 \
    -v /tmp/lucid-out:/out \
    -e GENIE_XSEC_FILE=/opt/genie_xsec/3_04_00/G18_10a_02_11b/gxspl-min.xml.gz \
    lucid:latest \
    lucid-run-job \
        --config /opt/LUCiD/lucid/production/configs/GeV/13_genie_numu_nue.json \
        --output-dir /out --job-id 1 --test
```

For other tunes, set `GENIE_XSEC_FILE` to your own spline path.

## 3. Dev loop — bind-mount your checkout

To iterate on LUCiD Python code without rebuilding the image, bind the
local clone over `/opt/LUCiD`:

```bash
docker run --rm --platform linux/amd64 \
    -v "$PWD/LUCiD:/opt/LUCiD" \
    -v /tmp/lucid-out:/out \
    lucid:latest \
    lucid-run-job --config /opt/LUCiD/lucid/production/configs/GeV/01_mu.json \
                  --output-dir /out --job-id 1 --test
```

For a PhotonSim source edit, bind-mount and rebuild in place:

```bash
docker run --rm --platform linux/amd64 \
    -v "$PWD/PhotonSim:/opt/PhotonSim" \
    -v "$PWD/LUCiD:/opt/LUCiD" \
    -v /tmp/lucid-out:/out \
    lucid:latest \
    bash -c "cmake --build /opt/PhotonSim/build -j && \
             lucid-run-job --config /opt/LUCiD/lucid/production/configs/GeV/01_mu.json \
                           --output-dir /out --job-id 1 --test"
```

The baked `/opt/PhotonSim/build` stays intact, so incremental compiles
land in <1 min.

## 4. Inspect the output

The HDF5 files are LUCiD-format (see
[LUCID_DATASET.md](LUCID_DATASET.md)). To browse on the host:

```bash
python3 -c "import h5py; f=h5py.File('/tmp/lucid-out/labl/wc_labl_0000.h5'); \
    print('events:', len(f['config/source_event_idx'])); \
    print('dataset_name:', f.attrs.get('dataset_name', b'').decode())"
```

Or run the LUCiD viewer from the clone (no container needed):

```bash
cd LUCiD
pip install -e .
python3 viewer/serve_viewer.py /tmp/lucid-out --open
```

## Opt-in: build the image from source

You only need this if you're modifying the Dockerfile itself. Clone
LUCiD and PhotonSim as siblings and build from the parent:

```bash
mkdir lucid-work && cd lucid-work
git clone https://github.com/CIDeR-ML/LUCiD.git
git clone https://github.com/cesarjesusvalls/PhotonSim.git
docker build --platform linux/amd64 --provenance=false --sbom=false \
    -f LUCiD/container/Dockerfile -t lucid:latest .
```

`--provenance=false --sbom=false` suppresses BuildKit attestation
manifests; ghcr.io stalls on them for this package. Expect ~10 min cold
on Apple Silicon. Subsequent rebuilds reuse layers — editing LUCiD
source retriggers just the last layer (~30 s).

## Troubleshooting

- **`exec /bin/bash: exec format error`** — Apple Silicon without
  Rosetta enabled in Docker Desktop. See step 0.
- **BuildKit silence during source builds** — long solve/compile steps
  buffer output. Use `docker stats` to see CPU use, or
  `--progress=plain`.
- **GENIE complains about missing splines** — the config's `tune` does
  not match a tune in the image. Swap to `G18_10a_02_11b` (or
  `AR23_20i_00_000`, `G21_11a_00_000`) and set `GENIE_XSEC_FILE`
  to the matching `/opt/genie_xsec/3_04_00/<tune>/gxspl-min.xml.gz`.
- **"No space left on device"** — `docker system prune -af` reclaims
  old layers.
