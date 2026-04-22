# Docker Quickstart — Produce v3 events on macOS or generic Linux

This runbook targets **macOS (Apple Silicon or Intel)** and any Linux
machine without Apptainer/Singularity — e.g. a laptop or a VM with only
Docker installed. The image bundles GEANT4 11.3, ROOT 6.28, Pythia6,
GENIE 3.06, PhotonSim, and LUCiD; **GENIE cross-section splines are
baked in** at build time, so runs work offline.

For S3DF/SLURM see [QUICKSTART_S3DF.md](QUICKSTART_S3DF.md); for a
host-native conda build (no container) see
[QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md).

## Prerequisites

- **Docker Desktop** (macOS/Windows) or `docker` engine (Linux).
- **~20 GB free disk** (image is ~6 GB; build cache adds a few more).
- **~90 min** for the first build on Apple Silicon (Rosetta overhead) or
  ~60 min on native x86_64.

### Apple Silicon (M1/M2/M3/M4): enable Rosetta 2

The image is `linux/amd64` — conda-forge does not ship `geant4` for
`linux-aarch64`, so native `arm64` is not an option today. Rosetta 2
runs amd64 images at roughly 70–80% of native speed, which is fine for
test runs and development.

1. Install Rosetta 2: `softwareupdate --install-rosetta --agree-to-license`.
2. In Docker Desktop → Settings → General, enable
   **"Use Rosetta for x86/amd64 emulation on Apple Silicon"**.

## 1. Clone LUCiD and PhotonSim as siblings

```bash
mkdir lucid-work && cd lucid-work
git clone https://github.com/CIDeR-ML/LUCiD.git
git clone https://github.com/cesarjesusvalls/PhotonSim.git
```

Layout must be:

```
lucid-work/
├── LUCiD/
└── PhotonSim/
```

## 2. Build the image (one time, ~60–90 min)

From the `lucid-work` directory (the parent of both repos):

```bash
docker build \
    --platform linux/amd64 \
    -f LUCiD/container/Dockerfile \
    -t lucid:latest \
    .
```

Breakdown of the build:

| Layer            | Wall time (Rosetta) | What it does                                                   |
|------------------|---------------------|----------------------------------------------------------------|
| apt + conda      | ~10 min             | Ubuntu base, GEANT4 11.3, Python stack, compiler toolchain.    |
| Pythia6          | ~1 min              | `libPythia6.so` from ROOT's prebuilt tarball.                  |
| ROOT 6.28.12     | ~45 min             | Source build with `-Dpythia6=ON` (the slow part).              |
| GENIE 3.06.02    | ~5 min              | Links against `libEGPythia6.so`.                               |
| Xsec splines     | ~2 min              | 455 MB download from FNAL scisoft (public mirror).             |
| PhotonSim, LUCiD | ~3 min              | Geant4 app + editable pip install.                             |

Subsequent builds re-use Docker's layer cache — editing only LUCiD
source code retriggers just the last layer (~30 s).

## 3. Run a test job

```bash
mkdir -p /tmp/lucid-out
docker run --rm --platform linux/amd64 \
    -v "$PWD:/work" \
    -v /tmp/lucid-out:/out \
    lucid:latest \
    lucid-run-job \
        --config /opt/LUCiD/lucid/production/configs/dataprod_13_numu.json \
        --output-dir /out \
        --job-id 1 \
        --test
```

Expect ~2 minutes. Output under `/tmp/lucid-out/`:

```
sensor/wc_sensor_0000.h5
inst/wc_inst_0000.h5
seg/wc_seg_0000.h5
labl/wc_labl_0000.h5
```

The `dataprod_13_numu.json` config runs **νμ + H₂O** at 0.1–2 GeV via
the baked-in `G18_02a_00_000` tune. Use `dataprod_14_nue.json` for the
νₑ chain.

## 4. Inspect the output

The HDF5 files are v3-format (see
[LUCID_DATASET.md](LUCID_DATASET.md)). To browse on the host:

```bash
python3 -c "import h5py; f=h5py.File('/tmp/lucid-out/labl/wc_labl_0000.h5'); \
    print('events:', len(f['event_ids'])); \
    print('dataset_name:', f.attrs.get('dataset_name', b'').decode())"
```

Or run the LUCiD viewer from a clone (no container needed for this
step):

```bash
cd LUCiD
pip install -e .          # or `uv pip install -e .`
python3 viewer/serve_viewer.py /tmp/lucid-out --open
```

## Using a different xsec spline

The image bakes `G18_02a_00_000` at `/opt/genie_xsec/gxspl-FNALsmall.xml`.
To override at runtime:

```bash
# Put your own gxspl XML on the host, bind-mount over /opt/genie_xsec:
docker run --rm --platform linux/amd64 \
    -v /path/to/my/xsec:/opt/genie_xsec \
    -e GENIE_XSEC_FILE=/opt/genie_xsec/your-file.xml \
    ...
```

Or fetch a different tune into the image before building: edit the
`fetch-xsec-splines.sh` call in `Dockerfile` and pass `--tune
G18_10a_02_11a` (etc.).

## Troubleshooting

- **`exec /bin/bash: exec format error`** — you're on Apple Silicon
  without Rosetta enabled. See step 0 above.
- **Build hangs during ROOT step** — not hung; just slow under Rosetta.
  Expect 40–60 min for that one layer. `docker stats` should show a
  busy container.
- **`GENIE_XSEC_FILE unset or missing`** — the xsec fetch failed at
  build time (network blip?). Rebuild just the affected layer:
  `docker build --target … --no-cache …` or delete the image and
  rebuild.
- **Apple Silicon very slow (<50% native)** — make sure Rosetta is
  enabled in Docker Desktop settings, not just installed on the Mac.
- **"No space left on device"** — `docker system prune -af` reclaims
  old layers and dangling images.

## Publishing the image (follow-up, not included here)

Current setup is **local build only**. If you want to skip the 60 min
build, the image can be pushed to a registry (ghcr.io, Docker Hub) and
pulled instead — but that is a separate decision not covered here.
