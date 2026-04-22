# Local Quickstart — Produce v3 events on any machine

> **On macOS, or prefer a container?** See
> [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md) — one `docker build`,
> then one `docker run`, with GEANT4 + ROOT + GENIE already set up.

Three steps to a working v3 dataset with no S3DF, no SLURM, no
singularity.

## Prerequisites

- **GEANT4 11.3+** and **ROOT 6.0+** installed on your system. On
  Linux, `conda install -c conda-forge geant4 root` is the fastest
  path.
- **Python 3.9+**.
- **GENIE v3** (only if you want neutrino-flux configs like
  `dataprod_13_numu.json` / `dataprod_14_nue.json`). Easiest: use the
  unified container — see the bottom of this doc.

## 1. Build PhotonSim (once)

```bash
git clone https://github.com/cesarjesusvalls/PhotonSim.git
cd PhotonSim
mkdir build && cd build
cmake -DGeant4_DIR=$(geant4-config --prefix)/lib/cmake/Geant4 ..
make -j$(nproc)
# Result: ./PhotonSim (binary)
```

## 2. Install LUCiD

```bash
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD
pip install -e .
# The console script `lucid-run-job` is now on your PATH.
```

## 3. Run one job

```bash
# Tell lucid-run-job where the PhotonSim binary lives:
export PHOTONSIM_BIN=/absolute/path/to/PhotonSim/build/PhotonSim

# Pick any of the bundled configs:
CONFIG=$(python3 -c "from importlib.resources import files; \
    print(files('lucid.production.configs').joinpath('dataprod_01_mu.json'))")

# Run:
mkdir -p /tmp/my_dataset
lucid-run-job \
    --config "$CONFIG" \
    --output-dir /tmp/my_dataset \
    --job-id 1 \
    --test
```

Output structure under `/tmp/my_dataset/`:

```
sensor/wc_sensor_0000.h5
inst/wc_inst_0000.h5
seg/wc_seg_0000.h5
labl/wc_labl_0000.h5
```

See [LUCID_DATASET.md](LUCID_DATASET.md) for the v3 schema.

## 4. Inspect in a browser (optional)

```bash
python3 LUCiD/viewer/serve_viewer.py /tmp/my_dataset --open
```

Opens the interactive PMT + segment display on
`http://127.0.0.1:8765/`.

## Running a full dataset locally

Drop `--test` and add `--n-events N` to run the config's full event
count. Run one job per batch via shell loop:

```bash
for i in {1..10}; do
    mkdir -p /tmp/my_dataset
    lucid-run-job --config "$CONFIG" --output-dir /tmp/my_dataset \
        --job-id $i --n-events 100
done
# → 10 parallel batches: wc_*_0000.h5 through wc_*_0009.h5 in each subdir.
```

For SLURM or HPC submission on S3DF, see
[QUICKSTART_S3DF.md](QUICKSTART_S3DF.md).

## Neutrino-flux configs (GENIE chain)

Configs with `"primary_source": "genie"` (e.g. `dataprod_13_numu.json`,
`dataprod_14_nue.json`) chain **gevgen → gntpc → PhotonSim → LUCiD**
per job. Dependencies are GEANT4 + ROOT + GENIE v3 + the LUCiD Python
env. The simplest way to cover all of those is the unified container:

```bash
# One-time build (needs Apptainer 1.2+ with fakeroot support; ~75 min
# including the ~60 min base layer).
cd LUCiD/container
apptainer build --fakeroot lucid-base.sif lucid-base.def
apptainer build --fakeroot lucid.sif       lucid.def

# Run any GENIE config in one invocation:
apptainer exec lucid.sif \
    lucid-run-job \
        --config /opt/LUCiD/lucid/production/configs/dataprod_13_numu.json \
        --output-dir /tmp/genie_test --job-id 1 --test
```

The container ships PhotonSim pre-built, LUCiD pip-installed, and the
GENIE cross-section splines for tune `G18_02a_00_000` baked in at
`/opt/genie_xsec/gxspl-FNALsmall.xml` — no cvmfs, no runtime
bind-mount, no first-run network. To use a different tune, override
`GENIE_XSEC_FILE` at runtime (e.g.
`apptainer exec -B /my/xsec:/opt/genie_xsec --env GENIE_XSEC_FILE=... lucid.sif ...`).

Docker users see [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md) for the
equivalent Docker invocation.

## Troubleshooting

- **`lucid-run-job: command not found`** — did you run `pip install -e .`
  from the LUCiD repo root?
- **`Error: PHOTONSIM_BIN env var must point to the built PhotonSim binary`** —
  export the absolute path.
- **PhotonSim complains about missing `libG4…` shared libs** — your
  `LD_LIBRARY_PATH` is missing the GEANT4 install. Either `source
  $GEANT4_PREFIX/bin/geant4.sh` or rebuild PhotonSim with RPATHs.
- **LUCiD complains about missing JAX** — `pip install -e .` should
  have pulled JAX. If not, `pip install jax jaxlib`. For GPU,
  `pip install -U "jax[cuda12]"`.

## What the runner does

1. Parse the JSON config.
2. Generate a Geant4 macro (`job_<id>.mac`).
3. Run `$PHOTONSIM_BIN <macro>` (subprocess) → ROOT file.
4. Run the LUCiD v3 writer in-process → four `wc_*_NNNN.h5` files.
5. Verify the four files open cleanly and carry the right provenance.
6. Delete the ROOT file if `cleanup_root_files: true` in the config.

Under the hood it's `lucid/production/run_job.py`; inspect that file to
add custom steps or swap components.
