# Local production — produce events on any machine

> **On macOS, or prefer a container?** See
> [container.md](container.md) — one pull,
> then one `docker run`, with GEANT4 + ROOT + GENIE already set up.

Three steps to a working dataset with no S3DF, no SLURM, no
singularity.

## Prerequisites

- **GEANT4 11.3+** and **ROOT 6.0+** installed on your system. On
  Linux, `conda install -c conda-forge geant4 root` is the fastest
  path.
- **Python 3.10+** (`requires-python` in `pyproject.toml`).
- **GENIE v3** (only if you write a neutrino-flux config; none ship in the
  repo today). Easiest: use the unified container — see the bottom of this doc.

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

Follow [Install](../../getting-started/install.md) (clone, then
`pip install -e . -c tests/constraints.txt`). The console script
`lucid-run-job` is then on your PATH.

## 3. Run one job

```bash
# Tell lucid-run-job where the PhotonSim binary lives:
export PHOTONSIM_BIN=/absolute/path/to/PhotonSim/build/PhotonSim

# Pick any of the bundled configs:
CONFIG=$(python3 -c "from importlib.resources import files; \
    print(files('lucid.production.dataprod.configs').joinpath('GeV/02_mu.json'))")

# Run:
mkdir -p /tmp/my_dataset
lucid-run-job \
    --config "$CONFIG" \
    --detector SK_WAND \
    --output-dir /tmp/my_dataset \
    --job-id 1 \
    --test
```

`/tmp/my_dataset/` then holds one batch, `{sensor,hits,step,labl}/wc_*_0000.h5`
(layout and fields in the [dataset schema](../../reference/dataset-schema.md)).

## 4. Inspect in a browser (optional)

```bash
python3 viewer/serve_viewer.py /tmp/my_dataset --open
```

Opens the interactive PMT + segment display on
`http://127.0.0.1:8765/`.

## Running a full dataset locally

Drop `--test` and pass `--n-events N` to run `N` events per job. Without
either flag a job runs the config's `n_events_per_job`, which no shipped
config sets, so it runs 100. Run one job per batch via a shell loop:

```bash
for i in {1..10}; do
    mkdir -p /tmp/my_dataset
    lucid-run-job --config "$CONFIG" --detector SK_WAND --output-dir /tmp/my_dataset \
        --job-id $i --n-events 100
done
# → 10 batches: wc_*_0000.h5 through wc_*_0009.h5 in each subdir.
```

For SLURM or HTCondor submission, see
[cluster deployment](clusters.md).

## Neutrino-flux configs (GENIE chain)

A config with `"primary_source": "genie"` chains
**gevgen → gntpc → PhotonSim → LUCiD**
per job. Dependencies are GEANT4 + ROOT + GENIE v3 + the LUCiD Python
env. The simplest way to cover all of those is the published container:

```bash
# Docker
docker pull ghcr.io/cider-ml/lucid:latest
docker run --rm --platform linux/amd64 \
    -v /tmp/genie_test:/out \
    -e GENIE_XSEC_FILE=/opt/genie_xsec/3_04_00/G18_10a_02_11b/gxspl-min.xml.gz \
    ghcr.io/cider-ml/lucid:latest \
    lucid-run-job \
        --config /path/to/your_genie_config.json \
        --detector SK_WAND \
        --output-dir /out --job-id 1 --test

# Apptainer
apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest
apptainer exec lucid.sif lucid-run-job \
    --config /path/to/your_genie_config.json \
    --detector SK_WAND \
    --output-dir /tmp/genie_test --job-id 1 --test
```

No GENIE config ships in the repo; the image has GENIE and its splines
(contents in [`container/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/container/README.md)).
For a tune other than the one `GENIE_XSEC_FILE` names, point it at your own
spline.

See [container.md](container.md) for more container
details (bind-mount dev loop, Rosetta setup, etc.).

## Troubleshooting

- **`lucid-run-job: command not found`** — did you run `pip install -e .`
  from the LUCiD repo root?
- **`Error: PHOTONSIM_BIN env var must point to the built PhotonSim binary`** —
  export the absolute path.
- **PhotonSim complains about missing `libG4…` shared libs** — your
  `LD_LIBRARY_PATH` is missing the GEANT4 install. Either `source
  $GEANT4_PREFIX/bin/geant4.sh` or rebuild PhotonSim with RPATHs.
- **LUCiD complains about missing JAX** — the install pulls it; see
  [Install](../../getting-started/install.md) for the CPU and GPU lines.

## What the runner does

1. Parse the JSON config.
2. Generate a Geant4 macro (`job_<id>.mac`).
3. Run `$PHOTONSIM_BIN <macro>` (subprocess) → ROOT file.
4. Run the LUCiD writer in-process → four `wc_*_NNNN.h5` files.
5. Verify the four files open cleanly and carry the right provenance.
6. Delete the ROOT file if `cleanup_root_files: true` in the config.

Under the hood it's `lucid/production/dataprod/run_job.py`; inspect that file to
add custom steps or swap components.
