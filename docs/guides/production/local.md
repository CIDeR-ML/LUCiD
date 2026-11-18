# Produce a dataset on your machine

This builds PhotonSim against your own GEANT4 and ROOT. On macOS, or to skip the build, use
[the container](container.md).

You need GEANT4 11.3 or later, ROOT 6, and Python 3.10 or later. On Linux,
`conda install -c conda-forge geant4 root` is the quickest route. GENIE is needed only for a
neutrino-flux config (none ships); the container has it.

## 1. Build PhotonSim

```bash
git clone https://github.com/cesarjesusvalls/PhotonSim.git
cd PhotonSim && mkdir build && cd build
cmake -DGeant4_DIR=$(geant4-config --prefix)/lib/cmake/Geant4 ..
make -j$(nproc)          # builds ./PhotonSim
```

## 2. Install LUCiD

[Install](../../getting-started/install.md) it; `lucid-run-job` is then on your `PATH`.

## 3. Run one job

```bash
export PHOTONSIM_BIN=/absolute/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu.json \
    --detector SK_WAND --output-dir /tmp/my_dataset --job-id 1 --test
```

`--test` runs 2 events. The job writes a GEANT4 macro, runs PhotonSim, runs the LUCiD writer,
checks the four output files, and deletes the ROOT file unless the config sets
`"cleanup_root_files": false` or you pass `--keep-root`. `/tmp/my_dataset/` then holds
`{sensor,hits,step,labl}/wc_*_0000.h5` ([schema](../../reference/dataset-schema.md)).

To look at it in a browser:

```bash
python viewer/serve_viewer.py /tmp/my_dataset --open      # http://127.0.0.1:8765/
```

## More events

Drop `--test` and set `--n-events` (default 100). Each job is one batch; give each a different
`--job-id`:

```bash
for i in {1..10}; do
    lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu.json --detector SK_WAND \
        --output-dir /tmp/my_dataset --job-id $i --n-events 100
done        # batches wc_*_0000.h5 to wc_*_0009.h5
```

For SLURM or HTCondor, see [on a cluster](clusters.md).

## Troubleshooting

- `lucid-run-job: command not found`: LUCiD is not installed in the active environment.
- `Error: PHOTONSIM_BIN env var must point to the built PhotonSim binary`: export the absolute
  path.
- PhotonSim cannot find `libG4...` libraries: `source $GEANT4_PREFIX/bin/geant4.sh`, or rebuild
  PhotonSim with RPATHs.
- `no 'digitizer' block`: the detector has no readout; see
  [detectors](container.md#4-detectors).
