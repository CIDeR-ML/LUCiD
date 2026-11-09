# Produce a dataset from the container

The image `ghcr.io/cider-ml/lucid` holds LUCiD, PhotonSim, GEANT4, ROOT, GENIE with its cross
sections, and the dataset configs. With it, a job needs only a writable output directory. It
runs under Apptainer or Docker. For batch farms, see [on a cluster](clusters.md).

## 1. Get the image

Use a release tag rather than `latest`, so the data can be traced to a build:

```bash
TAG='<release-tag>'     # e.g. v0.1.0
```

Apptainer:

```bash
apptainer build lucid_$TAG.sif docker://ghcr.io/cider-ml/lucid:$TAG
```

The build unpacks about 27 GB before squashing to about 3.5 GB: point `APPTAINER_TMPDIR` at
30 GB of scratch and allow a few tens of minutes.

Docker or podman (about 4 GB; keep 10 GB free):

```bash
docker pull --platform linux/amd64 ghcr.io/cider-ml/lucid:$TAG
```

The image is `linux/amd64` only. On Apple Silicon, install Rosetta 2
(`softwareupdate --install-rosetta --agree-to-license`) and enable "Use Rosetta for x86/amd64
emulation" in Docker Desktop (Settings, General); expect 70 to 80% of native speed.

`apptainer exec lucid_$TAG.sif cat /opt/VERSIONS` prints what the image contains.

## 2. Run one job

```bash
mkdir -p out
apptainer exec -B "$PWD/out:/out" lucid_$TAG.sif \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/01_pbomb.json \
      --detector SK_WAND --output-dir /out --job-id 1 --n-events 20 --master-seed 12345
```

or with Docker:

```bash
docker run --rm --platform linux/amd64 -v "$PWD/out:/out" ghcr.io/cider-ml/lucid:$TAG \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/01_pbomb.json \
      --detector SK_WAND --output-dir /out --job-id 1 --n-events 20 --master-seed 12345
```

It reports three stages (GEANT4 macro, PhotonSim, LUCiD writer) and ends with a line starting
`OK`; anything else is a failure, with a non-zero exit code.

- `--detector` (`SK_WAND` or `HK_WAND`) chooses the geometry and the readout for a `GeV` config
  (section 4).
- `--master-seed` is random by default; set it to make the job reproducible.

All flags: [command-line tools](../../reference/cli.md#lucid-run-job).

## 3. The configs

| config | events | kinetic energy (MeV) |
|---|---|---|
| `GeV/01_pbomb.json` | 1 to 5 particles from {e-, e+, mu-, mu+, pi+, pi-, pi0, gamma} at one vertex | e 1 to 2000, mu and pi 200 to 2000, gamma 1 to 2130 |
| `GeV/02_mu.json` | one mu, charge drawn per event | 200 to 2000 |
| `GeV/03_pi.json` | one charged pi, charge drawn per event | 200 to 2000 |
| `GeV/04_e.json` | one e, charge drawn per event | 1 to 2000 |
| `GeV/06_pi0.json` | one pi0 | 200 to 2000 |
| `GeV/07_gamma.json` | one gamma | 1 to 2130 |

`01` is the training set; `02` to `07` are one-species test sets. The single-particle sets draw
the charge per event because the two charges end differently (mu- can be captured, mu+ always
decays; pi- is captured, pi+ decays), while their Cherenkov light does not tell them apart. In a
config of your own, express "one particle of either charge" as a bomb with
`min_particles = max_particles = 1`: a `particles` list with two entries puts both in every
event. Splits and counts of all blocks: [config blocks](index.md#config-blocks).

## 4. Detectors

`--detector NAME` reads `config/NAME_geom_config.json` and `config/NAME_physics_config.json`
inside the image.

| detector | PMTs | digitizer | trigger |
|---|--:|---|---|
| `SK_WAND` | 11096 | `ski` | 200 ns window, `n_thr` 25 |
| `HK_WAND` | 19746 | `hk` | 200 ns window, `n_thr` 45 |

These two fix the optics, digitizer and trigger for production. A `GeV` config declares no
detector and no readout, so it takes both from `--detector`. The other detectors (`SK`, `HK`,
`SK_like`, ...) have no readout, and `lucid-run-job` refuses them for a `GeV` config before
PhotonSim runs. To use one, add `"digitizer": "basic"` (or `"ski"`, `"hk"`) and
`"trigger": null` (or a trigger block) to a copy of the dataset config
([digitizer and trigger](../../reference/digitizer-and-trigger.md)).

The `Solar`, `SN` and `Test` configs declare `"detector": "HK"` and their own readout; they need
no `--detector`, and a different one stops the job.

## 5. Changing things

- More events: `--n-events N`. The bomb takes about 10 to 30 s per event.
- Fixed energy: `--override-energy-MeV 500`.
- Keep the PhotonSim ROOT file (about 10 MB per event): `--keep-root`.
- Your own config: copy one out, edit it, and bind it in:

  ```bash
  apptainer exec lucid_$TAG.sif \
      cat /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json > mine.json
  apptainer exec -B "$PWD:/w" lucid_$TAG.sif \
      lucid-run-job --config /w/mine.json --detector SK_WAND --output-dir /w/out --job-id 1 --n-events 20
  ```

  The fields to edit are `particles` or `bomb.candidates` (species and energy range),
  `nominal_train` and `nominal_test`, and `selection`.
- A GENIE config (`"primary_source": "genie"`; none ships) runs `gevgen` and `gntpc` first. The
  image's cross sections are named by
  `GENIE_XSEC_FILE=/opt/genie_xsec/3_04_00/G18_10a_02_11b/gxspl-min.xml.gz`
  (`-e` for Docker); point it at another spline file for another tune.

Production jobs use the hard sensor assignment and do not read the soft-overlap tables. For
interactive use, overlap tables are read from `spatial_overlap_integrals/` in the install, then
from `~/.cache/lucid/spatial_overlap_integrals/`, and a new table is written to the first
writable one. `lucid-run-job` also honours `$XDG_CACHE_HOME` (moves the second location) and
`$LUCID_OVERLAP_CACHE_DIR` (replaces both).

## 6. Dev loop — run your own checkout

Bind a clone over `/opt/LUCiD` to run your edits without rebuilding the image:

```bash
apptainer exec -B "$PWD/LUCiD:/opt/LUCiD" -B "$PWD/out:/out" lucid_$TAG.sif \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json \
      --detector SK_WAND --output-dir /out --job-id 1 --test
```

For PhotonSim edits, bind its clone over `/opt/PhotonSim` and rebuild inside the image
(incremental builds take under a minute):

```bash
apptainer exec -B "$PWD/PhotonSim:/opt/PhotonSim" -B "$PWD/out:/out" lucid_$TAG.sif \
    bash -c "cmake --build /opt/PhotonSim/build -j && lucid-run-job ..."
```

A clone with no `build/` yet needs the first configure described in
[the cluster dev loop](cluster-abstraction.md#dev-loop-run-your-own-checkouts-in-the-batch-jobs).

## 7. Where the output goes

`--output-dir` is the dataset root. A job writes one batch, `sensor/`, `hits/`, `step/` and
`labl/wc_<kind>_NNNN.h5`, with `NNNN = job-id − 1`, so parallel jobs into one directory need
distinct `--job-id`. Each file records what made it:

```bash
python3 -c "
import h5py; a = h5py.File('out/sensor/wc_sensor_0000.h5')['config'].attrs
print({k: a[k] for k in ('git_commit', 'digitizer_model', 'n_sensors', 'trigger', 'complete')})"
```

Events the trigger rejects are dropped, so a batch can hold fewer than `--n-events`. Join
datasets on `config/source_event_idx`, not on position: two detectors drop different events.

## Building the image

Only needed to change the `Dockerfile`; see
[`container/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/container/README.md).

## Troubleshooting

- `exec /bin/bash: exec format error`: Apple Silicon without Rosetta enabled (section 1).
- `declares no 'detector'; pass --detector`, or `detector needs to be specified` from the
  fan-out: pass `--detector SK_WAND` or `HK_WAND`.
- `no 'digitizer' block` or `no 'trigger' block`: the detector has no readout and the config
  states none (section 4).
- `declares detector 'HK', but --detector is ...`: a `Solar`, `SN` or `Test` config with another
  detector; drop `--detector`.
- Silent long Docker build steps: use `--progress=plain`.
- `No space left on device`: `docker system prune -af`.
