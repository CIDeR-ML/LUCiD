# Produce a dataset from the container

Everything needed to generate WAND events lives in one image: LUCiD, PhotonSim,
Geant4 and the configs. You need the image and a writable output directory —
nothing else, no checkout, no build.

This page is about producing data by hand. Driving a batch farm is a separate
concern; see [deploy-lxplus.md](deploy-lxplus.md) and friends.

## 1. Get the image

Pick a release tag, not `latest`, so your data is traceable to a known build:

```bash
apptainer build lucid_v1.0.0.sif docker://ghcr.io/cider-ml/lucid:v1.0.0
```

Budget ~30 GB of scratch and a few tens of minutes: the layers unpack to ~27 GB
before being squashed back to ~3.5 GB. `APPTAINER_TMPDIR` must point somewhere
with that much room. Docker/podman users can skip the build and
`docker run ghcr.io/cider-ml/lucid:v1.0.0` directly.

The image can tell you what it is:

```bash
apptainer exec lucid_v1.0.0.sif cat /opt/VERSIONS
```

## 2. Run one job

```bash
mkdir -p out
apptainer exec -B "$PWD/out:/out" lucid_v1.0.0.sif \
    lucid-run-job \
      --config /opt/LUCiD/lucid/production/configs/GeV/01_pbomb.json \
      --detector SK_WAND \
      --output-dir /out \
      --job-id 1 \
      --n-events 20 \
      --master-seed 12345
```

It prints three stages — Geant4 macro, PhotonSim, LUCiD writer — and ends with a
line starting `OK`. Anything else means it failed; the exit code is non-zero.

Two arguments are not optional in practice:

- `--detector` has no useful default. It selects the geometry **and** the
  digitizer and trigger, so getting it wrong silently produces a different
  detector's data. Use `SK_WAND` or `HK_WAND`.
- `--master-seed` defaults to random. Set it if you want to be able to
  reproduce the job.

## 3. The configs

| config | contents | energies (kinetic) | split |
|---|---|---|---|
| `GeV/01_pbomb.json` | 1–5 particles from {e-, mu-, pi+, pi-, pi0, gamma} at one vertex | 1–2000 e-, 200–2000 mu/pi, 1–2130 gamma | train + test |
| `GeV/02_mu.json` | single mu- | 200–2000 | test |
| `GeV/03_pi_plus.json` | single pi+ | 200–2000 | test |
| `GeV/04_e.json` | single e- | 1–2000 | test |
| `GeV/05_pi_minus.json` | single pi- | 200–2000 | test |
| `GeV/06_pi0.json` | single pi0 | 200–2000 | test |
| `GeV/07_gamma.json` | single gamma | 1–2130 | test |

01 is the training set — it mixes species and multiplicities the way a real
interaction does. 02–07 are one-species evaluation sets, one per member of the
01 pool.

Blocks other than `GeV` exist (`Solar`, `SN`, `Test`); those declare their own
detector and are not part of WAND.

## 4. Detectors

`--detector NAME` reads `config/NAME_geom_config.json` and
`config/NAME_physics_config.json` from inside the image.

| detector | PMTs | digitizer | trigger |
|---|---|---|---|
| `SK_WAND` | 11096 | `ski` | 200 ns window, n_thr 25 |
| `HK_WAND` | 19746 | `hk` | 200 ns window, n_thr 45 |

These are frozen descriptions: they pin the optics, digitizer and trigger so a
rerun cannot drift when the generic `SK`/`HK` configs are edited. Use them.

A dataset config carries no detector of its own, so `--detector` is what decides
it. Passing a non-`*_WAND` detector to one of these configs is refused rather
than silently run, because the generic descriptions carry no digitizer or
trigger and you would get `basic` digitization with no readout trigger.

## 5. Changing things

- **More events**: `--n-events N`. Cost is roughly linear; the bomb runs
  ~10–30 s/event depending on detector and energies.
- **Fixed energy**: `--override-energy-MeV 500` forces monoenergetic primaries.
- **Keep the intermediate ROOT**: `--keep-root`. It is large (~10 MB/event) and
  deleted by default.
- **Your own config**: copy one out of the image, edit, and bind it back in:

  ```bash
  apptainer exec lucid_v1.0.0.sif \
      cat /opt/LUCiD/lucid/production/configs/GeV/02_mu.json > mine.json
  # edit mine.json
  apptainer exec -B "$PWD:/w" lucid_v1.0.0.sif \
      lucid-run-job --config /w/mine.json --detector SK_WAND \
        --output-dir /w/out --job-id 1 --n-events 20
  ```

  The fields worth touching are `particles` (type and energy range),
  `nominal_train`/`nominal_test`, and `selection`. Leave `digitizer` and
  `trigger` out — they come from the detector.

- **Shared overlap cache**: the first job on a fresh install computes a lookup
  table. Release images ship it warm. If you need to relocate it, the order is
  `$LUCID_OVERLAP_CACHE_DIR` → install dir if writable →
  `$XDG_CACHE_HOME/lucid/spatial_overlap_integrals`.

## 6. Where the output goes

`--output-dir` is written directly, as four HDF5 files:

```
<output-dir>/
  sensor/wc_sensor_0000.h5    digitized PMT hits (charge, time)
  hits/wc_hits_0000.h5        per-digit truth decomposition
  step/wc_step_0000.h5        Geant4 steps + per-step photon counts
  labl/wc_labl_0000.h5        event/interaction/track labels
```

The `0000` is `file_index = job-id - 1`, so parallel jobs writing to one
directory each need a distinct `--job-id`.

Every file records what produced it:

```bash
python3 -c "
import h5py; a=h5py.File('out/sensor/wc_sensor_0000.h5')['config'].attrs
print({k: a[k] for k in ('git_commit','digitizer_model','n_sensors','trigger_n_thr')})"
```

Note that events failing the trigger are dropped, so the stored event count can
be lower than `--n-events`. `labl`'s `config/source_event_idx` maps each stored
event back to its generated index — use it to join datasets, never shard
position, since two detectors drop different events.

For what is inside each file, see
[dataset-schema.md](../../reference/dataset-schema.md).
