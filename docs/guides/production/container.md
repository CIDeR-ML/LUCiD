# Produce a dataset from the container

Everything needed to generate WAND events lives in one image: LUCiD, PhotonSim,
Geant4 and the configs. To run a job you need the image and a writable output
directory, no build. Works with Apptainer (HPC) or Docker.

This page is about producing data by hand. Driving a batch farm is a separate
concern; see [cluster deployment](clusters.md). For a host-native
install without a container, see [local.md](local.md).

## 1. Get the image

Pick a release tag, not `latest`, so your data is traceable to a known build,
and set it once; the commands below use it:

```bash
TAG='<release-tag>'     # e.g. v0.1.0
```

**Apptainer:** build the `.sif` from the published image:

```bash
apptainer build lucid_$TAG.sif docker://ghcr.io/cider-ml/lucid:$TAG
```

Budget ~30 GB of scratch and a few tens of minutes: the layers unpack to ~27 GB
before being squashed back to ~3.5 GB. `APPTAINER_TMPDIR` must point somewhere
with that much room.

**Docker/podman** skip the build entirely — pull and go (~4 GB, ~10 GB free
needed):

```bash
docker pull --platform linux/amd64 ghcr.io/cider-ml/lucid:$TAG
```

The image is `linux/amd64`; conda-forge ships no `geant4` for `linux-aarch64`.
On Apple Silicon install Rosetta 2
(`softwareupdate --install-rosetta --agree-to-license`) and enable *Use Rosetta
for x86/amd64 emulation* in Docker Desktop → Settings → General. Expect 70–80%
of native speed — fine for test runs.

The image can tell you what it is:

```bash
apptainer exec lucid_$TAG.sif cat /opt/VERSIONS
```

## 2. Run one job

```bash
mkdir -p out
apptainer exec -B "$PWD/out:/out" lucid_$TAG.sif \
    lucid-run-job \
      --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/01_pbomb.json \
      --detector SK_WAND \
      --output-dir /out \
      --job-id 1 \
      --n-events 20 \
      --master-seed 12345
```

The Docker equivalent — same arguments, different mount syntax:

```bash
docker run --rm --platform linux/amd64 -v "$PWD/out:/out" \
    ghcr.io/cider-ml/lucid:$TAG \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/01_pbomb.json \
                  --detector SK_WAND --output-dir /out --job-id 1 --n-events 20
```

It prints three stages — Geant4 macro, PhotonSim, LUCiD writer — and ends with a
line starting `OK`. Anything else means it failed; the exit code is non-zero.

Two arguments are not optional in practice:

- `--detector` is required for a GeV config (unless `--skip-lucid`). It selects
  the geometry **and** the digitizer and trigger: pass `SK_WAND` or `HK_WAND`.
  Any other detector carries no readout, and the job stops before PhotonSim
  (§4). A config that declares its own `detector` needs no flag, and one given
  must match it.
- `--master-seed` defaults to random. Set it if you want to be able to
  reproduce the job.

## 3. The configs

| config | contents | energies (kinetic, MeV) |
|---|---|---|
| `GeV/01_pbomb.json` | 1–5 particles from {e-, e+, mu-, mu+, pi+, pi-, pi0, gamma} at one vertex | 1–2000 e, 200–2000 mu/pi, 1–2130 gamma |
| `GeV/02_mu.json` | single mu, charge sampled per event from {mu-, mu+} | 200–2000 |
| `GeV/03_pi.json` | single charged pi, charge sampled per event from {pi+, pi-} | 200–2000 |
| `GeV/04_e.json` | single e, charge sampled per event from {e-, e+} | 1–2000 |
| `GeV/06_pi0.json` | single pi0 | 200–2000 |
| `GeV/07_gamma.json` | single gamma | 1–2130 |

01 is the training set — it mixes species and multiplicities the way a real
interaction does. 02–07 are one-species evaluation sets, one per member of the
01 pool. The splits and event counts of every block are listed in
[Producing datasets](index.md#config-blocks).

Both lepton and pion charges are in the pool, and the single-species sets sample
the charge per event rather than fixing it. Cherenkov light carries no charge
information, so a set of one charge only measures half of what reconstruction is
asked to do — and the charges are not interchangeable at the end of the range:
mu- captures on oxygen while mu+ always yields its decay electron, and pi- is
captured by a nucleus while pi+ decays to mu+ nu. The charge-sampled sets are
what state how often each signature appears.

Sampling is expressed through the bomb source with `min_particles =
max_particles = 1`, not through a two-entry `particles` list: `dataprod/generators/macro.py`
emits one `/gun/addPrimary` per list entry, so a list of {e-, e+} would put
*both* charges in every event.

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

A GeV dataset config carries no detector of its own, so `--detector` is what
decides it, and the config takes its `digitizer` and `trigger` from that
detector's physics config. The generic descriptions (`SK`, `HK`, `SK_like`, ...)
carry neither, so `lucid-run-job` and the batch fan-out
(`lucid.production.dataprod.fanout`) both refuse them for such a config, before
PhotonSim runs ("no 'digitizer' block"). To run one anyway, state the readout
in a copy of the dataset config: `"digitizer": "basic"` (or `"ski"`, `"hk"`)
and `"trigger": null` (or a trigger block).

A config that declares its own `detector` (`Solar`, `SN`, `Test`: `HK`) needs no
`--detector`; one given must name the same detector, or the job stops.

## 5. Changing things

- **More events**: `--n-events N`. Cost is roughly linear; the bomb runs
  ~10–30 s/event depending on detector and energies.
- **Fixed energy**: `--override-energy-MeV 500` forces monoenergetic primaries.
- **Keep the intermediate ROOT**: `--keep-root`. It is large (~10 MB/event) and
  deleted by default.
- **Your own config**: copy one out of the image, edit, and bind it back in:

  ```bash
  apptainer exec lucid_$TAG.sif \
      cat /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json > mine.json
  # edit mine.json
  apptainer exec -B "$PWD:/w" lucid_$TAG.sif \
      lucid-run-job --config /w/mine.json --detector SK_WAND \
        --output-dir /w/out --job-id 1 --n-events 20
  ```

  The fields worth touching are `particles` or `bomb.candidates` (type and
  energy range),
  `nominal_train`/`nominal_test`, and `selection`. Leave `digitizer` and
  `trigger` out — they come from the `*_WAND` detector (§4).

- **Spatial overlap cache**: the soft-overlap lookup tables
  (`spatial_overlap_integrals/*.json`) are looked up in the install dir, then
  in `~/.cache/lucid/spatial_overlap_integrals`; a newly computed table is
  written to the first of the two that is writable. The library reads no
  environment variable for this. `lucid-run-job` reads two: `$XDG_CACHE_HOME`
  moves the second location to `$XDG_CACHE_HOME/lucid/...`, and
  `$LUCID_OVERLAP_CACHE_DIR` replaces both. Production jobs run the hard sensor step
  (`temperature=0`) and never read these tables; the image ships them warm for
  the `*_WAND` detectors at the library's default temperature, for
  interactive use.

## 6. Dev loop — run your own checkout

To exercise local edits without rebuilding, bind your clone over `/opt/LUCiD`:

```bash
apptainer exec -B "$PWD/LUCiD:/opt/LUCiD" -B "$PWD/out:/out" lucid_$TAG.sif \
    lucid-run-job --config /opt/LUCiD/lucid/production/dataprod/configs/GeV/02_mu.json \
      --detector SK_WAND --output-dir /out --job-id 1 --test
```

For a PhotonSim source edit, bind it too and rebuild in place; the baked
`/opt/PhotonSim/build` stays intact so incremental compiles take under a minute:

```bash
apptainer exec -B "$PWD/PhotonSim:/opt/PhotonSim" -B "$PWD/out:/out" \
    lucid_$TAG.sif bash -c \
    "cmake --build /opt/PhotonSim/build -j && lucid-run-job ..."
```

Note that a bind-mounted checkout is writable, so the overlap cache lands there
rather than in the user cache — see §5.

## 7. Where the output goes

`--output-dir` is the dataset root. The job writes one batch there, four HDF5
files `sensor/`, `hits/`, `step/` and `labl/wc_<sub>_NNNN.h5` (layout and fields
in [dataset-schema.md](../../reference/dataset-schema.md)). `NNNN` is
`file_index = job-id - 1`, so parallel jobs writing to one directory each need a
distinct `--job-id`.

Every file records what produced it:

```bash
python3 -c "
import h5py; a=h5py.File('out/sensor/wc_sensor_0000.h5')['config'].attrs
print({k: a[k] for k in ('git_commit','digitizer_model','n_sensors','trigger','complete')})"
```

Note that events failing the trigger are dropped, so the stored event count can
be lower than `--n-events`. `labl`'s `config/source_event_idx` maps each stored
event back to its generated index — use it to join datasets, never shard
position, since two detectors drop different events.

## Building the image yourself

Only needed if you are changing the `Dockerfile`. What the image contains and
the build command (LUCiD and PhotonSim cloned as siblings, built from the
parent directory) are in
[`container/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/container/README.md).

## Troubleshooting

- **`exec /bin/bash: exec format error`** — Apple Silicon without Rosetta
  enabled in Docker Desktop. See §1.
- **`Error: detector needs to be specified`** (fan-out) or **`declares no
  'detector'; pass --detector`** (`lucid-run-job`) — the GeV configs declare no
  detector; pass `-D SK_WAND` / `--detector SK_WAND` (or `HK_WAND`).
- **`no 'digitizer' block`** / **`no 'trigger' block`** — the detector you
  passed (`SK`, `HK`, ...) carries no readout and the dataset config states
  none. Use a `*_WAND` detector, or state the readout in the config (§4).
- **`declares detector 'HK', but --detector is ...`** — a `Solar`/`SN`/`Test`
  config run with another detector. Drop `--detector` or pass the declared one.
- **BuildKit silence during source builds** — long compile steps buffer output;
  use `--progress=plain`.
- **"No space left on device"** — `docker system prune -af` reclaims old layers.
