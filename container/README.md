# Container build

Single `Dockerfile` producing the unified GENIE → PhotonSim → LUCiD
image. The published image at `ghcr.io/cider-ml/lucid:latest` is the
canonical runtime; rebuild only when editing the Dockerfile itself.

## What the image contains

This page is the one description of the image; the production guides link here.

- Base `nuisancemc/tutorial:nuint2024`: ROOT 6.30 (with Pythia6), GENIE 3.04,
  Pythia6, and xsec splines for tunes `AR23_20i_00_000`, `G18_10a_02_11b` and
  `G21_11a_00_000`.
- Conda (miniforge) Python 3.11 with GEANT4 11.3 and LUCiD's Python stack,
  including `jax=0.4.38` (CPU `jaxlib`). LUCiD is installed editable at
  `/opt/LUCiD` with `pip install --no-deps`, so pip does not enforce
  `pyproject.toml`'s `jax>=0.5.3,<0.6`: the image runs a jax outside the
  range the library declares.
- PhotonSim built at `/opt/PhotonSim/build`.
- The spatial-overlap tables warmed for the `*_WAND` detectors
  (production jobs do not read them; see the container guide, §5).
- `/opt/VERSIONS`: the LUCiD and PhotonSim commits, whether the LUCiD build
  context had tracked changes (`lucid_dirty`), the build date and the Geant4
  version. Every dataset batch the image writes records `lucid_sha` and
  `lucid_dirty` as `config/git_commit` and `git_dirty`.

## Pull (default path)

- Docker: `docker pull ghcr.io/cider-ml/lucid:latest` — see
  [../docs/guides/production/container.md](../docs/guides/production/container.md).
- Apptainer/S3DF:
  `apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest` —
  see [../docs/guides/production/clusters.md](../docs/guides/production/clusters.md).

## Rebuild (only for Dockerfile edits)

From the parent directory containing both `LUCiD/` and `PhotonSim/`:

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false \
    -f LUCiD/container/Dockerfile -t lucid:latest .
```

`--provenance=false --sbom=false` suppresses BuildKit attestation manifests
(ghcr.io has trouble with them for this package). Without
`--build-arg LUCID_SHA=$(git -C LUCiD rev-parse HEAD)` (and `LUCID_DIRTY=true`/`false`) a
local image records `lucid_sha=unknown`, and so does every batch it writes. Cold build is ~10 min
on Apple Silicon. Editing LUCiD source retriggers only the last layer
(~30 s). For a quicker dev loop without rebuilding, bind-mount the
checkout — see [the container guide, §6](../docs/guides/production/container.md#6-dev-loop--run-your-own-checkout).
