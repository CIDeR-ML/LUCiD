# Container build

Single `Dockerfile` producing the unified GENIE → PhotonSim → LUCiD
image. The published image at `ghcr.io/cider-ml/lucid:latest` is the
canonical runtime; rebuild only when editing the Dockerfile itself.

Layered on top of `nuisancemc/tutorial:nuint2024`, which ships
ROOT 6.30 (with Pythia6), GENIE 3.04, Pythia6, and xsec splines for
tunes `AR23_20i_00_000`, `G18_10a_02_11b`, and `G21_11a_00_000`. The
Dockerfile adds conda GEANT4 11.3 + PhotonSim + LUCiD.

## Pull (default path)

- Docker: `docker pull ghcr.io/cider-ml/lucid:latest` — see
  [../docs/QUICKSTART_DOCKER.md](../docs/QUICKSTART_DOCKER.md).
- Apptainer/S3DF:
  `apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest` —
  see [../docs/QUICKSTART_S3DF.md](../docs/QUICKSTART_S3DF.md).

## Rebuild (only for Dockerfile edits)

From the parent directory containing both `LUCiD/` and `PhotonSim/`:

```bash
docker build --platform linux/amd64 \
    -f LUCiD/container/Dockerfile -t lucid:latest .
```

Cold build is ~10 min on Apple Silicon. Editing LUCiD source retriggers
only the last layer (~30 s). For a quicker dev loop without rebuilding,
bind-mount the checkout — see QUICKSTART_DOCKER §3.
