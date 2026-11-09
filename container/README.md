# Container

`Dockerfile` builds the image that runs the dataset production chain (GENIE, PhotonSim, LUCiD).
The published image is `ghcr.io/cider-ml/lucid:latest`; rebuild only to change the Dockerfile.
Using the image is described in the [container guide](../docs/guides/production/container.md)
and, for clusters, in [clusters](../docs/guides/production/clusters.md).

```bash
docker pull ghcr.io/cider-ml/lucid:latest
apptainer pull lucid.sif docker://ghcr.io/cider-ml/lucid:latest      # Apptainer, e.g. on S3DF
```

## Contents

- Base image `nuisancemc/tutorial:nuint2024`: ROOT 6.30 with Pythia6, GENIE 3.04, and cross-section
  splines for the tunes `AR23_20i_00_000`, `G18_10a_02_11b` and `G21_11a_00_000`.
- Miniforge Python 3.11 at `/opt/conda` with Geant4 11.3 and LUCiD's Python dependencies,
  including `jax=0.4.38` with a CPU-only `jaxlib`.
- LUCiD installed editable at `/opt/LUCiD` with `pip install --no-deps`. pip therefore does not
  apply `pyproject.toml`'s `jax>=0.5.3,<0.6`, and the image runs jax 0.4.38.
- PhotonSim built at `/opt/PhotonSim/build`.
- PMT overlap tables pre-computed for the `*_WAND` detectors. Production jobs do not use them.
- `/opt/VERSIONS`: the LUCiD and PhotonSim commits, whether the LUCiD build context had
  uncommitted changes (`lucid_dirty`), the build date and the Geant4 version. Every dataset batch
  written in the image records the LUCiD commit and dirty flag as `config/git_commit` and
  `git_dirty`.

## Rebuild

From the directory that contains both `LUCiD/` and `PhotonSim/`:

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false \
    --build-arg LUCID_SHA=$(git -C LUCiD rev-parse HEAD) --build-arg LUCID_DIRTY=false \
    -f LUCiD/container/Dockerfile -t lucid:latest .
```

Without `LUCID_SHA` the image, and every batch it writes, records `lucid_sha=unknown`. Set
`LUCID_DIRTY=true` when the checkout has uncommitted changes. A change to LUCiD's source rebuilds
only the last layers. To run a modified checkout without rebuilding, bind-mount it over
`/opt/LUCiD` as the [container guide](../docs/guides/production/container.md) describes.
