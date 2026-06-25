# LUCiD Production Pipeline

Turns a PhotonSim particle-based ROOT file into a v3 four-file HDF5 dataset
(`sensor/`, `inst/`, `seg/`, `labl/`). Full schema in
[`docs/LUCID_DATASET.md`](../../docs/LUCID_DATASET.md).

## Example

```bash
python -m lucid.production.generate_events_with_particles \
    --root-file /path/to/photonsim_output.root \
    --output     /path/to/dataset_root/ \
    --dataset-name my_run_2026_04_20 \
    --apply-smearing
```

Writes `{output}/{sensor,hits,step,labl}/wc_*_0000.h5`. Use `--help` for
the full flag list (`--batch-size`, `--n-events`, `--master-seed`,
`--physics-config`, ...).

**Units:** meters, nanoseconds, MeV throughout.

## Medium / scintillation (WbLS)

The medium is set by the **detector config** (`--detector`), not by the
dataprod config's `material` field. The default `--detector SK_like` is
water → Cherenkov-only output (every `emission_process` row is 0). To
produce scintillation, run with `--detector SK_like_wbls`; the log then
reports `geometry: cylinder / wbls` and `emission_process` carries both
Cherenkov (0) and scintillation (1) hits.

## End-to-end (PhotonSim + LUCiD + HTMLs)

`generate_validation_htmls.sh` runs PhotonSim, feeds its ROOT into the
script above, and emits per-event HTML visualizations. Currently wired
for S3DF (Singularity image + `/sdf/...` paths); adapt for local use.

```bash
./lucid/production/generate_validation_htmls.sh -c 07 -n 5
```

A longer doc with the full pipeline walkthrough and a portable wrapper
will land later.
