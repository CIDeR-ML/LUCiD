# Working with LUCiD data

`lucid-run-job` writes each batch as **four parallel HDF5 files**, sharing per-event
indexing, under `<output_dir>/{sensor,hits,step,labl}/wc_*_<F:04d>.h5`:

| modality | file | contents |
|----------|------|----------|
| `sensor` | `sensor/wc_sensor_NNNN.h5` | raw per-PMT readout (post-digitization) |
| `hits`   | `hits/wc_hits_NNNN.h5`     | per-particle decomposition of the PMT signal |
| `step`   | `step/wc_step_NNNN.h5`     | 3D GEANT4 track segments (the per-segment energy deposit is the `edep` column) |
| `labl`   | `labl/wc_labl_NNNN.h5`     | labels, truth metadata, and dimension tables (truth `t0` lives here) |

Detector times in `sensor`/`hits`/`step` are in the **detector frame** (per-event `t0` already
added); truth `t0` is in `labl`. The full schema is in the [dataset reference](../LUCID_DATASET.md).

> Note: older datasets may name the segment modality `edep` instead of `step`.

## Reading a batch

`lucid.sources.reader` holds the low-level per-file readers (standalone — only needs `h5py`);
`lucid.production.verify_output.batch_paths` resolves the four files of a batch:

```python
from lucid.production.verify_output import batch_paths
from lucid.sources.reader import (list_events, read_sensor_event,
                                  read_hits_event, read_step_event, read_labl_event)

paths = batch_paths('out/', 0)             # {'sensor': Path(...), 'hits': ..., 'step': ..., 'labl': ...}
n = len(list_events(paths['sensor']))      # events in this batch

sensor = read_sensor_event(paths['sensor'], 0)   # one event -> dict of arrays
labels = read_labl_event(paths['labl'], 0)       # nested: per_event / per_interaction / per_particle / per_track
```

For a quick look from the terminal, `python scripts/inspect_dataset.py --path out/`
prints event counts, charge/hit statistics, and per-event array shapes
(`--event N` for one event in detail). The `viewer/` app (`python viewer/serve_viewer.py`)
gives an interactive 3D event display over the same files.

## Derived views

`lucid/production/derived_views/` post-processes finished datasets without rerunning the
simulation — e.g. splitting pile-up batches into single-interaction events (`depile`) and
recomputing label views (`relabel_*`). See its README for the available views.
