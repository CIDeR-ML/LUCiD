# Working with v3 data

`lucid-run-job` writes the **v3** dataset as **four parallel HDF5 files per batch**, sharing
per-event indexing:

| modality | file | contents |
|----------|------|----------|
| `sensor` | `wc_sensor_NNNN.h5` | raw per-PMT readout (post-smearing) |
| `hits`   | `wc_hits_NNNN.h5`   | per-particle decomposition of the PMT signal |
| `step`   | `wc_step_NNNN.h5`   | 3D GEANT4 track segments (the per-segment energy deposit is the `edep` column) |
| `labl`   | `wc_labl_NNNN.h5`   | labels, truth metadata, and dimension tables (truth `t0` lives here) |

Detector times in `sensor`/`hits`/`step` are in the **detector frame** (per-event `t0` already
added); truth `t0` is in `labl`. The full schema is in the [v3 dataset reference](../LUCID_DATASET.md).

> Note: on `main` the segment modality is named `step` (the `edep` value is a column within it).
> Older data may still use `edep` as the modality name.

## Reading a batch

The helpers in `lucid.production.data_prod_utils` read the four-file batch and return per-event
dictionaries; `lucid.sources.v3_reader` holds the low-level readers.

```python
from lucid.production.data_prod_utils import read_multi_event_file, load_event_v3, print_event_info

events = read_multi_event_file('out/')    # list of per-event dicts (loads the batch)
ev = events[0]                            # one event as a dict of arrays
print_event_info(ev)                      # kinematics + summary
# a single event without loading the whole batch:
# ev = load_event_v3('out/', 0)           # (dataset_root, event_idx)
```

The `read_production_output.ipynb` notebook under `lucid/production/notebooks/` is the worked
reference for inspecting a dataset (HDF5 structure, single/all events, PDG/PE aggregates,
per-PMT inspection).
