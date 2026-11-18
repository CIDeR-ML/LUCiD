# Tests

```bash
pytest tests/ -q                          # fast tests (the default deselects `slow`)
pytest tests/ -q --slow                   # all tests except `gpu`
pytest tests/ -q -m slow                  # slow tests only
pytest tests/path/test_x.py::test_name    # one test
```

`tests/conftest.py` sets `JAX_PLATFORMS=cpu` for the whole suite. Most physics tests run on a
small detector (`helpers/tiny.py`: a 1 m radius water cylinder with 582 PMTs) whose simulators
are built once per session. `physics/test_forward_golden.py` compares the forward model and its
gradient on that detector with a stored reference; `LUCID_CAPTURE_GOLDEN=1` re-captures it.

## Markers

| marker | meaning |
|---|---|
| `slow` | an SK-scale detector, an extra simulator compile or a ROOT file; deselected by default |
| `gpu` | needs a GPU; deselected unless `LUCID_GATE_GPU=1`, even with `--slow` |
| `golden` | compares with a stored reference |
| `requires_data(*paths)` | needs these files under `data/`; a tuple means any one of them |
| `requires_emitter(particle, material, dedx=False)` | needs the trained SIREN emitter the simulator would load for that material |

A test without its data is skipped with a reason that names the missing files, and the run ends
with a list of them. `python scripts/download_data.py` installs the data;
`LUCID_REQUIRE_DATA=1` turns those skips into failures. Declare data with these markers rather
than with `skipif` or `pytest.skip`:

```python
@pytest.mark.requires_data('water/muon/1000MeV_100events.root')
@pytest.mark.requires_emitter('muon', 'wbls', dedx=True)
```

```bash
LUCID_GATE_GPU=1 pytest --slow tests/paper/test_published_resolution.py
```

## Layout

Tests are grouped by subject, not by cost; cost is a marker.

| directory | subject |
|---|---|
| `physics/` | the forward model: emission, optics, photon step, reflection, wavelength-dependent media, sensor response, the composed simulator, detector and physics configs |
| `propagation/` | detector geometry, ray-surface intersection, sensor lookup, overlap tables, propagators and their gradients |
| `siren/` | the SIREN emitters: loading, s_max(E), lookup tables, training, ray sampling |
| `fitting/` | losses, Gauss-Newton and `minimize`, calibration, timing, reconstruction, fit analysis |
| `production/` | digitizer, trigger, event generation, the PhotonSim reader, the dataset format, the dataprod jobs and cluster adapters |
| `paper/` | the paper's figure entry points, study configs, and the published-resolution check |
| `string/` | neutrino-telescope string detectors |
| top level | imports, shell-script safety, `scripts/download_data.py` |
| `helpers/` | shared fixtures and toy models (no tests); `helpers/paths.py` gives `REPO`, `CONFIG`, `DATA` |

A fixture goes in the `conftest.py` of the subject that uses it, or in `tests/conftest.py` when
several subjects do. More on development and CI is in [development](../docs/development.md).
