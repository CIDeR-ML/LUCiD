# The test suite

## Running it

```bash
pytest tests/ -q          # the fast lane. `addopts = -m 'not slow'` deselects the rest
pytest tests/ -q --slow   # everything
pytest tests/ -m slow -q  # the slow lane only -- the same set --slow adds
pytest tests/path/test_x.py::test_name   # one test
```

There is **one** mechanism: the `slow` marker, assigned by cost (a detector at SK
scale, a simulator compile beyond the session's, a ROOT file). No basename list, and no
`pytest_ignore_collect`, so `-m slow` and `--slow` cannot disagree about what is slow.

The forward model itself runs in the fast lane, on the tiny detector (`helpers/tiny.py`, 582 PMTs):
`conftest.py` (`tiny_sim`) and `physics/conftest.py` (`tiny_sampled_sim`, `tiny_data_sim`) build
its simulators once per session, and `physics/test_forward_composed.py`,
`physics/test_incident_qe.py`, `physics/test_physics_hit_time.py` and
`physics/test_outside_emission.py` hold the composed chain to closed forms and a direct ray trace.
`physics/test_forward_golden.py` is the one stored reference (marker `golden`): the same forward at
a working point that uses every stage, with its gradient, at 1e-3.
`tests/conftest.py` forces `JAX_PLATFORMS=cpu` for the whole suite.

## Layout

One directory per subject. Cost is a marker (`slow`, `gpu`, `requires_data`, `requires_emitter`),
never a directory, so a new test goes where its subject is, whatever it costs.

| directory | what its tests hold to account |
|---|---|
| `physics/` | the forward model: emission (sources, scintillation, shotgun, data-mode events), optics, the photon step, reflection, the wavelength-dependent medium, the sensor response, the composed simulator (tiny detector and SK scale), the golden, and the detector/physics parameters and configs |
| `propagation/` | detector geometry, ray-surface intersection, the sensor grid and lookup, the surface deposit, the overlap table, the propagators' output and gradients |
| `siren/` | the trained SIREN emitter: its files and loader, s_max(E), the lookup tables, training, ray sampling, and the track forward it feeds |
| `fitting/` | the losses, the Gauss-Newton transformation and `minimize`, calibration, timing, reconstruction (model, seeding, fit keys), the result analysis and gradient sweeps |
| `production/` | the readout (digitizer, trigger), event generation and placement, the PhotonSim reader, the dataset reader/writer and labels, and the dataprod jobs and cluster adapters |
| `paper/` | the paper's figures and studies: entry points, study configs and files, the stored fit record, the loss landscape, the reconstruction chain end to end, and the published-resolution gate |
| `string/` | the deferred string (IceCube-like) line |
| top level | the repository itself: imports, shell-script safety, `scripts/download_data.py` |
| `helpers/` | shared toys and fixtures (the tiny detector, the analytic recon model, paths), no tests |

A fixture lives in the `conftest.py` of the one subject that uses it, or in `tests/conftest.py`
when several do.

## Data, GPU

A test that needs a downloaded asset says so with a marker, and `conftest.py` resolves it:

```python
@pytest.mark.requires_data('water/muon/1000MeV_100events.root')   # files under data/
@pytest.mark.requires_data(paths.MUON_1GEV_EVENTS)    # a tuple: any one (5-event slice or 100 events)
@pytest.mark.requires_emitter('muon', 'wbls', dedx=True)   # the files the simulator would load
```

`requires_emitter` follows the material JSON's `"siren"` set to
`data/<set>/<particle>/<cherenkov|dedx>/{emitter.json, weights.npz}` the way the simulator does,
so a guard cannot point at a stale path. Without the data the test skips with a reason that names
the missing files, and the end of the run lists them; `LUCID_REQUIRE_DATA=1` turns every such skip
into a failure, for a job that is supposed to have the fetched assets. There is no other skip
spelling for data: no `skipif(os.path.exists(...))`, no `pytest.skip` in a body.

`gpu` tests are deselected unless `LUCID_GATE_GPU=1` (`--slow` does not add them):

```bash
LUCID_GATE_GPU=1 pytest --slow tests/paper/test_published_resolution.py
```

Repository paths come from `helpers/paths.py` (`REPO`, `CONFIG`, `DATA`), never from a test
file's own depth.
