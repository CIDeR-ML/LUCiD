# Quickstart

After [installing](install.md) and running `./scripts/download_data.sh`:

## Simulate & display one event

```bash
python examples/hello_simulate.py
```

or, interactively, open `tutorials/00_quickstart.ipynb`. The essential calls:

```python
import jax, numpy as np
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.visualization import create_detector_display

# build a JIT-compiled forward simulator for an SK-like tank
sim = setup_event_simulator('config/SK_like_geom_config.json', 250_000, K=6,
                            hit_mode='realistic', physics_config='config/SK_like_physics_config.json',
                            default_detector_params=True, particle='muon')

# a 1 GeV muon from the centre, along +x
track = ParticleParams.from_cartesian(energy=1000., position=[0,0,0], direction=[1,0,0], t0=0.)
charge, time = (np.asarray(x) for x in sim(track, jax.random.PRNGKey(0)))
print((charge > 0).sum(), 'PMTs lit,', charge.sum(), 'pe')

create_detector_display('config/SK_like_geom_config.json')(charge, time)   # 2D unrolled ring
```

## Where to go next

The narrated notebooks in `tutorials/` walk the main workflows:

| Notebook | Workflow |
|----------|----------|
| `00_quickstart` | simulate + display |
| `track_optimization` | reconstruct a track (+ a sphere example) |
| `calibration_optimization` | calibrate optical parameters + quote uncertainties |
| `calibration_gradients` | calibration loss landscape, Hessian, before/after |
| `track_gradients` | reconstruction loss landscapes |
| `data_vs_prediction` | per-PMT likelihood: data vs model |
| `event_displays` | 2D / animation / 3D across geometries & materials |

Short copy-paste scripts live in `examples/` (`hello_simulate`, `hello_reconstruct`,
`hello_calibrate`, `seed_reconstruct`).

Then read the [concepts](../concepts/photon-pipeline.md) to understand how the forward model works.
