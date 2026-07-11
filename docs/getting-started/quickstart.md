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
print((charge > 0).sum(), 'PMTs lit,', charge.sum(), 'pe')   # pe = photoelectrons, LUCiD's charge unit

create_detector_display('config/SK_like_geom_config.json', sparse=False)(charge, time)   # 2D unrolled ring
```

!!! note "`hit_mode` and `K` here"
    This synthetic-track call is a **2-argument** callable — `sim(track, key)` returning
    `(charge, time)` — because `default_detector_params=True` bakes the detector params in.
    Its `hit_mode='realistic'` is valid track-mode usage and is *distinct* from **data mode**
    (`is_data=True`), whose `realistic` row in the [pipeline table](../concepts/photon-pipeline.md)
    is a 4-argument callable fed per-PMT hits from a ROOT file. `K=6` scatter iterations is
    display-quality; use `K≥8` for [reconstruction](../guides/reconstruction.md) and
    [calibration](../guides/calibration.md).

## Shapes, gradients, GPU

`sim` returns two arrays — `charge` and `time` — each of shape `(n_sensors,)` and dtype
`float32` (10764 for this SK-like tank: one entry per PMT, `0` where a PMT saw no light).
Because the whole forward is JAX, it differentiates and batches with no extra machinery:

```python
import jax.numpy as jnp

# exact autodiff gradient of total detected charge w.r.t. track energy — one line
def total_charge(E):
    return sim(track._replace(energy=E), jax.random.PRNGKey(0))[0].sum()
print(jax.grad(total_charge)(1000.))     # ≈ 3.0 pe/MeV (charge scales ~linearly with energy)

# batch a whole energy scan in a single call with vmap
energies = jnp.array([500., 1000., 1500.])
charges, times = jax.vmap(lambda E: sim(track._replace(energy=E), jax.random.PRNGKey(0)))(energies)
print(charges.shape)                     # (3, 10764)
```

!!! note "GPU"
    JAX runs this on a GPU automatically whenever one is visible — no code change. Set
    `JAX_PLATFORMS=cpu` to keep it on the CPU (reproducible CI, or a shared node).

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
`hello_calibrate`, `seed_reconstruct`, and `hello_telescope` for an IceCube-style
string detector). Browse the [bundled detectors](detectors.md) to see every geometry
these can run against.

Then read the [concepts](../concepts/photon-pipeline.md) to understand how the forward model works.

## If something goes wrong

- **`FileNotFoundError` under `data/`** — the SIREN weights or example events aren't
  downloaded yet; run `./scripts/download_data.sh` from the repo root.
- **Zero PMTs lit** — usually a geometry/units problem (position outside the tank,
  direction pointing at the wall from close range); try the defaults above first.
- **JAX device warnings or slow first call** — the first invocation JIT-compiles
  (tens of seconds on CPU is normal); subsequent calls are fast. Set
  `JAX_PLATFORMS=cpu` to keep JAX off the GPU entirely (the older
  `JAX_PLATFORM_NAME` alone does not stop the CUDA plugin from probing).
- **Different numbers than a colleague** — results depend on the PRNG key and photon
  count; fix both (`PRNGKey(0)`, same `n_photons`) for byte-comparable output.
- **No display appears** — on a headless machine (SSH, no X) the display call silently
  no-ops; pass `file_name='ring.png'` to the display call to save the figure instead.
