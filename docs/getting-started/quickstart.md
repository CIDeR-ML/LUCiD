# Quickstart

After [installing](install.md) and running `python scripts/download_data.py`:

## Simulate & display one event

```bash
python examples/hello_simulate.py
```

The whole script, included here from `examples/hello_simulate.py`:

```python
--8<-- "examples/hello_simulate.py"
```

- `setup_event_simulator` reads the detector's geometry and physics configs and returns a
  `Simulator`, a callable wrapping the JIT-compiled forward. `default_detector_params=True` bakes the detector parameters in, so the
  callable takes `(track, key)`; leave it `False` to differentiate with respect to them
  (calibration, [parameters](../concepts/detector-params-vs-args.md)).
- For a particle (`hits='photon_list'`, the particle default) the forward returns a
  `PhotonList(log_w, times, indices, charge)` NamedTuple: per-deposit log-weights, times and
  sensor indices, and `.charge`, the expected charge per PMT, shape `(n_sensors,)` — 10764 for this
  SK-like tank — in photoelectrons (pe). The [photon pipeline](../concepts/photon-pipeline.md)
  lists what each kind of hits returns.
- `K` is the number of scatter iterations; the
  [photon pipeline](../concepts/photon-pipeline.md#choosing-n_photons-k-temperature) says how to
  choose it.

## Differentiate it

The forward is JAX end to end, so `jax.grad`, `jax.vmap` and `jax.jit` apply to it directly.
`examples/hello_gradient.py` takes the gradient of the total expected charge with respect to the
energy and of a Poisson loss with respect to the vertex, and maps the forward over a batch of
energies (about a minute on a CPU):

```python
--8<-- "examples/hello_gradient.py:gradient"
```

`examples/hello_calibrate_own_optimiser.py` runs gradient-based fits on the forward.

!!! note "GPU"
    JAX runs this on a GPU automatically whenever one is visible — no code change. Set
    `JAX_PLATFORMS=cpu` to keep it on the CPU (reproducible CI, or a shared node).

## Where to go next

[`examples/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/examples/README.md) lists the
runnable scripts: reconstruction (`seed_reconstruct.py` runs the published method), calibration
(`hello_calibrate.py`), a neutrino-telescope string array (`hello_telescope.py`) and more. Browse
the [bundled detectors](detectors.md) to see every geometry they can run against, and read the
[concepts](../concepts/photon-pipeline.md) to understand how the forward model works.

## If something goes wrong

- **`FileNotFoundError` under `data/`** — the SIREN weights or example events aren't installed
  yet; run `python scripts/download_data.py` from the repo root.
- **Zero expected charge** — usually a geometry/units problem (position outside the tank,
  direction pointing at the wall from close range); try the defaults above first.
- **JAX device warnings or slow first call** — the first invocation JIT-compiles
  (tens of seconds on CPU is normal); subsequent calls are fast. A sensor radius or temperature whose PMT overlap table
  did not come with the example data builds it on first use (a minute or two; see
  [install](install.md#verify)). Set
  `JAX_PLATFORMS=cpu` to keep JAX off the GPU entirely (the older
  `JAX_PLATFORM_NAME` alone does not stop the CUDA plugin from probing).
- **Different numbers than a colleague** — results depend on the PRNG key and photon
  count; fix both (`PRNGKey(0)`, same `n_photons`). Byte-identical output also needs the same
  machine type and thread count: float32 reductions differ in the last digits across CPUs and
  XLA's reduction order follows the thread count.
- **No display window** — `hello_simulate.py` saves the display to `hello_simulate.png`
  (`--out` names another file). The display call opens a window only when it is given no
  `file_name` and a GUI backend is available; on a headless machine it shows nothing.
