# Quickstart

This page assumes LUCiD and the example data are [installed](install.md). Run the examples
from the repository root.

## Simulate one event

```bash
python examples/hello_simulate.py
```

The script, in full:

```python
--8<-- "examples/hello_simulate.py"
```

- `setup_event_simulator` reads the detector's geometry and physics configs and returns a
  `Simulator`, a callable wrapping the compiled forward model. With
  `default_detector_params=True` the detector parameters are loaded from the physics config and
  fixed, so the call takes `(track, key)`. Leave it `False` to pass (and differentiate with
  respect to) the detector parameters yourself, as calibration does.
- `ParticleParams.from_cartesian` builds the track: energy in MeV, start position in metres, a
  direction vector and a start time in ns.
- For a particle the call returns a `PhotonList`; `.charge` is the expected charge per PMT in
  photoelectrons, shape `(n_sensors,)` (10764 for `SK_like`).
- `K=4` is the number of scattering iterations. The [simulator reference](../reference/simulator.md)
  lists every argument and what the call returns.

## Differentiate it

The forward model is a JAX function, so `jax.grad`, `jax.vmap` and `jax.jit` apply to it.
`examples/hello_gradient.py` takes the gradient of the total expected charge with respect to
the energy, the gradient of a Poisson loss with respect to the vertex, and maps the forward over
a batch of energies (about a minute on a CPU):

```python
--8<-- "examples/hello_gradient.py:gradient"
```

## The other examples

| script | shows | time |
|---|---|---|
| `hello_calibrate.py` | seven optical parameters and 10764 per-PMT gains recovered from laser and flasher light, with the Cramér-Rao bound ([Calibration](../guides/calibration.md)) | ~4 min on a GPU |
| `hello_calibrate_own_optimiser.py` | one calibration problem solved by Gauss-Newton, Adam, a hand-written loop and scipy | GPU |
| `seed_reconstruct.py` | the published reconstruction of a muon or electron track ([Reconstruction](../guides/reconstruction.md)) | ~8 min on an 11 GB GPU |
| `hello_telescope.py` | a muon track and a cascade in the IceCube string array, in ice | CPU |
| `hello_multiparticle.py` | two GEANT4 muons from one vertex, drawn together and per particle (`--split`) | CPU |

Each script prints its options with `--help`. The calibration and reconstruction examples are
much slower on a CPU; each has a small-size switch (`--n-photons`, `--steps`, `--quick`).

## If something goes wrong

- `FileNotFoundError` under `data/`: the example data is not installed; run
  `python scripts/download_data.py`.
- Zero total charge: the track sends no light at a sensor. Check that the vertex is inside the
  detector and that positions are in metres.
- The first call is slow: it compiles (tens of seconds on a CPU). Later calls with the same
  shapes are fast. A new sensor radius or temperature also builds an overlap table once
  ([install](install.md#check-the-install)).
- JAX probes a GPU you do not want: set `JAX_PLATFORMS=cpu`.
- Numbers differ from a colleague's: fix the PRNG key and the photon count. Bit-identical
  output also needs the same CPU type and thread count, because float32 reductions depend on
  both.
- No window opens: the display is written to `hello_simulate.png` (`--out` for another name).
