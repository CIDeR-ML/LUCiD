# LUCiD tutorials

Seven notebooks that build up from a single simulated event to full gradient-based
reconstruction and calibration. Read and run them **in order of interest**, not strictly top
to bottom — each is self-contained.

## Laptop-GPU mode vs. full mode

Every notebook imports a shared **run profile** (`tutorial_profile.py`) instead of hard-coding
photon counts, emission-grid resolution, and iteration budgets. This lets the *same* notebook
run two ways:

| Mode | For | How |
|------|-----|-----|
| **`laptop`** (default) | a **laptop GPU** — sized to fit in ≲ 3.5 GB of VRAM | just run it |
| **`full`** | full fidelity (the published-figure settings) on a **big GPU / cluster** | set `LUCID_TUTORIAL_MODE=full` |

```bash
jupyter lab                                   # laptop-GPU mode (default)
LUCID_TUTORIAL_MODE=full jupyter lab          # full fidelity (big GPU / cluster)
```

The first cell of every notebook prints the active profile, e.g.:

```
tutorial profile: mode=laptop  (laptop GPU, ~2.5 GB VRAM)
  photons: sim=80,000 cal=150,000 string=150,000  grid=80x120x80  K_water=6 K_ice=15
  sweeps: 1d=25 2d=15  recon: 2start x 80it  cal_fit=100 steps
```

**Only fidelity changes between modes** — the detector, physics, and narrative are identical.
Laptop mode has more shot noise and a looser fit (e.g. track reconstruction lands at tens of
cm rather than the few-cm of full mode); it is meant to show the machinery working, not to be
a precision run. One value is a **floor, not a knob**: `K_ice` stays at 15 in both modes,
because ice scatters strongly and a smaller `K` would silently drop a large fraction of the
scatter weight (a correctness bug, not a speed-up).

To change or add a mode, edit `tutorial_profile.py`.

## Expected time & peak VRAM (laptop mode)

Measured on an RTX 2080 Ti (11 GB). Times are wall-clock **including one-time JIT compilation**
— which is the bulk of the cost for the lighter notebooks; re-running cells in a live kernel is
much faster. All fit comfortably on a laptop GPU.

| Notebook | Time | Peak VRAM |
|----------|------|-----------|
| `00_quickstart` | ~50 s | 1.3 GB |
| `data_vs_prediction` | ~70 s | 1.3 GB |
| `event_displays` | ~6 min | 3.3 GB |
| `calibration_optimization` | ~2 min | 1.3 GB |
| `calibration_gradients` | ~3 min | 2.3 GB |
| `track_gradients` | ~4 min | 1.3 GB |
| `track_optimization` | ~9 min | 2.5 GB |

> `track_optimization` reconstructs in **two** detectors (SK-like + JUNO). Its first cell sets
> JAX's *platform* allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`) so freed VRAM is returned
> between them — without it the default pool accumulates to ~8.5 GB.

## Before you start

The tutorials need the example SIREN emitter weights + photon tables (~2.7 GB, not in git):

```bash
./scripts/download_data.sh
```

`event_displays` additionally exercises the bundled WbLS and IceCube (ice) configs, which the
same script wires up.

## The notebooks

| Notebook | What it shows |
|----------|---------------|
| `00_quickstart` | Simulate one muon in an SK-like tank and display the per-PMT charge ring. |
| `event_displays` | Event displays across geometries — water, WbLS, and an IceCube string in ice. |
| `data_vs_prediction` | Compare a "data" event (realistic hit-making) against the differentiable prediction. |
| `track_gradients` | The gradients of the forward model w.r.t. track parameters — the levers reconstruction pulls. |
| `track_optimization` | Full gradient-based track reconstruction (grid seed → Gauss-Newton), SK-like and JUNO. |
| `calibration_gradients` | Gradients w.r.t. optical parameters, plus the Fisher/Cramér-Rao bound. |
| `calibration_optimization` | Recover optical parameters by gradient descent from calibration light. |
