# t0 fit

The simulator's photon emission time, `t(d, E) = d/c + A(E)(exp((d/λ(E))^β(E)) − 1)`, is fitted
by PhotonSim's `tools/t0_correction/calculate_t0.py` from `PhotonHist_TimeDistanceNorm`, the
photon delay against `s/s_max`. PhotonSim fills that histogram whenever the macro sets
`/output/smax`, which every training-input cell does ([`../README.md`](../README.md)). So the t0
inputs are the training-input scan's `photonsim.root` files, one per energy, and this stage runs
no PhotonSim of its own: the shipped water fits were made this way, from the scans of
`../configs/water_mu.json` with `water_mu_lowE.json` (mu-, 307 energies from 150 MeV) and
`water_el.json` (e-, 339 energies; the fit used 329, from 11 MeV: it skips an energy with fewer
than 4 profile points beyond 50 mm).

`calculate_t0.py` fits `A`, `λ` and `β` at each energy, then each of `log10 A`, `log10 λ` and `β`
as a cubic in `log10 E`, and writes `timing_parameters.json` (form `stretched_exp_delay`).
`lucid-siren-promote --t0` copies its cubics into the emitter's `t0` block. It needs the PhotonSim
checkout in `PHOTONSIM_DEV_PATH` (its `calculate_t0.py`, from May 2026, and its
`data/<material>/<particle>/smax_fit.csv`).

```bash
RUN=../../platform/run_in_container.sh

# prepare (links the inputs, writes the job), then submit
$RUN lucid.production.siren_inputs.t0.generate_jobs --material water --particle mu-
$RUN lucid.production.siren_inputs.t0.generate_jobs --material water --particle mu- -s
```

| | |
|---|---|
| inputs | `<SIREN_INPUTS_BASE_PATH or SIREN_OUTPUT_BASE_PATH>/training_inputs/<m>/<p>/<E>MeV/photonsim.root` (`--inputs`), `$PHOTONSIM_DEV_PATH/data/<m>/<p>/smax_fit.csv` |
| outputs | `<SIREN_OUTPUT_BASE_PATH>/t0_parametrization/<m>/<p>/` (`--output`): `inputs/<p>_<E>MeV.root` (links), `timing_parameters.json`, `pred_vs_data.png`, `pred_vs_data_examples.png`, `trends.png`, `submit.sbatch` and its logs |
| job | one CPU job on `SLURM_PARTITION` (or `CONDOR_JOB_FLAVOUR`), in the container |

A finished fit (`timing_parameters.json` present) is not redone unless `--no-skip-existing`.
Check `trends.png` and the per-energy RMS in the job log before promoting.
