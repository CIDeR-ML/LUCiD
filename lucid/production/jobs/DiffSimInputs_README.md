# DiffSim Input Generation

Configurations for generating PhotonSim data used as input for LUCiD. These configs produce raw photon data to generate data-like events in LUCiD, or train PhysicsSIREN in LUCiD.

> **SIREN training inputs live elsewhere now.** This page covers the
> data-like-event configs only. For the SIREN pipeline (s/s_max axis, up to
> 100 GeV) use:
>
> - Stage 0 — `s_max(E)` fit scan: [`smax/README.md`](smax/README.md)
> - Stage 1 — per-cell `photonsim.root` with `PhotonHist_AngleDistanceNorm`: [`siren_inputs/README.md`](siren_inputs/README.md)
> - Stage 2/3 — `.h5` build + training: [`../../../docs/guides/production/siren-training-inputs.md`](../../../docs/guides/production/siren-training-inputs.md)
>
> The `water_lookup_table_*.json` configs below are the legacy pre-`s_max`
> path (10–2000 MeV, absolute distance axis). Kept for non-SIREN reuse; do
> not use them to train new SIREN models.

## Quick Start

```bash
# 1. Pull the unified container (one-time)
apptainer pull /sdf/data/neutrino/<user>/software/images/lucid.sif \
    docker://ghcr.io/cider-ml/lucid:latest

# 2. Configure your paths
cp user_paths.sh.template user_paths.sh
vim user_paths.sh   # set LUCID_IMAGE_PATH, OUTPUT_BASE_PATH, SLURM_*
```

PhotonSim ships pre-built inside the container — no host-side build
needed. See `DataProduction_README.md` for the full configuration
schema.

## Usage

```bash
# Test mode (1 job, 1 energy point)
./dataprod/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json -t

# Generate all jobs (prepare only)
./dataprod/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json

# Generate and submit
./dataprod/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json -s
```


## Available Configs

Located in `macros/diffsim_input/`:

| Config | Particle | Energy | Output |
|--------|----------|--------|--------|
| `water_lookup_table_mu.json` | mu- | 100-2000 MeV (10 MeV steps) | Averaged photon data |
| `water_lookup_table_el.json` | e- | 100-2000 MeV (10 MeV steps) | Averaged photon data |
| `photonsim_single_neg_mu_monoenergetic_for_various_energies.json` | mu- | 200-2000 MeV (50 MeV steps) | Individual photons |
| `photonsim_single_neg_mu_monoenergetic.json` | mu- | 1050 MeV | Individual photons |
| `photonsim_single_neg_mu_uniform.json` | mu- | 210-1500 MeV uniform | Individual photons |

The "lookup" configurations provide inputs to train PhysicsSIREN.
The "photonsim" configurations provide inputs to generate data-like events in LUCiD.
All configs have `disable_decays: true`.

## Output Structure

Output path depends of your user_paths.sh configuration, and goes to:

```
$OUTPUT_BASE_PATH/water/<output_path>/<energy>MeV/
```

For example, `water_lookup_table_mu.json` outputs to:
```
$OUTPUT_BASE_PATH/water/monoenergetic/averaged/mu-/100MeV/
$OUTPUT_BASE_PATH/water/monoenergetic/averaged/mu-/110MeV/
...
$OUTPUT_BASE_PATH/water/monoenergetic/averaged/mu-/2000MeV/
```

See `DataProduction_README.md` for additional documentation of the job system in s3df.

## Next step

For SIREN inputs, use the new s/s_max pipeline ([`smax/`](smax/) +
[`siren_inputs/`](siren_inputs/) + [`docs/guides/production/siren-training-inputs.md`](../../../docs/guides/production/siren-training-inputs.md));
the configs documented above are for non-SIREN data-like-event generation.