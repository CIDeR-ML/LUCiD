# DiffSim Input Generation

Configurations for generating PhotonSim data used as input for LUCiD. These configs produce raw photon data to generate data-like events in LUCiD, or train PhysicsSIREN in LUCiD.

## Quick Start

```bash
# 1. Configure your paths
cp user_paths.sh.template user_paths.sh
vim user_paths.sh  # Edit with your paths

# 2. Build PhotonSim
./utils/build_photonsim.sh

## Usage

```bash
# Test mode (1 job, 1 energy point)
./jobs/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json -t

# Generate all jobs (prepare only)
./jobs/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json

# Generate and submit
./jobs/generate_jobs.sh -c ../macros/diffsim_input/water_lookup_table_mu.json -s
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