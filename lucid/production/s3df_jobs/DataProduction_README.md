# S3DF Scripts for PhotonSim

Scripts for building, running, and managing PhotonSim jobs on the S3DF cluster at SLAC.

## Quick Start

```bash
# 1. Configure your paths
cp user_paths.sh.template user_paths.sh
vim user_paths.sh  # Edit with your paths

# 2. Build PhotonSim
./utils/build_photonsim.sh

# 3. Test with one job (doesn't submit)
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_single_neg_mu.json -t

# 4. Generate and submit all jobs
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_single_neg_mu.json -s

# 5. Monitor your jobs
./jobs/monitor_jobs.sh -w
```

## Directory Structure

```
s3df_scripts/
├── user_paths.sh.template     # Configuration template (copy to user_paths.sh)
├── utils/                     # Build and environment utilities
│   ├── setup_environment.sh   # Environment setup
│   ├── build_photonsim.sh     # Build script
│   ├── clean_build.sh         # Clean build directory
│   └── check_installation.sh  # Verify installation
└── jobs/                      # Job submission and management
    ├── generate_jobs.sh       # Generate jobs for a single config
    ├── submit_all_configs.sh  # Submit jobs for all configs at once
    ├── monitor_jobs.sh        # Monitor running jobs
    ├── cleanup_jobs.sh        # Clean job outputs
    └── report_time_performance.py  # Generate timing statistics and plots
```

## Configuration

### user_paths.sh

Copy `user_paths.sh.template` to `user_paths.sh` and configure:

```bash
# Software installations
export GEANT4_INSTALL_DIR="/path/to/geant4/install"
export ROOT_INSTALL_DIR="/path/to/root/install"

# Output paths
export OUTPUT_BASE_PATH="/path/to/photonsim/output"
export LUCID_PATH="/path/to/LUCiD"

# SLURM configuration (use GPU partition for LUCiD processing)
export SLURM_PARTITION="ampere"
export SLURM_ACCOUNT="your-account:your-group"

# Resource defaults (GPU-enabled for LUCiD)
export DEFAULT_CPUS="1"
export DEFAULT_MEMORY="39936"  # Memory in MB for GPU nodes
export DEFAULT_GPUS="1"        # Number of GPUs per job
export DEFAULT_TIME="23:00:00"
```

On S3DF, you can use existing installations:
```bash
export GEANT4_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/geant4"
export ROOT_INSTALL_DIR="/sdf/data/neutrino/cjesus/software/builds/root"
```

## Job Generation

All job generation uses JSON configuration files in `macros/data_production_config/`.

### generate_jobs.sh

```bash
./jobs/generate_jobs.sh -c <config_json> [-s] [-t]
```

**Options:**
- `-c`: Path to JSON configuration file (required)
- `-s`: Submit jobs to SLURM (default: prepare only)
- `-t`: Test mode - create only one job

### JSON Configuration Format

```json
{
  "config_number": 1,
  "name": "single mu-",
  "description": "Uniform energy muons with LUCiD processing",
  "material": "water",
  "output_path": "uniform_energy",
  "energy_distribution": "uniform",
  "store_individual_photons": true,
  "run_lucid": true,
  "disable_decays": false,
  "particles": [
    {
      "type": "mu-",
      "energy_min_MeV": 210,
      "energy_max_MeV": 1500
    }
  ],
  "lucid_options": {
    "apply_smearing": true,
    "apply_translation": true
  },
  "n_jobs": 100,
  "n_events_per_job": 1000
}
```

### Configuration Fields

| Field                 | Type    | Description                                       |
|-----------------------|---------|---------------------------------------------------|
| `config_number`       | integer | Unique ID (used in output folder: `config_XXXXXX`)|
| `name`                | string  | Human-readable dataset label; passed to LUCiD as `--dataset-name` and recorded in HDF5 provenance |
| `material`            | string  | "water" (other materials coming soon)             |
| `output_path`         | string  | Subdirectory for output                           |
| `energy_distribution` | string  | "uniform" or "monoenergetic"                      |
| `particles`           | array   | List of particles with energy ranges              |
| `disable_decays`      | boolean | Disable decay processes (for lookup tables)       |
| `lucid_options`       | object  | LUCiD processing flags (see below)                |
| `n_jobs`              | integer | Number of SLURM jobs                              |
| `n_events_per_job`    | integer | Events per job                                    |

#### LUCiD Options

| Field                   | Type    | Default | Description                                                   |
|-------------------------|---------|---------|---------------------------------------------------------------|
| `apply_smearing`        | boolean | true    | Apply PMT charge and timing smearing                          |
| `apply_translation`     | boolean | true    | Apply random translation to place vertex within detector      |

### Multi-Particle Events

For events with multiple primaries, specify per-particle energy ranges:

```json
{
  "particles": [
    {"type": "mu-", "energy_min_MeV": 105, "energy_max_MeV": 1500},
    {"type": "pi+", "energy_min_MeV": 122, "energy_max_MeV": 1500}
  ]
}
```

### submit_all_configs.sh

Submit jobs for multiple configurations at once with custom settings.

```bash
./jobs/submit_all_configs.sh [-p pattern] [-s] [-t] [-d] [-n n_jobs] [-e events] [-g] [-P partition] [-o output_base]
```

**Options:**
| Option | Description |
|--------|-------------|
| `-p` | Pattern to match config files (default: `dataprod*.json`) |
| `-s` | Submit jobs to SLURM (default: prepare only) |
| `-t` | Test mode - create only one job per config |
| `-d` | Dry run - show what would be submitted without doing it |
| `-n` | Override number of jobs per config |
| `-e` | Override events per job |
| `-g` | Enable GPU mode (request 1 GPU per job) |
| `-P` | SLURM partition override (e.g., `ampere`, `roma`, `milano`) |
| `-o` | Output base path override |

**Note:** When using `-s` to submit jobs, the script will prompt for confirmation before submitting. Pipe `echo "y"` to bypass:
```bash
echo "y" | ./jobs/submit_all_configs.sh -n 10 -e 100 -s
```

**Examples:**

```bash
# Dry run to see all configs that would be processed
./jobs/submit_all_configs.sh -d

# Submit all configs with GPU mode on ampere partition
./jobs/submit_all_configs.sh -n 10 -e 100 -g -P ampere -s

# Submit all configs on CPU partition (roma)
./jobs/submit_all_configs.sh -n 10 -e 100 -P roma -s

# Submit to custom output directory
./jobs/submit_all_configs.sh -n 5 -e 50 -P milano -o /path/to/custom/output -s

# Test all configs (1 job each, no submission)
./jobs/submit_all_configs.sh -t
```

## Available Configurations

Pre-defined configurations in `macros/data_production_config/`:

| Config # | File | Dataset name | Energy Range |
|----------|------|--------------|--------------|
| 1 | `dataprod_01_mu.json` | `single mu-` | 210-1500 MeV |
| 2 | `dataprod_02_pi_plus.json` | `single pi+` | 210-1500 MeV |
| 3 | `dataprod_03_e.json` | `single e-` | 10-1500 MeV |
| 4 | `dataprod_04_pi_minus.json` | `single pi-` | 100-1500 MeV |
| 5 | `dataprod_05_pi0.json` | `single pi0` | 100-1500 MeV |
| 6 | `dataprod_06_e_low_energy.json` | `single low energy e-` | 1-20 MeV |
| 7 | `dataprod_07_mu_pi_plus.json` | `mu- + pi+` | 105-1500 MeV |
| 8 | `dataprod_08_e_pi_plus.json` | `e- + pi+` | 100-1500 MeV |
| 9 | `dataprod_09_e_pi0.json` | `e- + pi0` | 100-1500 MeV |
| 10 | `dataprod_10_mu_pi_plus_pi0.json` | `mu- + pi+ + pi0` | 100-1500 MeV |
| 11 | `dataprod_11_mu_pi_plus_pi_minus.json` | `mu- + pi+ + pi-` | 100-1500 MeV |
| 12 | `dataprod_12_e_pi_plus_pi0.json` | `e- + pi+ + pi0` | 100-1500 MeV |

## Output Structure

Each `config_XXXXXX/` directory is one LUCiD **dataset**. Every job in that
config contributes one **batch** (`file_index = job_id - 1`) to the dataset,
producing four parallel HDF5 files spread across the `sensor/`, `inst/`,
`seg/`, and `labl/` subdirectories.

```
OUTPUT_BASE_PATH/
└── water/
    └── uniform_energy/
        ├── config_000001/                           # dataset: "single mu-"
        │   ├── job_000001.mac                       # PhotonSim macro for job 1
        │   ├── run_job_000001.sh
        │   ├── submit_job_000001.sbatch
        │   ├── output_job_000001.root               # (deleted if cleanup_root_files: true)
        │   ├── sensor/wc_sensor_0000.h5             # ─┐
        │   ├── inst/wc_inst_0000.h5                 #  ├── batch 0 (job 1)
        │   ├── seg/wc_seg_0000.h5                   #  │
        │   ├── labl/wc_labl_0000.h5                 # ─┘
        │   ├── sensor/wc_sensor_0001.h5             # ─┐ batch 1 (job 2)
        │   └── ...                                  #  │
        ├── config_000002/                           # dataset: "single pi+"
        ├── ...
        └── config_000012/                           # dataset: "e- + pi+ + pi0"
```

### LUCiD v3 Dataset Schema

The four files per batch are the v3 production format defined in
[`LUCiD/docs/LUCID_DATASET.md`](../../LUCiD/docs/LUCID_DATASET.md). Summary:

| File | Role |
|------|------|
| `sensor/wc_sensor_NNNN.h5` | Raw PMT readout (post-smearing): `PE`, `T`, `sensor_idx` per event. |
| `inst/wc_inst_NNNN.h5`     | Per-particle decomposition of the PMT signal (pre-smearing truth). |
| `seg/wc_seg_NNNN.h5`       | Geant4 3D track segments (`start_*`, `end_*`, `dir_*`, `edep`, `time`). |
| `labl/wc_labl_NNNN.h5`     | Labels and truth metadata (categories, genealogy, containment, `t0`). |

Each file's `config/` group carries the provenance block (`dataset_name`,
`run_id`, `file_index`, `source_file`, `lucid_master_seed`, ...). See
[`LUCID_DATASET.md`](../../LUCiD/docs/LUCID_DATASET.md) for the full
schema (field names, shapes, dtypes).

### Particle Categories

Photons are grouped into categorized particles (stored in `labl/`) based on
the track that produced them:

| Category | Code | Condition | Description |
|----------|------|-----------|-------------|
| Primary | 0 | `parentID == 0` | Primary particle from the gun |
| DecayElectron | 1 | e± from μ/π decay, KE > 1 MeV | Michel electrons from muon decay; electrons from pion decay |
| Gamma | 2 | γ from π⁰ decay | Electromagnetic showers from neutral pion decay |
| SecondaryPion | 3 | π± from inelastic scatter or deflection > 5°, p ≥ 195 MeV/c | Charged pions from hadronic interactions or large-angle elastic scatters |

### Segment merging (PhotonSim side)

Track segments in `seg/` are produced by PhotonSim and merged before
writing to keep storage bounded while preserving trajectory detail:

| Track Energy | Merge Condition |
|--------------|-----------------|
| ≥ 10 MeV | Save when length ≥ 10mm OR direction change > 2° |
| < 10 MeV | Save when cumulative Edep ≥ 1 MeV |

## Job Management

### Monitor Jobs

```bash
./jobs/monitor_jobs.sh [-a] [-w] [-o output_dir]
```

- `-a`: Show all jobs (default: only PhotonSim jobs)
- `-w`: Watch mode - refresh every 30 seconds
- `-o`: Check specific output directory

### Cleanup Jobs

```bash
./jobs/cleanup_jobs.sh -o <output_dir> [-l] [-m] [-s] [-a]
```

- `-l`: Clean log files (*.out, *.err)
- `-m`: Clean macro files (*.mac)
- `-s`: Clean script files (*.sh, *.sbatch)
- `-a`: Clean all temporary files

**Note**: Runs in dry-run mode by default. ROOT and HDF5 files are always preserved.

### Performance Timing Report

Generate statistics and histograms of LUCiD per-event processing times from job outputs.

```bash
python ./jobs/report_time_performance.py --all --base-dir <path_to_configs> [--output report.png]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--config-dir` | Analyze a specific config directory |
| `--all` | Analyze all configs in base directory |
| `--base-dir` | Base directory containing `config_XXXXXX` folders |
| `--output` | Output file for histogram (PNG) |
| `--no-plot` | Skip plot generation, only print statistics |

**Examples:**

```bash
# Analyze all configs and generate histogram
python ./jobs/report_time_performance.py --all \
  --base-dir /path/to/water/uniform_energy \
  --output timing_report.png

# Analyze a single config
python ./jobs/report_time_performance.py \
  --config-dir /path/to/water/uniform_energy/config_000001

# Print statistics only (no plot)
python ./jobs/report_time_performance.py --all \
  --base-dir /path/to/water/uniform_energy \
  --no-plot
```

The script parses SLURM job output files to extract per-event LUCiD processing times and generates:
- Per-config statistics (min, max, mean, median, percentiles)
- Summary table comparing all configurations
- Histogram plots showing time distributions

### LUCiD Processing Time Benchmarks

Median LUCiD processing times (seconds/event) across different S3DF partitions. PhotonSim adds ~2s per event regardless of partition.

| Config | Description    | Ampere (GPU) | Roma (CPU) | Milano (CPU) |
|--------|----------------|--------------|------------|--------------|
| 000001 | muon           | 0.51s        | 6.79s      | 9.04s        |
| 000002 | charged pion   | 0.28s        | 7.54s      | 8.93s        |
| 000003 | electron       | 0.46s        | 2.26s      | 5.94s        |
| 000004 | mu + pi mixed  | 0.83s        | 26.22s     | 21.59s       |
| 000005 | e + pi mixed   | 0.74s        | 16.31s     | 19.14s       |
| 000006 | neg pion       | 0.26s        | 5.70s      | 10.12s       |
| 000007 | low-e electron | 0.06s        | 0.06s      | 0.08s        |
| 000008 | mu + 2pi mixed | 1.17s        | 34.24s     | 26.69s       |
| 000009 | mu + pi+ + pi- | 1.06s        | 32.71s     | 28.65s       |

## Utilities

### Build PhotonSim

```bash
./utils/build_photonsim.sh
```

### Check Installation

```bash
./utils/check_installation.sh
```

### Clean Build

```bash
./utils/clean_build.sh
```

## Example Workflows

### 1. Single Particle Dataset

```bash
# Generate muon dataset with uniform energy 210-1500 MeV
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_single_neg_mu.json -s
```

### 2. Multi-Particle Dataset

```bash
# Generate mu- + pi+ mixed events
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_neg_mu_pos_pion.json -s
```

### 3. Test Before Production

```bash
# Test mode creates only one job
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_single_neg_mu.json -t

# Check the generated files
ls -la $OUTPUT_BASE_PATH/water/uniform_energy/config_000001/

# If looks good, submit all
./jobs/generate_jobs.sh -c ../macros/data_production_config/dataprod_single_neg_mu.json -s
```

## Troubleshooting

### Environment Issues

1. Check `user_paths.sh` configuration
2. Verify GEANT4 and ROOT installations
3. Run `./utils/check_installation.sh`

### Build Failures

1. Clean build: `./utils/clean_build.sh`
2. Check environment: `source ./utils/setup_environment.sh`
3. Rebuild: `./utils/build_photonsim.sh`

### Job Failures

1. Check SLURM logs: `cat config_XXXXXX/job_*-*.out`
2. Check error logs: `cat config_XXXXXX/job_*-*.err`
3. Test locally by running the generated `run_job_*.sh` script

## Output Validation with LUCiD Visualization

> **Deprecated / in transition.** The current `visualize_particle_events.py`
> viewer is being replaced by a new viewer and may be broken against v3
> output. The usage example below is kept for reference only; update this
> section once the replacement viewer lands.

After jobs complete, you can generate interactive HTML visualizations to validate the output using LUCiD's `visualize_particle_events.py` script.

### Generate Validation HTML

```bash
# Using singularity on S3DF
singularity exec -B /sdf,/fs,/sdf/scratch,/lscratch \
  /sdf/group/neutrino/images/develop.sif python \
  $LUCID_PATH/lucid/production/visualize_particle_events.py \
  <hdf5_file> \
  $LUCID_PATH/config/SK_geom_config.json \
  --event <event_index> \
  --output-dir <output_directory>
```

### Example

```bash
# Visualize event 0 from config_000001 job 1
singularity exec -B /sdf,/fs,/sdf/scratch,/lscratch \
  /sdf/group/neutrino/images/develop.sif python \
  $LUCID_PATH/lucid/production/visualize_particle_events.py \
  $OUTPUT_BASE_PATH/water/uniform_energy/config_000001/events_job_000001.h5 \
  $LUCID_PATH/config/SK_geom_config.json \
  --event 0 \
  --output-dir $LUCID_PATH/validation_html
```

### Options

| Option | Description |
|--------|-------------|
| `--event N` | Event index to visualize (default: 0) |
| `--min-charge X` | Minimum charge threshold in PE (default: 1.0) |
| `--output-dir DIR` | Output directory for HTML file (default: current directory) |

### Output

The script generates an interactive HTML file with:
- 3D visualization of sensor hits colored by charge or categorized particle
- Slider to switch between views (Track Segments, By Particle, All, individual particles)
- Event genealogy information showing categorized particle hierarchy
- Light containment metrics per particle

The HTML file can be opened in any web browser for interactive exploration.

### Automated Validation Script

For quick local validation without SLURM jobs:

```bash
$LUCID_PATH/lucid/production/generate_validation_htmls.sh [-c CONFIG] [-n EVENTS] [-o OUTPUT]
```

- `-c`: Dataprod config number (default: 07)
- `-n`: Number of events (default: 5)
- `-o`: Output path (default: `validation_html/track_segments_test`)

Output goes to `<output>/config<NUM>/`. Includes "Track Segments" slider showing particle trajectories.
