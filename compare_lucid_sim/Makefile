# 532 nm laser three-way comparison (LUCiD / JUNOsw / FSMP).
#
#   make help                     all targets and variables
#   make compare-all              per-run time/summary/amplitude PDFs
#   make tts-irf-all              TTS-vs-IRF pages (irf125 channels)
#   make pe-scale-all             all-channel PE-scale PDFs
#   make theta-intensity-all      per-PMT PE vs theta
#   make scale-vs-z               scale-vs-z summary
#   make all                      everything above
#
# Run from anywhere: make -C compare_lucid_sim <target>; prepend
# submit- to run as a Slurm job.  Paths default to the author's
# directories and outputs follow DATA_ROOT -- always override
# DATA_ROOT=... before running your own comparisons.
COMPARE_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(COMPARE_DIR)/..)
PYTHON       ?= /opt/gentoo/usr/bin/python3
MPLCONFIGDIR ?= /tmp/lucid-compare-make
export MPLCONFIGDIR

# ---- data paths (reproduction entry points) -----------------------
DATA_ROOT    ?= \
    /junofs/C/liukainan/LUCiD/simulation_CleanBasisModel
FSMP_ROOT    ?= /junofs/C/PE
JUNOSW_ROOT  ?= /mnt/stage/haoch/junores/laser/532
PMT_MAP      ?= /junofs/C/junoid_J25.7.3.pq
IRF_H5       ?= \
    /junofs/PmtCharacterization/IRF/jpwaptool_ser_baseline.h5
# Optional custom channel list for the TTS-vs-IRF check; by default
# the characterized channels are derived from IRF_H5 and PMT_MAP.
CHANNEL_SELECTION ?=
CHANNEL_SEL_ARG := \
    $(if $(CHANNEL_SELECTION),--channel-selection $(CHANNEL_SELECTION),)

# ---- comparison settings (keep in sync with published outputs) ----
CHANNEL_COUNT ?= 100
# Subsample for the per-channel amplitude/time pages; empty draws
# every selected channel.
PAGE_CHANNEL_COUNT ?=
PAGE_COUNT_ARG := \
    $(if $(PAGE_CHANNEL_COUNT),\
        --page-channel-count $(PAGE_CHANNEL_COUNT),)
SEED          ?= 0
BINS          ?= 600
FSMP_PE_SCALE ?= 10
QE            ?= 0.097
LUCID_LABEL   ?= LUCiD (diffuser)
JUNOSW_LABEL  ?= JUNOsw (J26.5.0)
TAG           ?= tts_dark_on_irf_off
PE_SCALE_BINS ?= 100
args          ?=

# ---- output roots (caches live beside the PDFs) --------------------
DIFFUSER_OUT ?= $(DATA_ROOT)/compare_sim
IRF125_OUT   ?= $(DATA_ROOT)/compare_sim-irf125
PE_SCALE_OUT ?= $(DATA_ROOT)/compare_sim-pe_scale
SCALE_Z_OUT  ?= $(DATA_ROOT)/compare_sim-scale_vs_z_all_channels

# ---- Slurm submission (submit-* targets) -------------------------
# SBATCH defaults to the project submission account; members
# override with SBATCH=sbatch.  Logs follow the data root.
SBATCH        ?= sudo -u\#35905 sbatch
SBATCH_OPTS   ?= --partition=gpu --gres=gpu:mi100:1 \
    --cpus-per-task=8 --mem=64G --time=24:00:00
SLURM_LOG_DIR ?= $(DATA_ROOT)/slurm-compare

# ---- run mapping: run -> z [m], LUCiD dir, JUNOsw dir, FSMP dir ----
Z_15040 := 0
LUCID_15040  := $(DATA_ROOT)/z0
JUNOSW_15040 := $(JUNOSW_ROOT)/0
FSMP_15040   := $(FSMP_ROOT)/15040

Z_15030 := 15
LUCID_15030  := $(DATA_ROOT)/z1500
JUNOSW_15030 := $(JUNOSW_ROOT)/15000
FSMP_15030   := $(FSMP_ROOT)/15030

Z_15022 := 16.1
LUCID_15022  := $(DATA_ROOT)/z1610
JUNOSW_15022 := $(JUNOSW_ROOT)/16100
FSMP_15022   := $(FSMP_ROOT)/15022

Z_15012 := 17.1
LUCID_15012  := $(DATA_ROOT)/z1710
JUNOSW_15012 := $(JUNOSW_ROOT)/17100
FSMP_15012   := $(FSMP_ROOT)/15012

RUNS := 15040 15030 15022 15012

# ---- recipes -------------------------------------------------------
# Three-per-run set: time/summary/amplitude PDFs + comparison CSV.
# Macro arguments: 1=z [m]  2=LUCiD dir  3=JUNOsw dir  4=FSMP dir
define COMPARE_RECIPE
	@mkdir -p $(MPLCONFIGDIR)
	cd $(PROJECT_ROOT) && $(PYTHON) -u compare_lucid_sim/compare_sim_vs_real.py \
	    -z $(1) \
	    --lucid-sim-dir $(2) \
	    --lucid-label "$(LUCID_LABEL)" \
	    --junosw-sim-dir $(3) \
	    --junosw-label "$(JUNOSW_LABEL)" \
	    --fsmp-dir $(4) \
	    -n $(CHANNEL_COUNT) --seed $(SEED) \
	    $(PAGE_COUNT_ARG) \
	    --output-root $(DIFFUSER_OUT) --output-tag $(TAG) \
	    --add-dark-to-simulation --lucid-qe-correction $(args)
endef

# TTS-vs-IRF five-curve comparison over the 125 characterized
# channels; same arguments as COMPARE_RECIPE.
define TTS_IRF_RECIPE
	@mkdir -p $(MPLCONFIGDIR)
	cd $(PROJECT_ROOT) && $(PYTHON) -u compare_lucid_sim/plot_tts_vs_irf.py \
	    -z $(1) \
	    $(CHANNEL_SEL_ARG) \
	    --lucid-sim-dir $(2) \
	    --lucid-label "$(LUCID_LABEL)" \
	    --junosw-sim-dir $(3) \
	    --junosw-label "$(JUNOSW_LABEL)" \
	    --fsmp-dir $(4) \
	    --irf-h5 $(IRF_H5) \
	    --no-time-difference-panels \
	    --add-dark-to-simulation --lucid-qe-correction \
	    --bins $(BINS) --fsmp-pe-scale $(FSMP_PE_SCALE) \
	    --output-root $(IRF125_OUT) $(args)
endef

# All-channel raw PE-scale comparison PDF.  --build-missing fills any
# absent raw cache from the source directories first.  Arguments:
# 1=run  2=LUCiD dir  3=JUNOsw dir  4=FSMP dir
define PE_SCALE_RECIPE
	@mkdir -p $(MPLCONFIGDIR) \
	    $(PE_SCALE_OUT)/$(1)/raw_all_pe_independent
	cd $(PROJECT_ROOT) && $(PYTHON) -u compare_lucid_sim/plot_pe_scale.py \
	    --run $(1) \
	    --fsmp-cache \
	        $(PE_SCALE_OUT)/$(1)/fsmp_data_all_raw_pe_scale_all.npz \
	    --simulation-cache \
	        $(PE_SCALE_OUT)/$(1)/sim_data_all_raw_pe_scale_0.npz \
	        $(PE_SCALE_OUT)/$(1)/junosw_sim_data_all_raw_pe_scale.npz \
	    --pmt-map $(PMT_MAP) \
	    --output-dir $(PE_SCALE_OUT)/$(1)/raw_all_pe_independent \
	    --bins $(PE_SCALE_BINS) --lucid-qe-effective $(QE) \
	    --pe-per-event --build-missing \
	    --lucid-cache \
	        $(PE_SCALE_OUT)/$(1)/sim_data_all_raw_pe_scale_0.npz \
	    --junosw-cache \
	        $(PE_SCALE_OUT)/$(1)/junosw_sim_data_all_raw_pe_scale.npz \
	    --fsmp-dir $(4) --lucid-sim-dir $(2) \
	    --junosw-sim-dir $(3) \
	    --lucid-label "$(LUCID_LABEL)" \
	    --junosw-label "$(JUNOSW_LABEL)" $(args)
endef

SCALE_Z_CSV := $(SCALE_Z_OUT)/channel_selection_all_17612.csv

# ---- targets --------------------------------------------------------
# compare-z: run is looked up from Z via run_532.csv; the FSMP
# directory uses the script default root (/junofs/C/PE).
Z ?= 0

COMPARE_TARGETS  := $(addprefix compare-,$(RUNS))
TTS_IRF_TARGETS  := $(addprefix tts-irf-,$(RUNS))
PE_SCALE_TARGETS := $(addprefix pe-scale-,$(RUNS))
THETA_TARGETS := $(addprefix theta-intensity-,$(RUNS))

# theta-intensity reuses the all-channel pe-scale caches; each run
# depends on its pe-scale target (cache-hit reruns take seconds).
define THETA_INTENSITY_RECIPE
	cd $(PROJECT_ROOT) && $(PYTHON) -u \
	    compare_lucid_sim/plot_theta_intensity.py \
	    --run $(1) --cache-root $(PE_SCALE_OUT)/$(1) $(args)
endef

.PHONY: help all compare-all tts-irf-all pe-scale-all scale-vs-z \
    theta-intensity-all \
    $(COMPARE_TARGETS) $(TTS_IRF_TARGETS) $(PE_SCALE_TARGETS) \
    $(THETA_TARGETS)

.DEFAULT_GOAL := help

compare-15040:
	$(call COMPARE_RECIPE,$(Z_15040),$(LUCID_15040),\
	    $(JUNOSW_15040),$(FSMP_15040))

compare-15030:
	$(call COMPARE_RECIPE,$(Z_15030),$(LUCID_15030),\
	    $(JUNOSW_15030),$(FSMP_15030))

compare-15022:
	$(call COMPARE_RECIPE,$(Z_15022),$(LUCID_15022),\
	    $(JUNOSW_15022),$(FSMP_15022))

compare-15012:
	$(call COMPARE_RECIPE,$(Z_15012),$(LUCID_15012),\
	    $(JUNOSW_15012),$(FSMP_15012))

compare-z:
	@mkdir -p $(MPLCONFIGDIR)
	cd $(PROJECT_ROOT) && $(PYTHON) -u \
	    compare_lucid_sim/compare_sim_vs_real.py \
	    -z $(Z) \
	    --lucid-sim-dir $(DATA_ROOT)/z$(shell awk \
	        "BEGIN{printf \"%d\", $(Z)*100}") \
	    --lucid-label "$(LUCID_LABEL)" \
	    --junosw-sim-dir $(JUNOSW_ROOT)/$(shell awk \
	        "BEGIN{printf \"%d\", $(Z)*1000}") \
	    --junosw-label "$(JUNOSW_LABEL)" \
	    -n $(CHANNEL_COUNT) --seed $(SEED) \
	    $(PAGE_COUNT_ARG) \
	    --output-root $(DIFFUSER_OUT) --output-tag $(TAG) \
	    --add-dark-to-simulation --lucid-qe-correction $(args)

compare-all: $(COMPARE_TARGETS)

tts-irf-15040:
	$(call TTS_IRF_RECIPE,$(Z_15040),$(LUCID_15040),\
	    $(JUNOSW_15040),$(FSMP_15040))

tts-irf-15030:
	$(call TTS_IRF_RECIPE,$(Z_15030),$(LUCID_15030),\
	    $(JUNOSW_15030),$(FSMP_15030))

tts-irf-15022:
	$(call TTS_IRF_RECIPE,$(Z_15022),$(LUCID_15022),\
	    $(JUNOSW_15022),$(FSMP_15022))

tts-irf-15012:
	$(call TTS_IRF_RECIPE,$(Z_15012),$(LUCID_15012),\
	    $(JUNOSW_15012),$(FSMP_15012))

tts-irf-all: $(TTS_IRF_TARGETS)

pe-scale-15040:
	$(call PE_SCALE_RECIPE,15040,$(LUCID_15040),\
	    $(JUNOSW_15040),$(FSMP_15040))

pe-scale-15030:
	$(call PE_SCALE_RECIPE,15030,$(LUCID_15030),\
	    $(JUNOSW_15030),$(FSMP_15030))

pe-scale-15022:
	$(call PE_SCALE_RECIPE,15022,$(LUCID_15022),\
	    $(JUNOSW_15022),$(FSMP_15022))

pe-scale-15012:
	$(call PE_SCALE_RECIPE,15012,$(LUCID_15012),\
	    $(JUNOSW_15012),$(FSMP_15012))

pe-scale-all: $(PE_SCALE_TARGETS)

theta-intensity-15040: pe-scale-15040
	$(call THETA_INTENSITY_RECIPE,15040)

theta-intensity-15030: pe-scale-15030
	$(call THETA_INTENSITY_RECIPE,15030)

theta-intensity-15022: pe-scale-15022
	$(call THETA_INTENSITY_RECIPE,15022)

theta-intensity-15012: pe-scale-15012
	$(call THETA_INTENSITY_RECIPE,15012)

theta-intensity-all: $(THETA_TARGETS)

scale-vs-z:
	@mkdir -p $(MPLCONFIGDIR) $(SCALE_Z_OUT)
	@if [[ ! -f $(SCALE_Z_CSV) ]]; then \
	    { echo channel; seq 0 17611; } > $(SCALE_Z_CSV); \
	    echo "Wrote $(SCALE_Z_CSV)"; \
	fi
	cd $(PROJECT_ROOT) && $(PYTHON) -u \
	    compare_lucid_sim/plot_source_scale_vs_z.py \
	    --lucid-root $(DATA_ROOT) \
	    --junosw-root $(JUNOSW_ROOT) \
	    --channel-selection $(SCALE_Z_CSV) \
	    --output-dir $(SCALE_Z_OUT) $(args)

all: compare-all tts-irf-all pe-scale-all scale-vs-z theta-intensity-all

# submit-<any-target>: run that target as a Slurm job instead of in
# the foreground.  The job re-invokes this same Makefile (exported
# variables carry every override), so the command templates live in
# exactly one place.  Examples:
#   make -C compare_lucid_sim submit-compare-15040        (one job, one run)
#   make -C compare_lucid_sim submit-compare-all          (one job, four runs)
#   make -C compare_lucid_sim submit-compare-15040 submit-compare-15030 \
#       submit-compare-15022 submit-compare-15012   (four jobs)
# Note: submit-* names must stay OUT of .PHONY -- pattern rules never
# match phony targets in GNU make, and the rule would be dead.
# MAKE_PROG hides the literal $(MAKE) so that "make -n submit-*" is a
# pure dry run (a recipe referencing $(MAKE) directly would execute
# even under -n).  Overrides travel on the wrapped make command line:
# sudo env_reset would silently drop exported variables.
MAKE_PROG = $(MAKE)
SUBMIT_VARS = \
    PYTHON='$(PYTHON)' MPLCONFIGDIR='$(MPLCONFIGDIR)' \
    DATA_ROOT='$(DATA_ROOT)' FSMP_ROOT='$(FSMP_ROOT)' \
    JUNOSW_ROOT='$(JUNOSW_ROOT)' PMT_MAP='$(PMT_MAP)' \
    IRF_H5='$(IRF_H5)' CHANNEL_SELECTION='$(CHANNEL_SELECTION)' \
    CHANNEL_COUNT='$(CHANNEL_COUNT)' SEED='$(SEED)' \
    PAGE_CHANNEL_COUNT='$(PAGE_CHANNEL_COUNT)' \
    BINS='$(BINS)' FSMP_PE_SCALE='$(FSMP_PE_SCALE)' QE='$(QE)' \
    LUCID_LABEL='$(LUCID_LABEL)' JUNOSW_LABEL='$(JUNOSW_LABEL)' \
    TAG='$(TAG)' PE_SCALE_BINS='$(PE_SCALE_BINS)' \
    DIFFUSER_OUT='$(DIFFUSER_OUT)' IRF125_OUT='$(IRF125_OUT)' \
    PE_SCALE_OUT='$(PE_SCALE_OUT)' SCALE_Z_OUT='$(SCALE_Z_OUT)' \
    args='$(args)'

submit-%:
	@mkdir -p $(SLURM_LOG_DIR)
	$(SBATCH) $(SBATCH_OPTS) --job-name=LUCiD_$* \
	    -o $(SLURM_LOG_DIR)/$*_%j.out \
	    -e $(SLURM_LOG_DIR)/$*_%j.err \
	    --chdir=$(PROJECT_ROOT) \
	    --wrap "$(MAKE_PROG) -C $(COMPARE_DIR) $* $(SUBMIT_VARS)"

help:
	@printf '%s\n' \
	  '532 nm laser three-way comparison (LUCiD / JUNOsw / FSMP)' \
	  '' \
	  'targets:' \
	  '  compare-<run> / compare-all    time+summary+amplitude PDFs' \
	  '                                 (runs $(RUNS))' \
	  '  compare-z Z=16.1               same, run looked up from z' \
	  '  tts-irf-<run> / tts-irf-all    TTS-vs-IRF pages (irf125)' \
	  '  pe-scale-<run> / pe-scale-all  all-channel PE-scale PDF' \
	  '  theta-intensity-<run>/-all     per-PMT PE vs theta' \
	  '  scale-vs-z                     source-scale vs z PDF' \
	  '  all                            everything' \
	  '' \
	  'submit-<target>                  run as a Slurm job, e.g. four' \
	  '  jobs: submit-compare-15040 .. submit-compare-15012' \
	  '' \
	  'data paths (override!):' \
	  '  DATA_ROOT   = $(DATA_ROOT)' \
	  '  FSMP_ROOT   = $(FSMP_ROOT)' \
	  '  JUNOSW_ROOT = $(JUNOSW_ROOT)' \
	  '  PMT_MAP     = $(PMT_MAP)' \
	  '  IRF_H5      = $(IRF_H5)' \
	  '  CHANNEL_SELECTION = $(CHANNEL_SELECTION)' \
	  '    (empty: derived from IRF_H5 + PMT_MAP)' \
	  'outputs: DIFFUSER/IRF125/PE_SCALE/SCALE_Z_OUT, all under' \
	  '  $$(DATA_ROOT) -- override DATA_ROOT or your runs write' \
	  '  into the author'"'"'s directory above' \
	  '' \
	  'settings: CHANNEL_COUNT=$(CHANNEL_COUNT) SEED=$(SEED)' \
	  '  BINS=$(BINS) FSMP_PE_SCALE=$(FSMP_PE_SCALE) QE=$(QE)' \
	  '  PAGE_CHANNEL_COUNT=$(PAGE_CHANNEL_COUNT) (empty: draw all)' \
	  'args="$(args)" passes extra script flags to any target' \
	  'example: make -C compare_lucid_sim compare-z Z=16.1' \
	  '          DATA_ROOT=/path/to/data'

