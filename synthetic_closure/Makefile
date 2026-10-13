SHELL := /bin/bash

CLOSURE_DIR := $(patsubst %/,%,$(dir \
	$(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(CLOSURE_DIR)/..)
MAKE_PROG := $(MAKE)
LUCID_PY ?= /opt/gentoo/usr/bin/python3.13

OUTDIR ?= /junofs/C/liukainan/LUCiD/minimal_closure
SOURCE ?= 0,0,17.1
SOURCES ?= $(SOURCE)
FIVE_SOURCES ?= 0,0,0 0,0,9 0,0,15 0,0,16.1 0,0,17.1
N_EVENTS ?= 1
DATA_N_PHOTONS ?= 128
FIT_N_PHOTONS ?= 128
K ?= 4
INTENSITY ?= 100000
TIME_MIN_NS ?= 0
TIME_MAX_NS ?= 300
TIME_BIN_NS ?= 50
INTERPOLATION_SIGMA_NS ?=
WINDOW_EXPANSION_NS ?= 50
TRUNCATE_SIGMA ?= 5
JOINT_STEPS ?= 1
START_FRACTION ?= 0.1
SEED ?= 42
FIT_SEED_OFFSET ?= 0
MC_VALIDATION_SEED_OFFSET ?= 2000003
DATA_ENSEMBLE ?= 1
FIT_ENSEMBLE ?= 1
MC_VALIDATION_ENSEMBLE ?= 1
AD_FD_RELATIVE_STEP ?= 0
VALIDATION_EVERY ?= 1
EARLY_STOPPING_PATIENCE ?= 0
MIN_IMPROVEMENT ?= 0.0001
LEARNING_RATE ?= 0.001
GRAD_CLIP ?= 10000
SOURCE_CHUNK_SIZE ?= 0
OBJECTIVE_EXECUTION ?= fused
CHECKPOINT_EVERY ?= 0
RESTART_CHECKPOINT ?=
SCRIPT_ARGS ?=
PLATFORM ?= cpu

SBATCH ?= sudo -u\#35905 sbatch
SBATCH_OPTS ?= --partition=gpu --gres=gpu:mi100:1 --cpus-per-task=4 \
	--mem=128G --time=24:00:00
SLURM_LOG_DIR ?= /junofs/C/liukainan/LUCiD/slurm/calibration
SLURM_JOB ?= $(PROJECT_ROOT)/run/run_juno_full_geometry_closure.sh

SUBMIT_N_EVENTS ?= 100
SUBMIT_DATA_N_PHOTONS ?= 4000000
SUBMIT_FIT_N_PHOTONS ?= 4000000
SUBMIT_K ?= 32
SUBMIT_INTENSITY ?= 400000
SUBMIT_TIME_MAX_NS ?= 1000
SUBMIT_TIME_BIN_NS ?= 0.5
SUBMIT_INTERPOLATION_SIGMA_NS ?= 0.5
SUBMIT_DATA_ENSEMBLE ?= 2
SUBMIT_FIT_ENSEMBLE ?= 1
SUBMIT_MC_VALIDATION_ENSEMBLE ?= 2
SUBMIT_FIT_SEED_OFFSET ?= 1000003
SUBMIT_MC_VALIDATION_SEED_OFFSET ?= 2000003
SUBMIT_JOINT_STEPS ?= 600
SUBMIT_AD_FD_RELATIVE_STEP ?= 0.001
SUBMIT_VALIDATION_EVERY ?= 2
SUBMIT_OBJECTIVE_EXECUTION ?= per-source
SUBMIT_CHECKPOINT_EVERY ?= 4

.PHONY: help test run minimal-smoke submit-minimal-five

help:
	@echo "Minimal JUNO event-wise closure"
	@echo "  make -C synthetic_closure test"
	@echo "  make -C synthetic_closure minimal-smoke OUTDIR=/tmp/..."
	@echo "  make -n -C synthetic_closure submit-minimal-five OUTDIR=/junofs/..."
	@echo "  make -C synthetic_closure submit-minimal-five OUTDIR=/junofs/..."

test:
	cd "$(PROJECT_ROOT)"; \
	export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true; \
	"$(LUCID_PY)" -m pytest -q \
		tests/test_free_path_score_gradient.py \
		tests/test_juno_eventwise_likelihood.py \
		tests/test_juno_closure_plot.py \
		tests/test_juno_zero_support_plot.py \
		tests/test_juno_calibration_diagnostics.py \
		tests/test_juno_laser_histogram.py \
		tests/test_juno_laser_materials.py \
		tests/test_juno_laser_time_grid.py \
		tests/test_minimal_closure_contract.py

run:
	test -n "$(OUTDIR)"
	test ! -e "$(OUTDIR)/closure_result.json"
	cd "$(PROJECT_ROOT)"; \
	export JAX_PLATFORMS="$(PLATFORM)" JAX_ENABLE_X64=true; \
	"$(LUCID_PY)" scripts/run_juno_full_geometry_closure.py \
		$(foreach position,$(SOURCES),--source-position "$(position)") \
		--outdir "$(OUTDIR)" \
		--n-events "$(N_EVENTS)" \
		--data-n-photons "$(DATA_N_PHOTONS)" \
		--fit-n-photons "$(FIT_N_PHOTONS)" \
		--k "$(K)" \
		--intensity "$(INTENSITY)" \
		--time-min-ns "$(TIME_MIN_NS)" \
		--time-max-ns "$(TIME_MAX_NS)" \
		--time-bin-ns "$(TIME_BIN_NS)" \
		$(if $(INTERPOLATION_SIGMA_NS),--interpolation-sigma-ns \
		"$(INTERPOLATION_SIGMA_NS)") \
		--truncate-sigma "$(TRUNCATE_SIGMA)" \
		--window-stability-expansion-ns "$(WINDOW_EXPANSION_NS)" \
		--joint-steps "$(JOINT_STEPS)" \
		--start-fraction "$(START_FRACTION)" \
		--fit-seed-offset "$(FIT_SEED_OFFSET)" \
		--mc-validation-seed-offset "$(MC_VALIDATION_SEED_OFFSET)" \
		--data-ensemble-size "$(DATA_ENSEMBLE)" \
		--fit-ensemble-size "$(FIT_ENSEMBLE)" \
		--mc-validation-ensemble-size "$(MC_VALIDATION_ENSEMBLE)" \
		--ad-fd-relative-step "$(AD_FD_RELATIVE_STEP)" \
		--validation-every "$(VALIDATION_EVERY)" \
		--early-stopping-patience "$(EARLY_STOPPING_PATIENCE)" \
		--min-improvement "$(MIN_IMPROVEMENT)" \
		--learning-rate "$(LEARNING_RATE)" \
		--grad-clip "$(GRAD_CLIP)" \
		--source-chunk-size "$(SOURCE_CHUNK_SIZE)" \
		--objective-execution "$(OBJECTIVE_EXECUTION)" \
		--checkpoint-every "$(CHECKPOINT_EVERY)" \
		$(if $(RESTART_CHECKPOINT),--restart-checkpoint \
		"$(RESTART_CHECKPOINT)") \
		--seed "$(SEED)" \
		$(SCRIPT_ARGS)

minimal-smoke:
	$(MAKE_PROG) -f "$(CLOSURE_DIR)/Makefile" run \
		PLATFORM=cpu FIT_SEED_OFFSET=0 MC_VALIDATION_SEED_OFFSET=0 \
		START_FRACTION=0 LEARNING_RATE=0.00000001

SUBMIT_VARS = OUTDIR='$(OUTDIR)' \
	N_EVENTS='$(SUBMIT_N_EVENTS)' \
	DATA_N_PHOTONS='$(SUBMIT_DATA_N_PHOTONS)' \
	FIT_N_PHOTONS='$(SUBMIT_FIT_N_PHOTONS)' K='$(SUBMIT_K)' \
	INTENSITY='$(SUBMIT_INTENSITY)' \
	TIME_MAX_NS='$(SUBMIT_TIME_MAX_NS)' \
	TIME_BIN_NS='$(SUBMIT_TIME_BIN_NS)' \
	INTERPOLATION_SIGMA_NS='$(SUBMIT_INTERPOLATION_SIGMA_NS)' \
	DATA_ENSEMBLE='$(SUBMIT_DATA_ENSEMBLE)' \
	FIT_ENSEMBLE='$(SUBMIT_FIT_ENSEMBLE)' \
	MC_VALIDATION_ENSEMBLE='$(SUBMIT_MC_VALIDATION_ENSEMBLE)' \
	FIT_SEED_OFFSET='$(SUBMIT_FIT_SEED_OFFSET)' \
	MC_VALIDATION_SEED_OFFSET='$(SUBMIT_MC_VALIDATION_SEED_OFFSET)' \
	JOINT_STEPS='$(SUBMIT_JOINT_STEPS)' \
	START_FRACTION='$(START_FRACTION)' \
	AD_FD_RELATIVE_STEP='$(SUBMIT_AD_FD_RELATIVE_STEP)' \
	VALIDATION_EVERY='$(SUBMIT_VALIDATION_EVERY)' \
	SOURCE_CHUNK_SIZE='1' \
	OBJECTIVE_EXECUTION='$(SUBMIT_OBJECTIVE_EXECUTION)' \
	CHECKPOINT_EVERY='$(SUBMIT_CHECKPOINT_EVERY)' \
	RESTART_CHECKPOINT='$(RESTART_CHECKPOINT)' \
	SCRIPT_ARGS='$(SCRIPT_ARGS)'

submit-minimal-five:
	test -n "$(OUTDIR)"
	test "$(OUTDIR)" != "/junofs/C/liukainan/LUCiD/minimal_closure"
	test ! -e "$(OUTDIR)/closure_result.json"
	mkdir -p "$(SLURM_LOG_DIR)"
	$(SBATCH) $(SBATCH_OPTS) \
		--job-name="minimal_closure_five" \
		--output="$(SLURM_LOG_DIR)/minimal_closure_%j.out" \
		--error="$(SLURM_LOG_DIR)/minimal_closure_%j.err" \
		--chdir="$(PROJECT_ROOT)" \
		--wrap="env SOURCES='$(FIVE_SOURCES)' $(SUBMIT_VARS) \
		bash $(SLURM_JOB)"
