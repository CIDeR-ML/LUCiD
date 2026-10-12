SHELL := /bin/bash

CLOSURE_DIR := $(patsubst %/,%,$(dir \
	$(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(CLOSURE_DIR)/..)
MAKE_PROG = $(MAKE)

OUTDIR ?= /junofs/C/liukainan/LUCiD/lucid_eventwise_closure
MODEL_PROFILE ?= full
OBSERVATION_ROOT ?=
SOURCE ?= 0,0,17.1
SOURCES ?= $(SOURCE)
ONE_SOURCES ?= 0,0,17.1
FIVE_SOURCES ?= 0,0,0 0,0,9 0,0,15 0,0,16.1 0,0,17.1
N_EVENTS ?= 1
N_PHOTONS ?= 128
DATA_N_PHOTONS ?= $(N_PHOTONS)
FIT_N_PHOTONS ?= $(N_PHOTONS)
K ?= 4
INTENSITY ?= 100000
EVENT_EXPOSURE ?= 1
EVENT_TIME_OFFSET_NS ?= 0
BACKGROUND_RATE_HZ ?= 0
TIME_MIN_NS ?= 0
TIME_MAX_NS ?= 300
TIME_BIN_NS ?= 50
INTERPOLATION_SIGMA_NS ?=
WINDOW_EXPANSION_NS ?= 50
TRUNCATE_SIGMA ?= 5
STEPS ?= 1
OPTIMIZATION_PLAN ?= joint-only
JOINT_STEPS ?= $(STEPS)
START_FRACTION ?= 0.1
SEED ?= 42
FIT_SEED_OFFSET ?= 0
MC_VALIDATION_SEED_OFFSET ?= 2000003
DATA_ENSEMBLE ?= 1
FIT_ENSEMBLE ?= 1
MC_VALIDATION_ENSEMBLE ?= 1
DATA_IMPORTANCE ?= config
FIT_IMPORTANCE ?= config
AD_FD_RELATIVE_STEP ?= 0
HELDOUT_SOURCE_INDEX ?=
HELDOUT_SEED_OFFSET ?= 0
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
SUPPORT_DIAGNOSTIC_ONLY ?= 0
SBATCH ?= sudo -u\#35905 sbatch
SBATCH_OPTS ?= --partition=gpu --gres=gpu:mi100:1 --cpus-per-task=4 \
	--mem=128G --time=24:00:00
SLURM_LOG_DIR ?= /junofs/C/liukainan/LUCiD/slurm/calibration
SLURM_JOB ?= $(PROJECT_ROOT)/run/run_juno_full_geometry_closure.sh
# QUAL_* 是 submit-* 提交族的默认生产参数（原 qualification 档）。
QUAL_N_EVENTS ?= 100
QUAL_DATA_N_PHOTONS ?= 4000000
QUAL_FIT_N_PHOTONS ?= 4000000
QUAL_K ?= 32
QUAL_INTENSITY ?= 400000
QUAL_TIME_MAX_NS ?= 1000
QUAL_TIME_BIN_NS ?= 2
QUAL_INTERPOLATION_SIGMA_NS ?=
QUAL_DATA_ENSEMBLE ?= 2
QUAL_FIT_ENSEMBLE ?= 1
QUAL_MC_VALIDATION_ENSEMBLE ?= 2
QUAL_FIT_SEED_OFFSET ?= 1000003
QUAL_MC_VALIDATION_SEED_OFFSET ?= 2000003
QUAL_BACKGROUND_RATE_HZ ?= 30000
MINIMAL_BACKGROUND_RATE_HZ ?= 0
QUAL_JOINT_STEPS ?= 128
QUAL_START_FRACTION ?= 0.1
QUAL_VALIDATION_EVERY ?= 2
QUAL_SOURCE_CHUNK_SIZE ?= 0
QUAL_OBJECTIVE_EXECUTION ?= per-source
QUAL_CHECKPOINT_EVERY ?= 4
FORMAL_N_EVENTS ?= 100
FORMAL_JOINT_STEPS ?= 600
FORMAL_AD_FD_RELATIVE_STEP ?= 0.001
SUBMIT_N_EVENTS ?= $(FORMAL_N_EVENTS)
SUBMIT_JOINT_STEPS ?= $(FORMAL_JOINT_STEPS)
SUBMIT_AD_FD_RELATIVE_STEP ?= $(FORMAL_AD_FD_RELATIVE_STEP)
FISHER_BACKGROUND_RATE_HZ ?= 30000
FISHER_OUTDIR ?= /junofs/C/liukainan/LUCiD/lucid_fisher
FISHER_N_PHOTONS ?= $(QUAL_FIT_N_PHOTONS)
FISHER_ENSEMBLE ?= $(QUAL_FIT_ENSEMBLE)
FISHER_N_EVENTS ?= $(FORMAL_N_EVENTS)
FISHER_K ?= $(QUAL_K)
FISHER_SLURM_JOB ?= $(PROJECT_ROOT)/run/run_juno_synthetic_fisher.sh
CLOSURE_RESULT ?= $(OUTDIR)/closure_result.json
CLOSURE_DIAGNOSTICS_PDF ?= $(OUTDIR)/closure_diagnostics.pdf
CLOSURE_DIAGNOSTICS_JSON ?= $(OUTDIR)/closure_diagnostics.json
FISHER_RESULT ?= $(FISHER_OUTDIR)/fisher_result.json
FISHER_DIAGNOSTICS_PDF ?= $(FISHER_OUTDIR)/fisher_diagnostics.pdf
JUNO_OUTPUT_ROOT ?= /junofs/C/liukainan/LUCiD
FULL_SCORE_FIXED_TAG ?= closure_full_geometry_scatter_score_fixed_s600
FULL_SCORE_FIXED_ROOT ?= $(JUNO_OUTPUT_ROOT)/$(FULL_SCORE_FIXED_TAG)
FULL_SCORE_FIXED_OUTDIR ?= $(FULL_SCORE_FIXED_ROOT)/five_sources
FULL_SCORE_FIXED_FISHER_OUTDIR ?= \
	$(FULL_SCORE_FIXED_ROOT)/five_sources_fisher

.PHONY: help test run smoke minimal-smoke \
	submit-one submit-five submit-custom submit-fit-only \
	submit-five-fit-only submit-minimal-one submit-minimal-five \
	submit-five-support-diagnostic \
	submit-one-fisher submit-five-fisher \
	submit-full-score-fixed-five \
	submit-full-score-fixed-five-fisher \
	closure-diagnostics fisher-diagnostics \
	full-score-fixed-five-diagnostics

help:
	@echo "Synthetic event-wise closure workflow"
	@echo "  make -C synthetic_closure test"
	@echo "  make -C synthetic_closure smoke OUTDIR=/tmp/closure-smoke"
	@echo "  make -C synthetic_closure minimal-smoke OUTDIR=/tmp/..."
	@echo "  make -C synthetic_closure submit-minimal-five OUTDIR=/junofs/..."
	@echo "  make -C synthetic_closure submit-five OUTDIR=/junofs/..."
	@echo "  make -C synthetic_closure submit-five-fit-only \
OUTDIR=/junofs/... OBSERVATION_ROOT=/junofs/.../generation"
	@echo "  make -C synthetic_closure submit-one-fisher \
FISHER_OUTDIR=/junofs/..."
	@echo "  make -C synthetic_closure closure-diagnostics \
OUTDIR=/junofs/.../completed-closure"
	@echo "  make -C synthetic_closure submit-full-score-fixed-five"
	@echo "  make -C synthetic_closure \
full-score-fixed-five-diagnostics"
	@echo "Override SOURCES='x,y,z ...' for an arbitrary source set."

test:
	cd "$(PROJECT_ROOT)"; source run/env.sh; \
	export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true; \
	"$$LUCID_PY" -m pytest -q \
		tests/test_free_path_score_gradient.py \
		tests/test_juno_eventwise_likelihood.py \
		 tests/test_juno_closure_plot.py \
		 tests/test_juno_closure_cli.py \
		 tests/test_juno_zero_support_plot.py \
		tests/test_synthetic_closure_makefile.py

run:
	test -n "$(OUTDIR)"
	test ! -e "$(OUTDIR)/closure_result.json"
	cd "$(PROJECT_ROOT)"; source run/env.sh; \
	export JAX_PLATFORMS="$(PLATFORM)" JAX_ENABLE_X64=true; \
	"$$LUCID_PY" scripts/run_juno_full_geometry_closure.py \
		$(foreach position,$(SOURCES),--source-position "$(position)") \
		--model-profile "$(MODEL_PROFILE)" \
		--outdir "$(OUTDIR)" \
		--n-events "$(N_EVENTS)" \
		--data-n-photons "$(DATA_N_PHOTONS)" \
		--fit-n-photons "$(FIT_N_PHOTONS)" \
		--k "$(K)" \
		--intensity "$(INTENSITY)" \
		--event-exposure "$(EVENT_EXPOSURE)" \
		--event-time-offset-ns "$(EVENT_TIME_OFFSET_NS)" \
		--background-rate-hz "$(BACKGROUND_RATE_HZ)" \
		--time-min-ns "$(TIME_MIN_NS)" \
		--time-max-ns "$(TIME_MAX_NS)" \
		--time-bin-ns "$(TIME_BIN_NS)" \
		$(if $(INTERPOLATION_SIGMA_NS),--interpolation-sigma-ns \
		"$(INTERPOLATION_SIGMA_NS)") \
		--truncate-sigma "$(TRUNCATE_SIGMA)" \
		--window-stability-expansion-ns "$(WINDOW_EXPANSION_NS)" \
		--optimization-plan "$(OPTIMIZATION_PLAN)" \
		--joint-steps "$(JOINT_STEPS)" \
		--stage1-steps "$(STEPS)" \
		--stage2-steps "$(STEPS)" \
		--stage3-steps "$(STEPS)" \
		--start-fraction "$(START_FRACTION)" \
		--fit-seed-offset "$(FIT_SEED_OFFSET)" \
		--mc-validation-seed-offset \
		"$(MC_VALIDATION_SEED_OFFSET)" \
		--data-ensemble-size "$(DATA_ENSEMBLE)" \
		--fit-ensemble-size "$(FIT_ENSEMBLE)" \
		--mc-validation-ensemble-size \
		"$(MC_VALIDATION_ENSEMBLE)" \
		--data-importance-normalization "$(DATA_IMPORTANCE)" \
		--fit-importance-normalization "$(FIT_IMPORTANCE)" \
		--ad-fd-relative-step "$(AD_FD_RELATIVE_STEP)" \
		$(if $(HELDOUT_SOURCE_INDEX),--heldout-source-index \
		"$(HELDOUT_SOURCE_INDEX)") \
		--heldout-seed-offset "$(HELDOUT_SEED_OFFSET)" \
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

PLATFORM ?= rocm

smoke:
	$(MAKE_PROG) -f "$(CLOSURE_DIR)/Makefile" run \
		PLATFORM=cpu

minimal-smoke:
	$(MAKE_PROG) -f "$(CLOSURE_DIR)/Makefile" smoke \
		MODEL_PROFILE=minimal BACKGROUND_RATE_HZ=0 \
		EVENT_EXPOSURE=1 EVENT_TIME_OFFSET_NS=0 \
		OPTIMIZATION_PLAN=joint-only FIT_SEED_OFFSET=0 \
		MC_VALIDATION_SEED_OFFSET=0 START_FRACTION=0 \
		LEARNING_RATE=0.00000001

SUBMIT_VARS = OUTDIR='$(OUTDIR)' \
	MODEL_PROFILE='$(MODEL_PROFILE)' \
	CLOSURE_MODE='$(SUBMIT_MODE)' \
	OBSERVATION_ROOT='$(OBSERVATION_ROOT)' \
		N_EVENTS='$(SUBMIT_N_EVENTS)' \
	DATA_N_PHOTONS='$(QUAL_DATA_N_PHOTONS)' \
	FIT_N_PHOTONS='$(QUAL_FIT_N_PHOTONS)' K='$(QUAL_K)' \
	INTENSITY='$(QUAL_INTENSITY)' TIME_MAX_NS='$(QUAL_TIME_MAX_NS)' \
	TIME_BIN_NS='$(QUAL_TIME_BIN_NS)' \
	INTERPOLATION_SIGMA_NS='$(QUAL_INTERPOLATION_SIGMA_NS)' \
	DATA_ENSEMBLE='$(QUAL_DATA_ENSEMBLE)' \
	FIT_ENSEMBLE='$(QUAL_FIT_ENSEMBLE)' \
	MC_VALIDATION_ENSEMBLE='$(QUAL_MC_VALIDATION_ENSEMBLE)' \
	FIT_SEED_OFFSET='$(QUAL_FIT_SEED_OFFSET)' \
	MC_VALIDATION_SEED_OFFSET='$(QUAL_MC_VALIDATION_SEED_OFFSET)' \
	BACKGROUND_RATE_HZ='$(QUAL_BACKGROUND_RATE_HZ)' \
	OPTIMIZATION_PLAN='joint-only' JOINT_STEPS='$(SUBMIT_JOINT_STEPS)' \
	START_FRACTION='$(QUAL_START_FRACTION)' \
	AD_FD_RELATIVE_STEP='$(SUBMIT_AD_FD_RELATIVE_STEP)' \
	VALIDATION_EVERY='$(QUAL_VALIDATION_EVERY)' \
	SOURCE_CHUNK_SIZE='$(QUAL_SOURCE_CHUNK_SIZE)' \
	OBJECTIVE_EXECUTION='$(QUAL_OBJECTIVE_EXECUTION)' \
	CHECKPOINT_EVERY='$(QUAL_CHECKPOINT_EVERY)' \
	RESTART_CHECKPOINT='$(RESTART_CHECKPOINT)' \
	SUPPORT_DIAGNOSTIC_ONLY='$(SUPPORT_DIAGNOSTIC_ONLY)' \
	SCRIPT_ARGS='$(SCRIPT_ARGS)'

submit-one: SUBMIT_SOURCES = $(ONE_SOURCES)
submit-five: SUBMIT_SOURCES = $(FIVE_SOURCES)
submit-custom: SUBMIT_SOURCES = $(SOURCES)
submit-fit-only: SUBMIT_SOURCES = $(ONE_SOURCES)
submit-fit-only: SUBMIT_MODE = fit-only
submit-fit-only: SUBMIT_N_EVENTS = $(QUAL_N_EVENTS)
submit-fit-only: SUBMIT_JOINT_STEPS = $(QUAL_JOINT_STEPS)
submit-fit-only: SUBMIT_AD_FD_RELATIVE_STEP = 0
submit-five-fit-only: SUBMIT_SOURCES = $(FIVE_SOURCES)
submit-five-fit-only: SUBMIT_MODE = fit-only
submit-five-fit-only: SUBMIT_N_EVENTS = $(QUAL_N_EVENTS)
submit-five-fit-only: SUBMIT_JOINT_STEPS = $(QUAL_JOINT_STEPS)
submit-five-fit-only: SUBMIT_AD_FD_RELATIVE_STEP = 0
submit-minimal-one: SUBMIT_SOURCES = $(ONE_SOURCES)
submit-minimal-one: override MODEL_PROFILE = minimal
submit-minimal-one: override QUAL_BACKGROUND_RATE_HZ = $(MINIMAL_BACKGROUND_RATE_HZ)
submit-minimal-one: override QUAL_TIME_BIN_NS = 0.5
submit-minimal-one: override QUAL_INTERPOLATION_SIGMA_NS = 0.5
submit-minimal-five: SUBMIT_SOURCES = $(FIVE_SOURCES)
submit-minimal-five: override MODEL_PROFILE = minimal
submit-minimal-five: override QUAL_BACKGROUND_RATE_HZ = $(MINIMAL_BACKGROUND_RATE_HZ)
submit-minimal-five: override QUAL_TIME_BIN_NS = 0.5
submit-minimal-five: override QUAL_INTERPOLATION_SIGMA_NS = 0.5
submit-five-support-diagnostic: SUBMIT_SOURCES = $(FIVE_SOURCES)
submit-five-support-diagnostic: SUBMIT_MODE = fit-only
submit-five-support-diagnostic: SUBMIT_N_EVENTS = $(QUAL_N_EVENTS)
submit-five-support-diagnostic: SUBMIT_JOINT_STEPS = 1
submit-five-support-diagnostic: SUBMIT_AD_FD_RELATIVE_STEP = 0
submit-five-support-diagnostic: override MODEL_PROFILE = minimal
submit-five-support-diagnostic: override QUAL_BACKGROUND_RATE_HZ = 0
submit-five-support-diagnostic: override QUAL_TIME_BIN_NS = 0.5
submit-five-support-diagnostic: override QUAL_INTERPOLATION_SIGMA_NS = 0.5
submit-five-support-diagnostic: override SUPPORT_DIAGNOSTIC_ONLY = 1

SUBMIT_MODE ?= full

submit-one submit-five submit-custom submit-fit-only submit-five-fit-only \
submit-minimal-one submit-minimal-five submit-five-support-diagnostic:
	test -n "$(OUTDIR)"
	test "$(OUTDIR)" != \
		"/junofs/C/liukainan/LUCiD/lucid_eventwise_closure"
	test ! -e "$(OUTDIR)/closure_result.json"
	if [[ "$(SUBMIT_MODE)" == "fit-only" ]]; then \
		test -d "$(OBSERVATION_ROOT)"; \
	fi
	mkdir -p "$(SLURM_LOG_DIR)"
	$(SBATCH) $(SBATCH_OPTS) \
		--job-name="closure_$(patsubst submit-%,%,$@)" \
		--output="$(SLURM_LOG_DIR)/closure_%j.out" \
		--error="$(SLURM_LOG_DIR)/closure_%j.err" \
		--chdir="$(PROJECT_ROOT)" \
		--wrap="env SOURCES='$(SUBMIT_SOURCES)' $(SUBMIT_VARS) \
	bash $(SLURM_JOB)"

submit-one-fisher: SUBMIT_FISHER_SOURCES = $(ONE_SOURCES)
submit-five-fisher: SUBMIT_FISHER_SOURCES = $(FIVE_SOURCES)

submit-one-fisher submit-five-fisher:
	test -n "$(FISHER_OUTDIR)"
	test ! -e "$(FISHER_OUTDIR)/fisher_result.json"
	mkdir -p "$(SLURM_LOG_DIR)"
	$(SBATCH) $(SBATCH_OPTS) \
		--job-name="$(patsubst submit-%,%,$@)" \
		--output="$(SLURM_LOG_DIR)/fisher_%j.out" \
		--error="$(SLURM_LOG_DIR)/fisher_%j.err" \
		--chdir="$(PROJECT_ROOT)" \
		--wrap="env SOURCES='$(SUBMIT_FISHER_SOURCES)' \
OUTDIR='$(FISHER_OUTDIR)' N_PHOTONS='$(FISHER_N_PHOTONS)' \
N_EVENTS='$(FISHER_N_EVENTS)' K='$(FISHER_K)' \
INTENSITY='$(QUAL_INTENSITY)' ENSEMBLE_SIZE='$(FISHER_ENSEMBLE)' \
BACKGROUND_RATE_HZ='$(FISHER_BACKGROUND_RATE_HZ)' \
TIME_MAX_NS='$(QUAL_TIME_MAX_NS)' TIME_BIN_NS='$(QUAL_TIME_BIN_NS)' \
bash $(FISHER_SLURM_JOB)"

submit-full-score-fixed-five: override OUTDIR = \
	$(FULL_SCORE_FIXED_OUTDIR)
submit-full-score-fixed-five: submit-five

submit-full-score-fixed-five-fisher: override FISHER_OUTDIR = \
	$(FULL_SCORE_FIXED_FISHER_OUTDIR)
submit-full-score-fixed-five-fisher: submit-five-fisher

closure-diagnostics:
	test -f "$(CLOSURE_RESULT)"
	cd "$(PROJECT_ROOT)"; source run/env.sh; \
	FISHER_ARGS=(); \
	if [[ -f "$(FISHER_RESULT)" ]]; then \
		FISHER_ARGS+=(--fisher-result "$(FISHER_RESULT)"); \
	fi; \
	"$$LUCID_PY" scripts/plot_juno_closure_result.py \
		--result "$(CLOSURE_RESULT)" \
		--output "$(CLOSURE_DIAGNOSTICS_PDF)" \
		--summary "$(CLOSURE_DIAGNOSTICS_JSON)" \
		"$${FISHER_ARGS[@]}"

fisher-diagnostics:
	test -f "$(FISHER_RESULT)"
	cd "$(PROJECT_ROOT)"; source run/env.sh; \
	"$$LUCID_PY" scripts/plot_juno_synthetic_fisher.py \
		--result "$(FISHER_RESULT)" \
		--output "$(FISHER_DIAGNOSTICS_PDF)"

full-score-fixed-five-diagnostics: override OUTDIR = \
	$(FULL_SCORE_FIXED_OUTDIR)
full-score-fixed-five-diagnostics: override FISHER_OUTDIR = \
	$(FULL_SCORE_FIXED_FISHER_OUTDIR)
full-score-fixed-five-diagnostics: closure-diagnostics fisher-diagnostics
