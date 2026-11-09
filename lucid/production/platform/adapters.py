"""Cluster adapters — the one place that knows SLURM, NERSC and HTCondor.

Each adapter owns the submit-description template, the submit command,
and the apptainer bind list for its cluster. Everything else (planning,
config emission, completion checks) is shared and lives with the
applications (`dataprod/`, `siren_inputs/`).

Selection is sticky via `CLUSTER=slurm|nersc|htcondor` in `user_paths.sh`.
Default is `slurm` to preserve backward compatibility with the existing
S3DF user_paths.sh that doesn't set the var.

To add a third cluster:

  1. Subclass `ClusterAdapter` and implement `render_*` methods plus the
     class-level constants.
  2. Add it to `ADAPTERS` at the bottom.
  3. Set `CLUSTER=<name>` in the new cluster's `user_paths.sh.template`.

The render methods all take `env` (the result of `load_user_paths()`)
plus a few cluster-agnostic fields. The HTCondor adapter ignores SLURM
env keys; the SLURM adapter ignores HTCondor env keys. Validation of
required keys happens inside each adapter's `render_*` so we get a clear
error when a cluster's user_paths.sh is missing a setting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


# --- Apptainer bind defaults --------------------------------------------------

# Sites that aren't auto-bound by apptainer's container build. We list them so
# the same `lucid-run-job` invocation works from any cluster's batch node.
SLURM_DEFAULT_BINDS = "/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs"
HTCONDOR_DEFAULT_BINDS = "/afs,/eos,/cvmfs"


def _dev_binds(env: Dict[str, str]) -> str:
    """Optional LUCID_DEV_PATH / PHOTONSIM_DEV_PATH bind shadows.

    Shared across clusters: if the user exports these in `user_paths.sh`,
    the host checkout shadows the baked container copy. Lets you validate
    source-level changes under the cluster without rebuilding the image.
    """
    pieces: List[str] = []
    if env.get("LUCID_DEV_PATH"):
        pieces.append(f"-B {env['LUCID_DEV_PATH']}:/opt/LUCiD")
    if env.get("PHOTONSIM_DEV_PATH"):
        pieces.append(f"-B {env['PHOTONSIM_DEV_PATH']}:/opt/PhotonSim")
    return " ".join(pieces)


# --- Base adapter ------------------------------------------------------------

class ClusterAdapter(ABC):
    """One per cluster. Implements `render_*` for each job kind."""

    name: str
    submit_extension: str       # e.g. "sbatch", "sub"
    submit_cmd: str             # e.g. "sbatch", "condor_submit"

    def __init__(self, env: Dict[str, str]) -> None:
        self.env = env
        # APPTAINER_BINDS env var can override the per-cluster default.
        self.apptainer_binds = env.get("APPTAINER_BINDS") or self._default_binds()

    @abstractmethod
    def _default_binds(self) -> str: ...

    # --- per-kind render methods ---

    @abstractmethod
    def render_siren_cell(self, *, cell_dir: Path, cell_cfg: Path,
                          energy_mev: int, job_name: str, job_id: int,
                          partition: str, use_gpu: bool,
                          array_spec: Optional[str] = None) -> str: ...

    @abstractmethod
    def render_smax_cell(self, *, cell_dir: Path, cell_cfg: Path,
                         energy_mev: int, job_name: str, job_id: int,
                         partition: str, use_gpu: bool) -> str: ...

    @abstractmethod
    def render_dataprod_job(self, *, cell_dir: Path, config_path: Path,
                            detector: str, job_id: str, job_name: str,
                            partition: str, use_gpu: bool, test: bool,
                            skip_lucid: bool, n_events: Optional[int],
                            override_energy_mev: Optional[float],
                            sn_model: Optional[str] = None,
                            sn_ordering: Optional[str] = None) -> str: ...

    @abstractmethod
    def render_command_job(self, *, command: str, cell_dir: Path, job_name: str,
                           log_stem: str, partition: str, use_gpu: bool = False,
                           request_disk_mb: int = 4096,
                           request_memory_mb: int = 0, gpus: str | None = None) -> str:
        """Submit description for an arbitrary command run inside the container.

        The stage-specific renderers above each bake in one entry point. Some
        work is a short pipeline instead (the fiTQun Cherenkov-profile scan runs
        PhotonSim on a generated macro, reduces the output, then deletes it),
        which is a shell command, not a `lucid-run-job` invocation.

        ``request_memory_mb`` overrides the cluster default for the merges that
        hold a whole dense table in memory; 0 keeps the default. ``gpus`` sets the
        GPU request outright (``"0"`` for a CPU-only job on a site whose
        ``DEFAULT_GPUS`` is 1); left out, it is 1 with ``use_gpu`` and the site's
        ``DEFAULT_GPUS`` otherwise.
        """
        raise NotImplementedError

    def render_train_run(self, *, run_dir: Path, run_name: str,
                         cli_args: str, resources: Dict[str, str]) -> str:
        """Submit description for one ``lucid-train-siren`` run (a GPU job).

        ``resources`` is the train_siren generator's: ``partition`` and ``account`` (Slurm, from
        ``user_paths.sh``), ``time``, ``memory``, ``cpus``, ``gpus`` and, on HTCondor,
        ``flavour``. ``LUCID_ENV_BASE`` (the user site with the GPU jaxlib) is exported as
        ``PYTHONUSERBASE``; ``LUCID_DEV_PATH``, when set, is bound over ``/opt/LUCiD``.
        """
        raise NotImplementedError


# --- SLURM -------------------------------------------------------------------

class SlurmAdapter(ClusterAdapter):
    name = "slurm"
    submit_extension = "sbatch"
    submit_cmd = "sbatch"

    def _default_binds(self) -> str:
        return SLURM_DEFAULT_BINDS

    def _container_exec(self, *, use_gpu: bool) -> str:
        """`apptainer exec ...` prefix, up to (not including) the image path.

        Factored out so per-site SLURM subclasses (e.g. NERSC, where the
        apptainer binary isn't on PATH and `--nv` must be gated on GPU) can
        swap the container-runtime invocation without re-implementing the
        whole render body. The base SLURM behaviour is unchanged: always
        `apptainer exec --nv` with the cluster bind list plus dev-loop binds.
        """
        binds = self.apptainer_binds
        dev = _dev_binds(self.env)
        dev_part = f" {dev}" if dev else ""
        return f"apptainer exec --nv -B {binds}{dev_part}"

    def _common_header(self, *, partition: str, job_name: str,
                       cell_dir: Path, log_stem: str,
                       cpus: str, gpus: str, memory: str, time: str,
                       array_spec: Optional[str] = None) -> str:
        # `log_stem` is the prefix passed to `#SBATCH --output=...-%j.out`.
        # siren / smax use `job-NNN`; dataprod uses `job_NNNNNN` (verify_jobs.py
        # SBATCH_RE depends on this distinction). When `array_spec` is set the
        # job is a SLURM array (one submission, one task per sub-job) and logs
        # switch to the array-aware `%A_%a` suffix.
        gpu_line = f"#SBATCH --gpus={gpus}\n" if gpus != "0" else ""
        qos = self.env.get("SLURM_QOS")
        qos_line = f"#SBATCH --qos={qos}\n" if qos else ""
        array_line = f"#SBATCH --array={array_spec}\n" if array_spec else ""
        log_suffix = "%A_%a" if array_spec else "%j"
        return (
            f"#!/bin/bash\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --account={self.env['SLURM_ACCOUNT']}\n"
            f"{qos_line}"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={cell_dir}/{log_stem}-{log_suffix}.out\n"
            f"#SBATCH --error={cell_dir}/{log_stem}-{log_suffix}.err\n"
            f"{array_line}"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"{gpu_line}"
            f"#SBATCH --mem={memory}\n"
            f"#SBATCH --time={time}\n"
            f"\n"
            f"set -eu -o pipefail\n"
            f"echo \"SLURM Job ID: ${{SLURM_JOB_ID}}\"\n"
            f"echo \"Job started:  $(date)\"\n"
            f"echo \"Node:         $(hostname)\"\n"
        )

    def _siren_or_smax(self, *, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                       job_name: str, job_id: int, partition: str,
                       use_gpu: bool, array_spec: Optional[str] = None) -> str:
        # Uses 3-digit `job-{job_id:03d}` log naming, matching the cell-based
        # workflows (siren_inputs and smax). In array mode one submission covers
        # the whole cell: the per-task `--job-id` comes from SLURM_ARRAY_TASK_ID
        # so each task still writes its own output_job_<id>.root (merge.sh-ready).
        cpus = self.env.get("DEFAULT_CPUS", "1")
        memory = self.env.get("DEFAULT_MEMORY", "16000")
        time = self.env.get("DEFAULT_TIME", "08:00:00")
        gpus = "1" if use_gpu else self.env.get("DEFAULT_GPUS", "0")
        log_stem = "job" if array_spec else f"job-{job_id:03d}"
        header = self._common_header(
            partition=partition, job_name=job_name, cell_dir=cell_dir,
            log_stem=log_stem,
            cpus=cpus, gpus=gpus, memory=memory, time=time,
            array_spec=array_spec,
        )
        job_id_expr = "${SLURM_ARRAY_TASK_ID}" if array_spec else str(job_id)
        body = (
            f"\nexport APPTAINERENV_GENIE_XSEC_FILE={self.env.get('GENIE_XSEC_FILE', '')}\n"
            f"{self._container_exec(use_gpu=use_gpu)} \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    lucid-run-job \\\n"
            f"        --config \"{cell_cfg}\" \\\n"
            f"        --output-dir \"{cell_dir}\" \\\n"
            f"        --job-id {job_id_expr} \\\n"
            f"        --skip-lucid \\\n"
            f"        --override-energy-MeV {energy_mev}\n"
            f"\necho \"Job ended: $(date)\"\n"
        )
        return header + body

    def render_siren_cell(self, *, cell_dir, cell_cfg, energy_mev, job_name,
                          job_id, partition, use_gpu, array_spec=None):
        return self._siren_or_smax(
            cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
            job_name=job_name, job_id=job_id, partition=partition,
            use_gpu=use_gpu, array_spec=array_spec,
        )

    def render_smax_cell(self, *, cell_dir, cell_cfg, energy_mev, job_name,
                         job_id, partition, use_gpu):
        return self._siren_or_smax(
            cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
            job_name=job_name, job_id=job_id, partition=partition,
            use_gpu=use_gpu,
        )

    def render_dataprod_job(self, *, cell_dir, config_path, detector, job_id,
                            job_name, partition, use_gpu, test, skip_lucid,
                            n_events, override_energy_mev,
                            sn_model=None, sn_ordering=None, master_seed=None,
                            lucid_seed=None):
        # Uses 6-digit `job_{job_id}` log naming, matching the dataprod
        # convention (verify_jobs.py SBATCH_RE matches this).
        cpus = self.env.get("DEFAULT_CPUS", "1")
        memory = self.env.get("DEFAULT_MEMORY", "16000")
        time = self.env.get("DEFAULT_TIME", "08:00:00")
        gpus = "1" if use_gpu else self.env.get("DEFAULT_GPUS", "0")
        header = self._common_header(
            partition=partition, job_name=job_name, cell_dir=cell_dir,
            log_stem=f"job_{job_id}",   # job_id is a zero-padded string for dataprod
            cpus=cpus, gpus=gpus, memory=memory, time=time,
        )
        flags = []
        if test:
            flags.append("--test")
        if skip_lucid:
            flags.append("--skip-lucid")
        if n_events is not None:
            flags.extend(["--n-events", str(n_events)])
        if override_energy_mev is not None:
            flags.extend(["--override-energy-MeV", str(override_energy_mev)])
        if sn_model is not None:
            flags.extend(["--sn-model", str(sn_model)])
        if sn_ordering is not None:
            flags.extend(["--sn-ordering", str(sn_ordering)])
        if master_seed is not None:
            flags.extend(["--master-seed", str(master_seed)])
        if lucid_seed is not None:
            flags.extend(["--lucid-seed", str(lucid_seed)])
        flag_str = " ".join(flags)
        # Supernova jobs resolve sntools from a CFS PYTHONUSERBASE (SN_ENV_BASE)
        # bound into the container Python — mirrors LUCID_ENV_BASE for training.
        sn_env_line = ""
        if sn_model is not None and self.env.get("SN_ENV_BASE"):
            sn_env_line = (
                f"export APPTAINERENV_PYTHONUSERBASE=\"{self.env['SN_ENV_BASE']}\"\n")
        body = (
            f"\n# Unified container: GEANT4 + ROOT + GENIE + PhotonSim + LUCiD.\n"
            f"export APPTAINERENV_GENIE_XSEC_FILE={self.env.get('GENIE_XSEC_FILE', '')}\n"
            f"{sn_env_line}"
            f"{self._container_exec(use_gpu=use_gpu)} \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    lucid-run-job \\\n"
            f"        --config \"{config_path}\" \\\n"
            f"        --output-dir \"{cell_dir}\" \\\n"
            f"        --detector {detector} \\\n"
            f"        --job-id {job_id} {flag_str}\n"
            f"\necho \"Job ended: $(date)\"\n"
        )
        return header + body

    def render_command_job(self, *, command, cell_dir, job_name, log_stem,
                           partition, use_gpu=False, request_disk_mb=4096,
                           request_memory_mb=0, gpus=None):
        header = self._common_header(
            partition=partition, job_name=job_name, cell_dir=cell_dir,
            log_stem=log_stem,
            cpus=self.env.get("DEFAULT_CPUS", "1"),
            gpus=gpus or ("1" if use_gpu else self.env.get("DEFAULT_GPUS", "0")),
            memory=(str(request_memory_mb) if request_memory_mb
                    else self.env.get("DEFAULT_MEMORY", "16000")),
            time=self.env.get("DEFAULT_TIME", "08:00:00"),
        )
        return header + (
            f"\n{self._container_exec(use_gpu=use_gpu)} \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    bash -c '{command}'\n"
            f"\necho \"Job ended: $(date)\"\n"
        )

    def render_train_run(self, *, run_dir, run_name, cli_args, resources):
        gpus = resources.get("gpus", "1")
        cpus = resources.get("cpus", "4")
        memory = resources.get("memory", "32000")
        time = resources.get("time", "04:00:00")
        # Use a single-segment "job" log stem; train_siren doesn't carry job_id.
        out = (
            f"#!/bin/bash\n"
            f"#SBATCH --partition={resources['partition']}\n"
            f"#SBATCH --account={resources['account']}\n"
            f"#SBATCH --job-name=train_{run_name}\n"
            f"#SBATCH --output={run_dir}/job-%j.out\n"
            f"#SBATCH --error={run_dir}/job-%j.err\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"#SBATCH --gpus={gpus}\n"
            f"#SBATCH --mem={memory}\n"
            f"#SBATCH --time={time}\n"
            f"\n"
            f"set -eu -o pipefail\n"
            f"echo \"SLURM Job ID: ${{SLURM_JOB_ID}}\"\n"
            f"echo \"Job started:  $(date)\"\n"
            f"echo \"Node:         $(hostname)\"\n"
            f"\n"
            f"# The user site holds the CUDA jax plugin (the image's jaxlib is CPU-only).\n"
            f"# PYTHONPATH puts /opt/LUCiD (the image's LUCiD, or LUCID_DEV_PATH bound\n"
            f"# there) ahead of it, so a LUCiD installed in that user site is never the\n"
            f"# one that runs.\n"
            f"export APPTAINERENV_PYTHONUSERBASE=\"{self.env['LUCID_ENV_BASE']}\"\n"
            f"export APPTAINERENV_PYTHONPATH=\"/opt/LUCiD\"\n"
            f"\n"
            f"{self._container_exec(use_gpu=True)} \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    python3 -m lucid.siren.training.train {cli_args}\n"
            f"\n"
            f"echo \"Job ended: $(date)\"\n"
        )
        return out


# --- NERSC / Perlmutter ------------------------------------------------------
# NERSC / Perlmutter cluster adapter.
#
# Perlmutter runs SLURM, so this subclasses :class:`SlurmAdapter` and reuses
# all of its planning/fan-out logic. Only three things differ from the SLAC
# (S3DF) SLURM deployment, and they are the only methods overridden here:
#
# 1. **Submit directives.** Perlmutter does not take ``--partition``; it uses
#    ``--qos`` + ``--constraint=cpu|gpu``, and GPU jobs must charge the ``_g``
#    variant of the allocation (e.g. ``dune`` for CPU, ``dune_g`` for GPU).
# 2. **Container runtime.** There is no system apptainer; the working binary is
#    the cvmfs OSG build (``APPTAINER_BIN``). ``--nv`` is gated on GPU so the
#    CPU-default jobs don't trip over a missing nvidia driver on Milan nodes.
# 3. **Bind list.** NERSC global filesystems, not ``/sdf``.
#
# Selection is sticky via ``CLUSTER=nersc`` in ``user_paths.sh``.

# cvmfs OSG apptainer — the only container runtime that runs the .sif on
# Perlmutter (no system apptainer/singularity; shifter/podman-hpc use a
# different exec surface). Overridable via APPTAINER_BIN in user_paths.sh.
DEFAULT_APPTAINER_BIN = (
    "/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer"
)

# Global filesystems a Perlmutter batch node needs visible inside the
# container: CFS (image + outputs), home (configs / dev checkouts), scratch,
# the DVS read-only mount, and cvmfs (optional cvmfs-staged GENIE splines).
NERSC_DEFAULT_BINDS = "/global/cfs,/global/homes,/global/u1,/pscratch,/dvs_ro,/cvmfs"


class NerscAdapter(SlurmAdapter):
    name = "nersc"
    # submit_extension / submit_cmd inherited from SlurmAdapter ("sbatch").

    def _default_binds(self) -> str:
        return NERSC_DEFAULT_BINDS

    def _container_exec(self, *, use_gpu: bool) -> str:
        apptainer = self.env.get("APPTAINER_BIN") or DEFAULT_APPTAINER_BIN
        dev = _dev_binds(self.env)
        dev_part = f" {dev}" if dev else ""
        nv = " --nv" if use_gpu else ""
        return f"{apptainer} exec{nv} -B {self.apptainer_binds}{dev_part}"

    def _common_header(self, *, partition: str, job_name: str,
                       cell_dir: Path, log_stem: str,
                       cpus: str, gpus: str, memory: str, time: str,
                       array_spec: str | None = None) -> str:
        # `partition` is ignored on Perlmutter (no --partition); cpu/gpu is
        # selected by --constraint, and the SlurmAdapter render methods encode
        # GPU-ness in `gpus` ("0" => CPU job). `array_spec`, when set, makes this
        # a SLURM array (one submission per cell) with `%A_%a` log naming.
        is_gpu = gpus != "0"
        constraint = "gpu" if is_gpu else "cpu"
        qos = self.env.get("SLURM_QOS", "shared")
        if is_gpu:
            account = (self.env.get("SLURM_ACCOUNT_GPU")
                       or f"{self.env.get('SLURM_ACCOUNT', '')}_g")
        else:
            account = self.env.get("SLURM_ACCOUNT", "")
        gpu_line = f"#SBATCH --gpus={gpus}\n" if is_gpu else ""
        array_line = f"#SBATCH --array={array_spec}\n" if array_spec else ""
        log_suffix = "%A_%a" if array_spec else "%j"
        return (
            f"#!/bin/bash\n"
            f"#SBATCH --qos={qos}\n"
            f"#SBATCH --constraint={constraint}\n"
            f"#SBATCH --account={account}\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={cell_dir}/{log_stem}-{log_suffix}.out\n"
            f"#SBATCH --error={cell_dir}/{log_stem}-{log_suffix}.err\n"
            f"{array_line}"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"{gpu_line}"
            f"#SBATCH --mem={memory}\n"
            f"#SBATCH --time={time}\n"
            f"\n"
            f"set -eu -o pipefail\n"
            f"echo \"SLURM Job ID: ${{SLURM_JOB_ID}}\"\n"
            f"echo \"Job started:  $(date)\"\n"
            f"echo \"Node:         $(hostname)\"\n"
        )

    def render_siren_pack(self, *, job_name: str, log_dir: Path, manifest: Path,
                          run_script: Path, n_workers: int, wall: str) -> str:
        """Whole-node packed SIREN job: one `regular` node runs many work units
        in parallel via the container's xargs worker pool (see run_script).

        Charges the CPU account on `--qos=regular --constraint=cpu` with
        `--exclusive` (the whole node), so a single submission drains a manifest
        of units instead of fanning out one shared sub-job each. Each unit still
        writes its own output_job_<id>.root, so merge.sh/recovery are unchanged.
        """
        account = self.env.get("SLURM_ACCOUNT", "")
        genie = self.env.get("GENIE_XSEC_FILE", "")
        return (
            f"#!/bin/bash\n"
            f"#SBATCH --qos=regular\n"
            f"#SBATCH --constraint=cpu\n"
            f"#SBATCH --account={account}\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={log_dir}/{job_name}-%j.out\n"
            f"#SBATCH --error={log_dir}/{job_name}-%j.err\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --exclusive\n"
            f"#SBATCH --time={wall}\n"
            f"\n"
            f"set -eu -o pipefail\n"
            f"echo \"SLURM Job ID: ${{SLURM_JOB_ID}}\"\n"
            f"echo \"Job started:  $(date)\"\n"
            f"echo \"Node:         $(hostname)\"\n"
            f"echo \"Workers:      {n_workers}\"\n"
            f"\n"
            f"export APPTAINERENV_GENIE_XSEC_FILE={genie}\n"
            f"{self._container_exec(use_gpu=False)} \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    bash {run_script} {manifest} {n_workers}\n"
            f"\necho \"Job ended: $(date)\"\n"
        )

    def render_train_run(self, *, run_dir, run_name, cli_args, resources):
        # Perlmutter's GPU jobs need --constraint=gpu and the _g account; not wired yet. Fail
        # loudly rather than emit a sbatch that calls a bare `apptainer` not on PATH.
        raise NotImplementedError(
            "train_siren is not yet wired for NERSC/Perlmutter: its GPU job needs "
            "--constraint=gpu and the SLURM_ACCOUNT_GPU account "
            "(docs/guides/production/clusters.md, NERSC).")


# --- HTCondor (LXPLUS) --------------------------------------------------------
# HTCondor adapter for LXPLUS-style batch.
#
# Submit-description model
# ------------------------
#
# We use HTCondor's *transparent* container integration:
#
#     universe              = vanilla
#     executable            = /bin/bash
#     arguments             = "-c 'lucid-run-job --config ... --output-dir ...'"
#     MY.SingularityImage   = "/eos/.../lucid.sif"
#     MY.SingularityBind    = "/afs,/eos,/cvmfs,/host/LUCiD:/opt/LUCiD"
#
# The STARTD wraps `/bin/bash -c '<job>'` in `apptainer exec` against the
# named image with the named binds, so the job sees `lucid-run-job` on
# PATH inside the container.
#
# Two LXPLUS quirks this avoids:
#
#   * Direct `executable = /usr/bin/apptainer` falls foul of LXPLUS
#     worker-node apptainer config quirks (e.g. apptainer searches for
#     its conf relative to `/pool/condor` because of the scratch dir
#     overlay).
#   * Standard LXPLUS schedds (bigbird*) reject submit descriptions whose
#     `output`, `error`, `log`, or `executable` paths point at `/eos`.
#     We honour this by directing the .sub-level log fields to
#     `LOG_BASE_PATH` from `user_paths.sh` (defaults to AFS), while the
#     actual ROOT outputs land where `--output-dir` says (EOS-friendly).
#
# Walltime is set via `+JobFlavour` (a CERN HTCondor convention):
#   espresso 20m | microcentury 1h | longlunch 2h |
#   workday 8h  | tomorrow 1d   | testmatch 3d | nextweek 1w
#
# Each sub-job is its own `.sub`, mirroring the SLURM one-file-per-sub-job
# model so `resubmit_failed` logic transposes 1:1.

class HTCondorAdapter(ClusterAdapter):
    name = "htcondor"
    submit_extension = "sub"
    submit_cmd = "condor_submit"

    def _default_binds(self) -> str:
        return HTCONDOR_DEFAULT_BINDS

    def _flavour(self) -> str:
        return self.env.get("CONDOR_JOB_FLAVOUR", "workday")

    def _train_flavour(self) -> str:
        return self.env.get("CONDOR_TRAIN_FLAVOUR", "testmatch")

    def _gpu_lines(self, gpus: str) -> str:
        return f"request_gpus            = {gpus}\n" if gpus != "0" else ""

    def _binds_for_classad(self) -> str:
        """Comma-separated bind list for `MY.SingularityBind`.

        Includes the base binds plus any LUCID_DEV_PATH / PHOTONSIM_DEV_PATH
        shadows so the host checkout overlays the baked container copy.
        """
        parts: List[str] = [self.apptainer_binds]
        if self.env.get("LUCID_DEV_PATH"):
            parts.append(f"{self.env['LUCID_DEV_PATH']}:/opt/LUCiD")
        if self.env.get("PHOTONSIM_DEV_PATH"):
            parts.append(f"{self.env['PHOTONSIM_DEV_PATH']}:/opt/PhotonSim")
        return ",".join(parts)

    def _env_block(self) -> str:
        """`environment = "..."` for the submit description.

        Only used for vars HTCondor reliably propagates into the
        container's process env. PhotonSim binary override is inlined
        into the bash command instead — see `_env_prefix()`.
        """
        return f"GENIE_XSEC_FILE={self.env.get('GENIE_XSEC_FILE', '')}"

    def _env_prefix(self) -> str:
        """Bash-shell-scope env assignments to prepend to the inner command.

        Used because HTCondor's transparent Singularity wrapper doesn't
        forward submit-description `environment = "..."` vars into the
        container's process env reliably (it forwards only `_CONDOR_*`
        plus a fixed allow-list). Inlining `KEY=val cmd ...` sets the
        var in the bash shell that runs *inside* the container, which
        always works.

        Both overrides work around the same nested-bind issue: HTCondor's
        Singularity silently ignores a `<host>:/opt/...` bind once `/afs`
        is already bound at the top level, leaving the image's baked copy
        in effect. We reach the host checkouts by absolute path via /afs
        (unconditionally bound) instead of trusting the dropped bind:

          * PHOTONSIM_DEV_PATH -> PHOTONSIM_BIN points run_job at the
            host's freshly-built binary rather than baked /opt/PhotonSim.
          * LUCID_DEV_PATH -> PYTHONPATH prepends the host LUCiD checkout
            so the interpreter's PathFinder resolves `import lucid` to it
            ahead of the container's editable-install meta-path finder
            (which hard-maps `lucid` to the baked /opt/LUCiD).
        """
        parts: List[str] = []
        if self.env.get("LUCID_DEV_PATH"):
            parts.append(
                f"PYTHONPATH={self.env['LUCID_DEV_PATH']}"
                "${PYTHONPATH:+:$PYTHONPATH}"
            )
        if self.env.get("PHOTONSIM_DEV_PATH"):
            parts.append(
                f"PHOTONSIM_BIN={self.env['PHOTONSIM_DEV_PATH']}/build/PhotonSim"
            )
        return (" ".join(parts) + " ") if parts else ""

    def _log_dir(self, cell_dir: Path) -> Path:
        """Where the .sub's `output =`, `error =`, `log =` files land.

        Standard LXPLUS schedds reject `/eos` paths in these fields, so we
        re-root them under `LOG_BASE_PATH` (typically AFS) if set. Falls
        back to `cell_dir` for setups where EOS-in-submit is allowed
        (EosSubmit schedds, or non-LXPLUS HTCondor sites).
        """
        log_base = self.env.get("LOG_BASE_PATH")
        out_base = self.env.get("OUTPUT_BASE_PATH")
        if not log_base or not out_base:
            return cell_dir
        try:
            rel = cell_dir.resolve().relative_to(Path(out_base).resolve())
        except ValueError:
            return cell_dir
        log_dir = Path(log_base) / rel
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _cell_submit(self, *, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                      job_name: str, job_id: int, partition: str,
                      use_gpu: bool, extra_run_flags: List[str],
                      log_stem: str) -> str:
        """Shared body for siren/smax cell sub-jobs."""
        cpus = self.env.get("DEFAULT_CPUS", "1")
        memory_mb = self.env.get("DEFAULT_MEMORY", "4000")
        gpus = "1" if use_gpu else self.env.get("DEFAULT_GPUS", "0")
        flavour = partition if partition else self._flavour()
        log_dir = self._log_dir(cell_dir)

        # `arguments` is a single-line bash -c command. Internal single
        # quotes work because HTCondor's "new-syntax" arguments treats the
        # whole thing as one string when wrapped in double quotes.
        run_args = " ".join(extra_run_flags)
        bash_cmd = (
            f"{self._env_prefix()}lucid-run-job "
            f"--config {cell_cfg} "
            f"--output-dir {cell_dir} "
            f"{run_args}"
        )
        return (
            f"# Auto-generated by lucid.production.platform.adapters\n"
            f"universe                = vanilla\n"
            f"executable              = /bin/bash\n"
            f"arguments               = \"-c '{bash_cmd}'\"\n"
            f"environment             = \"{self._env_block()}\"\n"
            f"output                  = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).out\n"
            f"error                   = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).err\n"
            f"log                     = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).log\n"
            f"MY.SingularityImage     = \"{self.env['LUCID_IMAGE_PATH']}\"\n"
            f"MY.SingularityBind      = \"{self._binds_for_classad()}\"\n"
            f"+JobBatchName           = \"{job_name}\"\n"
            f"request_cpus            = {cpus}\n"
            f"request_memory          = {memory_mb}\n"
            f"request_disk            = 4096\n"
            f"{self._gpu_lines(gpus)}"
            f"+JobFlavour             = \"{flavour}\"\n"
            f"should_transfer_files   = NO\n"
            f"queue\n"
        )

    def render_siren_cell(self, *, cell_dir, cell_cfg, energy_mev, job_name,
                          job_id, partition, use_gpu, array_spec=None):
        if array_spec is not None:
            raise NotImplementedError(
                "SLURM job-array mode (--array) has no HTCondor equivalent here; "
                "submit per-job on HTCondor (omit --array)."
            )
        return self._cell_submit(
            cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
            job_name=job_name, job_id=job_id, partition=partition,
            use_gpu=use_gpu,
            extra_run_flags=[
                f"--job-id {job_id}",
                "--skip-lucid",
                f"--override-energy-MeV {energy_mev}",
            ],
            log_stem=f"job-{job_id:03d}",
        )

    def render_smax_cell(self, *, cell_dir, cell_cfg, energy_mev, job_name,
                         job_id, partition, use_gpu):
        return self._cell_submit(
            cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
            job_name=job_name, job_id=job_id, partition=partition,
            use_gpu=use_gpu,
            extra_run_flags=[
                f"--job-id {job_id}",
                "--skip-lucid",
                f"--override-energy-MeV {energy_mev}",
            ],
            log_stem=f"job-{job_id:03d}",
        )

    def render_dataprod_job(self, *, cell_dir, config_path, detector, job_id,
                            job_name, partition, use_gpu, test, skip_lucid,
                            n_events, override_energy_mev,
                            sn_model=None, sn_ordering=None, master_seed=None,
                            lucid_seed=None):
        cpus = self.env.get("DEFAULT_CPUS", "1")
        memory_mb = self.env.get("DEFAULT_MEMORY", "4000")
        gpus = "1" if use_gpu else self.env.get("DEFAULT_GPUS", "0")
        flavour = partition if partition else self._flavour()
        log_dir = self._log_dir(cell_dir)

        flags = [f"--detector {detector}", f"--job-id {job_id}"]
        if test:
            flags.append("--test")
        if skip_lucid:
            flags.append("--skip-lucid")
        if n_events is not None:
            flags.append(f"--n-events {n_events}")
        if override_energy_mev is not None:
            flags.append(f"--override-energy-MeV {override_energy_mev}")
        if sn_model is not None:
            flags.append(f"--sn-model {sn_model}")
        if sn_ordering is not None:
            flags.append(f"--sn-ordering {sn_ordering}")
        if master_seed is not None:
            flags.append(f"--master-seed {master_seed}")
        if lucid_seed is not None:
            flags.append(f"--lucid-seed {lucid_seed}")
        run_args = " ".join(flags)
        bash_cmd = (
            f"{self._env_prefix()}lucid-run-job --config {config_path} "
            f"--output-dir {cell_dir} {run_args}"
        )
        return (
            f"# Auto-generated by lucid.production.platform.adapters\n"
            f"universe                = vanilla\n"
            f"executable              = /bin/bash\n"
            f"arguments               = \"-c '{bash_cmd}'\"\n"
            f"environment             = \"{self._env_block()}\"\n"
            f"output                  = {log_dir}/job_{job_id}-$(ClusterId).$(ProcId).out\n"
            f"error                   = {log_dir}/job_{job_id}-$(ClusterId).$(ProcId).err\n"
            f"log                     = {log_dir}/job_{job_id}-$(ClusterId).$(ProcId).log\n"
            f"MY.SingularityImage     = \"{self.env['LUCID_IMAGE_PATH']}\"\n"
            f"MY.SingularityBind      = \"{self._binds_for_classad()}\"\n"
            f"+JobBatchName           = \"{job_name}\"\n"
            f"request_cpus            = {cpus}\n"
            f"request_memory          = {memory_mb}\n"
            f"request_disk            = 4096\n"
            f"{self._gpu_lines(gpus)}"
            f"+JobFlavour             = \"{flavour}\"\n"
            f"should_transfer_files   = NO\n"
            f"queue\n"
        )

    def render_command_job(self, *, command, cell_dir, job_name, log_stem,
                           partition, use_gpu=False, request_disk_mb=4096,
                           request_memory_mb=0, gpus=None):
        gpus = gpus or ("1" if use_gpu else self.env.get("DEFAULT_GPUS", "0"))
        log_dir = self._log_dir(cell_dir)
        # The env prefix carries the dev-checkout overrides HTCondor's
        # Singularity wrapper drops; see _env_prefix. It becomes an `export`
        # here rather than a command prefix because `command` may be a chain:
        # in `A=1 cmd1 && cmd2`, bash scopes A to cmd1 alone.
        prefix = self._env_prefix().strip()
        exports = f"export {prefix}; " if prefix else ""
        bash_cmd = f"{exports}{command}"
        return (
            f"# Auto-generated by lucid.production.platform.adapters\n"
            f"universe                = vanilla\n"
            f"executable              = /bin/bash\n"
            f"arguments               = \"-c '{bash_cmd}'\"\n"
            f"environment             = \"{self._env_block()}\"\n"
            f"output                  = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).out\n"
            f"error                   = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).err\n"
            f"log                     = {log_dir}/{log_stem}-$(ClusterId).$(ProcId).log\n"
            f"MY.SingularityImage     = \"{self.env['LUCID_IMAGE_PATH']}\"\n"
            f"MY.SingularityBind      = \"{self._binds_for_classad()}\"\n"
            f"+JobBatchName           = \"{job_name}\"\n"
            f"request_cpus            = {self.env.get('DEFAULT_CPUS', '1')}\n"
            f"request_memory          = "
            f"{request_memory_mb or self.env.get('DEFAULT_MEMORY', '4000')}\n"
            f"request_disk            = {request_disk_mb}\n"
            f"{self._gpu_lines(gpus)}"
            f"+JobFlavour             = \"{partition if partition else self._flavour()}\"\n"
            f"should_transfer_files   = NO\n"
            f"queue\n"
        )

    def render_train_run(self, *, run_dir, run_name, cli_args, resources):
        # Reads cpus/memory/gpus/flavour from `resources`; partition and account are Slurm's.
        cpus = resources.get("cpus", "4")
        memory_mb = resources.get("memory", "32000")
        gpus = resources.get("gpus", "1")
        flavour = resources.get("flavour") or self._train_flavour()
        log_dir = self._log_dir(Path(run_dir))

        return (
            f"# Auto-generated by lucid.production.platform.adapters\n"
            f"universe                = vanilla\n"
            f"executable              = /bin/bash\n"
            f"arguments               = \"-c '{self._env_prefix()}"
            f"{'' if self.env.get('LUCID_DEV_PATH') else 'PYTHONPATH=/opt/LUCiD '}"
            f"python3 -m lucid.siren.training.train {cli_args}'\"\n"
            f"environment             = "
            f"\"PYTHONUSERBASE={self.env['LUCID_ENV_BASE']}\"\n"
            f"output                  = {log_dir}/job-$(ClusterId).$(ProcId).out\n"
            f"error                   = {log_dir}/job-$(ClusterId).$(ProcId).err\n"
            f"log                     = {log_dir}/job-$(ClusterId).$(ProcId).log\n"
            f"MY.SingularityImage     = \"{self.env['LUCID_IMAGE_PATH']}\"\n"
            f"MY.SingularityBind      = \"{self._binds_for_classad()}\"\n"
            f"+JobBatchName           = \"train_{run_name}\"\n"
            f"request_cpus            = {cpus}\n"
            f"request_memory          = {memory_mb}\n"
            f"request_disk            = 16384\n"
            f"request_gpus            = {gpus}\n"
            f"+JobFlavour             = \"{flavour}\"\n"
            f"should_transfer_files   = NO\n"
            f"queue\n"
        )


# --- Factory ------------------------------------------------------------------

ADAPTERS: Dict[str, type] = {
    "slurm": SlurmAdapter,
    "nersc": NerscAdapter,
    "htcondor": HTCondorAdapter,
}


def get_adapter(env: Dict[str, str]) -> ClusterAdapter:
    """Pick the adapter based on `CLUSTER` env var (default: 'slurm')."""
    name = (env.get("CLUSTER") or "slurm").lower()
    cls = ADAPTERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown CLUSTER={name!r}. Known adapters: {sorted(ADAPTERS)}.")
    return cls(env)
