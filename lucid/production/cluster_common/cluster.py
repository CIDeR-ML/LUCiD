"""Cluster adapter — the one place that knows SLURM vs HTCondor.

Each adapter owns the submit-description template, the submit command,
and the apptainer bind list for its cluster. Everything else (planning,
config emission, completion checks) is shared and lives elsewhere in
`cluster_common/`.

Selection is sticky via `CLUSTER=slurm|htcondor` in `user_paths.sh`.
Default is `slurm` to preserve backward compatibility with the existing
S3DF user_paths.sh that doesn't set the var.

To add a third cluster:

  1. Subclass `ClusterAdapter` and implement `render_*` methods plus the
     class-level constants.
  2. Register it in `_ADAPTERS` below.
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
    def render_train_run(self, *, run_dir: Path, run_name: str,
                         cli_args: str, slurm: Dict[str, str]) -> str: ...

    # --- monitor helpers (used by per-cluster monitor scripts) ---

    @abstractmethod
    def queue_status_cmd(self, user: str) -> List[str]: ...


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
        array_line = f"#SBATCH --array={array_spec}\n" if array_spec else ""
        log_suffix = "%A_%a" if array_spec else "%j"
        return (
            f"#!/bin/bash\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --account={self.env['SLURM_ACCOUNT']}\n"
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
                            sn_model=None, sn_ordering=None):
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

    def render_train_run(self, *, run_dir, run_name, cli_args, slurm):
        # train_siren has its own SLURM defaults (GPU job, longer wall),
        # passed in `slurm` (already merged with config overrides).
        gpus = slurm.get("gpus", "1")
        cpus = slurm.get("cpus", "4")
        memory = slurm.get("memory", "32000")
        time = slurm.get("time", "04:00:00")
        partition = slurm["partition"]
        # Use a single-segment "job" log stem; train_siren doesn't carry job_id.
        out = (
            f"#!/bin/bash\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --account={slurm.get('account', self.env.get('SLURM_ACCOUNT', ''))}\n"
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
            f"# Expose the isolated lucid user-site (jax-cuda12-plugin lives there)\n"
            f"# to the container, otherwise JAX falls back to CPU.\n"
            f"export APPTAINERENV_PYTHONUSERBASE=\"{self.env['LUCID_ENV_BASE']}\"\n"
            f"\n"
            f"apptainer exec --nv -B {self.apptainer_binds} \\\n"
            f"    -B {self.env['LUCID_DEV_PATH']}:/opt/LUCiD \\\n"
            f"    {self.env['LUCID_IMAGE_PATH']} \\\n"
            f"    lucid-train-siren {cli_args}\n"
            f"\n"
            f"echo \"Job ended: $(date)\"\n"
        )
        return out

    def queue_status_cmd(self, user: str) -> List[str]:
        return ["squeue", "-u", user]


# --- Factory ------------------------------------------------------------------

_ADAPTERS: Dict[str, type] = {
    "slurm": SlurmAdapter,
    # HTCondorAdapter registered when the htcondor module imports.
}


def register_adapter(name: str, cls: type) -> None:
    _ADAPTERS[name] = cls


def get_adapter(env: Dict[str, str]) -> ClusterAdapter:
    """Pick the adapter based on `CLUSTER` env var (default: 'slurm')."""
    name = (env.get("CLUSTER") or "slurm").lower()
    cls = _ADAPTERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown CLUSTER={name!r}. Known adapters: "
            f"{sorted(_ADAPTERS)}. Did you forget to import the adapter "
            f"module? (HTCondorAdapter lives in cluster_common.htcondor)"
        )
    return cls(env)
