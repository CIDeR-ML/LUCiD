"""HTCondor adapter for LXPLUS-style batch.

Submit-description model
------------------------

We use HTCondor's *transparent* container integration:

    universe              = vanilla
    executable            = /bin/bash
    arguments             = "-c 'lucid-run-job --config ... --output-dir ...'"
    MY.SingularityImage   = "/eos/.../lucid.sif"
    MY.SingularityBind    = "/afs,/eos,/cvmfs,/host/LUCiD:/opt/LUCiD"

The STARTD wraps `/bin/bash -c '<job>'` in `apptainer exec` against the
named image with the named binds, so the job sees `lucid-run-job` on
PATH inside the container.

Two LXPLUS quirks this avoids:

  * Direct `executable = /usr/bin/apptainer` falls foul of LXPLUS
    worker-node apptainer config quirks (e.g. apptainer searches for
    its conf relative to `/pool/condor` because of the scratch dir
    overlay).
  * Standard LXPLUS schedds (bigbird*) reject submit descriptions whose
    `output`, `error`, `log`, or `executable` paths point at `/eos`.
    We honour this by directing the .sub-level log fields to
    `LOG_BASE_PATH` from `user_paths.sh` (defaults to AFS), while the
    actual ROOT outputs land where `--output-dir` says (EOS-friendly).

Walltime is set via `+JobFlavour` (a CERN HTCondor convention):
  espresso 20m | microcentury 1h | longlunch 2h |
  workday 8h  | tomorrow 1d   | testmatch 3d | nextweek 1w

Each sub-job is its own `.sub`, mirroring the SLURM one-file-per-sub-job
model so `resubmit_failed` logic transposes 1:1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .cluster import (
    HTCONDOR_DEFAULT_BINDS, ClusterAdapter, register_adapter,
)


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

        When PHOTONSIM_DEV_PATH is set, override PHOTONSIM_BIN to point
        at the host's freshly-built binary via /afs (unconditionally
        bound). This bypasses the nested-bind issue where HTCondor's
        Singularity silently ignores `<host>:/opt/PhotonSim` once
        `/afs` is already bound at the top level, leaving the image's
        baked binary in effect.
        """
        if not self.env.get("PHOTONSIM_DEV_PATH"):
            return ""
        return f"PHOTONSIM_BIN={self.env['PHOTONSIM_DEV_PATH']}/build/PhotonSim "

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
            f"# Auto-generated by lucid.production.cluster_common.htcondor\n"
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
                            n_events, override_energy_mev):
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
        run_args = " ".join(flags)
        bash_cmd = (
            f"{self._env_prefix()}lucid-run-job --config {config_path} "
            f"--output-dir {cell_dir} {run_args}"
        )
        return (
            f"# Auto-generated by lucid.production.cluster_common.htcondor\n"
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

    def render_train_run(self, *, run_dir, run_name, cli_args, slurm):
        # `slurm` is the merged SLURM_DEFAULTS+config block; we ignore its
        # SLURM-specific entries and read cpus/memory/gpus from it.
        cpus = slurm.get("cpus", "4")
        memory_mb = slurm.get("memory", "32000")
        gpus = slurm.get("gpus", "1")
        flavour = slurm.get("flavour") or self._train_flavour()
        log_dir = self._log_dir(Path(run_dir))

        # train_siren wants LUCID_DEV_PATH bound to /opt/LUCiD; this is the
        # only adapter that hard-requires the dev-loop bind.
        binds = [self.apptainer_binds, f"{self.env['LUCID_DEV_PATH']}:/opt/LUCiD"]
        if self.env.get("PHOTONSIM_DEV_PATH"):
            binds.append(f"{self.env['PHOTONSIM_DEV_PATH']}:/opt/PhotonSim")

        return (
            f"# Auto-generated by lucid.production.cluster_common.htcondor\n"
            f"universe                = vanilla\n"
            f"executable              = /bin/bash\n"
            f"arguments               = \"-c 'lucid-train-siren {cli_args}'\"\n"
            f"environment             = "
            f"\"PYTHONUSERBASE={self.env['LUCID_ENV_BASE']}\"\n"
            f"output                  = {log_dir}/job-$(ClusterId).$(ProcId).out\n"
            f"error                   = {log_dir}/job-$(ClusterId).$(ProcId).err\n"
            f"log                     = {log_dir}/job-$(ClusterId).$(ProcId).log\n"
            f"MY.SingularityImage     = \"{self.env['LUCID_IMAGE_PATH']}\"\n"
            f"MY.SingularityBind      = \"{','.join(binds)}\"\n"
            f"+JobBatchName           = \"train_{run_name}\"\n"
            f"request_cpus            = {cpus}\n"
            f"request_memory          = {memory_mb}\n"
            f"request_disk            = 16384\n"
            f"request_gpus            = {gpus}\n"
            f"+JobFlavour             = \"{flavour}\"\n"
            f"should_transfer_files   = NO\n"
            f"queue\n"
        )

    def queue_status_cmd(self, user: str) -> List[str]:
        return ["condor_q", user, "-nobatch"]


register_adapter("htcondor", HTCondorAdapter)
