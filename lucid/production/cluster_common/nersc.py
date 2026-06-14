"""NERSC / Perlmutter cluster adapter.

Perlmutter runs SLURM, so this subclasses :class:`SlurmAdapter` and reuses
all of its planning/fan-out logic. Only three things differ from the SLAC
(S3DF) SLURM deployment, and they are the only methods overridden here:

1. **Submit directives.** Perlmutter does not take ``--partition``; it uses
   ``--qos`` + ``--constraint=cpu|gpu``, and GPU jobs must charge the ``_g``
   variant of the allocation (e.g. ``dune`` for CPU, ``dune_g`` for GPU).
2. **Container runtime.** There is no system apptainer; the working binary is
   the cvmfs OSG build (``APPTAINER_BIN``). ``--nv`` is gated on GPU so the
   CPU-default jobs don't trip over a missing nvidia driver on Milan nodes.
3. **Bind list.** NERSC global filesystems, not ``/sdf``.

Selection is sticky via ``CLUSTER=nersc`` in ``user_paths.sh``. Registering
this module (an ``import`` from any entrypoint) makes that value resolvable.
"""

from __future__ import annotations

from pathlib import Path

from .cluster import SlurmAdapter, _dev_binds, register_adapter

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

    def render_train_run(self, *, run_dir, run_name, cli_args, slurm):
        # train_siren's GPU/SLURM defaults (partition=roma, account=mli:cider-ml)
        # are SLAC values and aren't wired for Perlmutter yet. Fail loudly
        # rather than emit a sbatch that calls a bare `apptainer` not on PATH.
        raise NotImplementedError(
            "train_siren is not yet wired for NERSC/Perlmutter. "
            "See LUCiD/docs/QUICKSTART_NERSC.md (train_siren follow-up): the "
            "GPU defaults in jobs/train_siren/generate_jobs.py still carry SLAC "
            "values (partition=roma, account=mli:cider-ml) and need dune_g + "
            "--constraint=gpu before GPU training can run here."
        )


register_adapter("nersc", NerscAdapter)
