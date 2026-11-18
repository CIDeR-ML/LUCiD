"""Unit test for the cluster adapters' arbitrary-command job
(`lucid.production.platform.adapters.ClusterAdapter.render_command_job`)."""
from pathlib import Path

from lucid.production.platform.adapters import HTCondorAdapter


def test_command_job_exports_env_for_the_whole_chain():
    """`A=1 cmd1 && cmd2` scopes A to cmd1 in bash, which silently broke the
    reduction step of the fiTQun profile job: PhotonSim ran, then the `python -m
    lucid...` that follows it came up without PYTHONPATH. The dev-checkout
    overrides must be exported, not prefixed."""
    adapter = HTCondorAdapter({
        "LUCID_IMAGE_PATH": "/img.sif",
        "LUCID_DEV_PATH": "/dev/LUCiD",
        "PHOTONSIM_DEV_PATH": "/dev/PhotonSim",
        "LOG_BASE_PATH": "/logs",
    })
    body = adapter.render_command_job(
        command="first && second", cell_dir=Path("/out/cell"),
        job_name="j", log_stem="s", partition="")

    args = next(l for l in body.splitlines() if l.startswith("arguments"))
    assert "export PYTHONPATH=/dev/LUCiD" in args
    assert "PHOTONSIM_BIN=/dev/PhotonSim/build/PhotonSim;" in args
    assert args.index("export") < args.index("first && second")
    # HTCondor's arguments is itself a quoted string; an embedded double quote
    # would need doubling, so the command must not carry one.
    assert '"' not in args[args.index("-c"):-1]


def test_slurm_jobs_carry_the_site_qos_and_cpu_jobs_no_gpu():
    """A CPU partition that the account reaches only at a preemptable QOS (S3DF milano) needs
    `--qos` in the job, and a CPU-only job must not inherit a GPU from DEFAULT_GPUS."""
    from lucid.production.platform.adapters import SlurmAdapter
    env = {"CLUSTER": "slurm", "LUCID_IMAGE_PATH": "/img.sif", "SLURM_ACCOUNT": "fac:repo",
           "APPTAINER_BINDS": "/sdf", "DEFAULT_GPUS": "1"}
    job = dict(command="true", cell_dir=Path("/out/cell"), job_name="j", log_stem="s",
               partition="milano")
    body = SlurmAdapter({**env, "SLURM_QOS": "preemptable"}).render_command_job(**job, gpus="0")
    assert "#SBATCH --qos=preemptable\n" in body and "#SBATCH --gpus" not in body
    body = SlurmAdapter(env).render_command_job(**job)
    assert "--qos" not in body and "#SBATCH --gpus=1\n" in body   # unchanged when not asked


def test_a_training_job_runs_the_image_or_dev_lucid_not_a_user_site_one():
    """LUCID_ENV_BASE is a Python user site for the CUDA jax plugin. One that also carries an
    editable LUCiD (S3DF's shared one points at a home checkout) would shadow the image's
    /opt/LUCiD and LUCID_DEV_PATH, since a user site comes before site-packages; and an image's
    console script may name an old module. The job runs the module with /opt/LUCiD first."""
    from lucid.production.platform.adapters import HTCondorAdapter, SlurmAdapter
    res = {"partition": "ampere", "account": "fac:repo"}
    base = {"LUCID_IMAGE_PATH": "/img.sif", "LUCID_ENV_BASE": "/env", "APPTAINER_BINDS": "/sdf"}
    body = SlurmAdapter({**base, "CLUSTER": "slurm"}).render_train_run(
        run_dir=Path("/r"), run_name="s0", cli_args="--seed 0", resources=res)
    assert 'export APPTAINERENV_PYTHONPATH="/opt/LUCiD"' in body
    assert "python3 -m lucid.siren.training.train --seed 0" in body and "lucid-train-siren" not in body
    for dev, want in ((None, "PYTHONPATH=/opt/LUCiD python3 -m"), ("/dev/LUCiD", "PYTHONPATH=/dev/LUCiD")):
        env = {**base, "CLUSTER": "htcondor", **({"LUCID_DEV_PATH": dev} if dev else {})}
        sub = HTCondorAdapter(env).render_train_run(run_dir=Path("/r"), run_name="s0",
                                                    cli_args="--seed 0", resources={"flavour": "x"})
        args = next(l for l in sub.splitlines() if l.startswith("arguments"))
        assert want in args and "python3 -m lucid.siren.training.train --seed 0" in args
