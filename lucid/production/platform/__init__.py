"""Execution plumbing shared by the production applications; no physics.

``adapters`` holds one ``ClusterAdapter`` per batch system (SLURM, NERSC, HTCondor) -- the
submit-description template, the submit command and the apptainer binds -- and ``get_adapter``
picks one from ``CLUSTER`` in ``user_paths.sh``; ``user_paths`` sources that file. The
per-site templates are in ``templates/``. See ``docs/guides/production/cluster-abstraction.md``.
"""
