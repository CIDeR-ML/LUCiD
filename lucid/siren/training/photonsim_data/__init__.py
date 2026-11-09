"""PhotonSim → SIREN data preparation utilities.

``build_tables`` turns per-energy PhotonSim ROOT files into the lookup tables the trainer reads
(``lucid.siren.training.dataset.PhotonSimDataset``): ``main_photon`` the photon table and
``main_dedx`` the dE/dx table, the ``lucid-build-photon-table`` and ``lucid-build-dedx-table``
entry points. See `docs/guides/production/siren-training-inputs.md` for the end-to-end pipeline.
"""
