"""Photon-shotgun production pipeline.

Entry point: ``lucid.simulation.shotgun.setup_shotgun_simulator`` for building
a simulator, then the helpers here for orchestration (sampling positions,
streaming outputs to HDF5 in chunks).

For a CLI run, use::

    python -m lucid.production.photon_shotgun.run --help
"""
from lucid.production.photon_shotgun.io import (
    sparsify_waveform, densify_waveform,
    save_shotgun_waveform, load_shotgun_waveform,
    save_shotgun_per_photon, load_shotgun_per_photon,
    StreamingWaveformWriter, StreamingPerPhotonWriter,
)
from lucid.production.photon_shotgun.utils import (
    read_detector_bounds,
    sample_positions_uniform, sample_directions_isotropic,
    build_case_sources, batched_source_iter,
)
