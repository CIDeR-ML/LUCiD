"""The forward simulator: photon source -> transport -> per-sensor hits.

``setup_event_simulator`` builds it (the hub every guide starts from). ``photon_step`` is one
transport iteration (``photon_step_volume`` for string telescopes), ``hits`` what the sensors
report (the registry ``HITS``: one function per hits and detection), ``reflection`` the
reflection models, ``optics`` the scattering and direction helpers, and ``PhotonState`` the
per-photon carry. The data-production stages
after it, ``digitizer``, ``trigger`` and ``readout``, are imported from their modules.
"""
from lucid.simulation.simulator import setup_event_simulator
from lucid.simulation.optics import (
    normalize, compute_reflection_direction,
    create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere,
)
from lucid.simulation.photon_step import photon_iteration_sample, photon_iteration_update_factors
from lucid.simulation.hits import (
    HITS, FirstArrival, PhotonList, SegmentTruth, Waveform, PhotonHits,
    first_arrival_expected, first_arrival_sampled, photon_list_expected, segment_truth_sampled,
    waveform_expected, waveform_sampled, photon_hits_sampled,
)
from lucid.simulation.types import PhotonState

__all__ = [
    'setup_event_simulator',
    'normalize', 'compute_reflection_direction', 'create_local_frame', 'sample_scatter_distance',
    'solve_rayleigh_inverse_cdf', 'compute_scatter_direction', 'sample_cosine_hemisphere',
    'photon_iteration_sample', 'photon_iteration_update_factors',
    'HITS', 'FirstArrival', 'PhotonList', 'SegmentTruth', 'Waveform', 'PhotonHits',
    'first_arrival_expected', 'first_arrival_sampled', 'photon_list_expected',
    'segment_truth_sampled', 'waveform_expected', 'waveform_sampled', 'photon_hits_sampled',
    'PhotonState',
]
