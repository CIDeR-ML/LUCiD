"""JUNO laser simulation and per-PE calibration adapters."""

from .config import (
    JunoLaserConfig,
    PARAMETER_NAMES,
    POSITIVE_PARAMETER_NAMES,
)
from .eventwise import (
    EventwisePeObservation,
    gaussian_mixture_rate,
    gaussian_window_diagnostics,
    gaussian_window_probability,
    make_host_accumulated_gaussian_eventwise_objective,
    make_gaussian_eventwise_objective,
    pe_support_summary,
    require_positive_pe_support,
    sample_gaussian_mixture_events,
)
from .parameters import (
    CalibParams,
    decode_calibration_array,
    decode_calibration_vector,
    encode_calibration_values,
    fixed_params_from_geometry,
    medium_lengths_at,
    truth_from_geometry,
)
from .irf import make_identity_response_forward
from .time_grid import TimeGrid

__all__ = [
    "JunoLaserConfig",
    "PARAMETER_NAMES",
    "POSITIVE_PARAMETER_NAMES",
    "EventwisePeObservation",
    "CalibParams",
    "decode_calibration_array",
    "decode_calibration_vector",
    "encode_calibration_values",
    "fixed_params_from_geometry",
    "medium_lengths_at",
    "truth_from_geometry",
    "make_identity_response_forward",
    "TimeGrid",
    "gaussian_mixture_rate",
    "gaussian_window_diagnostics",
    "gaussian_window_probability",
    "make_gaussian_eventwise_objective",
    "make_host_accumulated_gaussian_eventwise_objective",
    "pe_support_summary",
    "require_positive_pe_support",
    "sample_gaussian_mixture_events",
]
