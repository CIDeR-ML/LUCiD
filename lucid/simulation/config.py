"""Simulation configuration container."""
from typing import NamedTuple, Optional


class SimConfig(NamedTuple):
    """Simulation parameters that control the propagation loop behavior.

    These are mode-independent physics settings that don't depend on the
    detector geometry or particle model.

    Parameters
    ----------
    n_photons : int
        Number of photon rays per event.
    K : int
        Maximum scattering iterations before forced detection.
    mode : str
        One of 'track', 'calibration', 'data'.
    use_expected_value : bool
        True → STE (differentiable), False → MC sampling.
    apply_smearing : bool
        Apply SK-like charge/time smearing in data mode.
    tts_sigma_ns : float
        Single-PE PMT transit-time spread (ns). 2.5 ns matches SK 20-inch
        R3600. Applied per-photon before segment_min in make_hits_data.
    n_grad_iters : int
        Iteration threshold for direction stop_gradient.
        Default derived from mode: track=K (gradient flows all bounces),
        calibration=2, data=0.
    """
    n_photons: int = 1_000_000
    K: int = 7
    mode: str = 'track'
    use_expected_value: bool = True
    apply_smearing: bool = True
    tts_sigma_ns: float = 2.5
    n_grad_iters: Optional[int] = None
    max_candidates_per_ray: int = 4

    @property
    def effective_n_grad_iters(self) -> int:
        """Resolve n_grad_iters: explicit value or mode default."""
        if self.n_grad_iters is not None:
            return self.n_grad_iters
        # Track default used to be 0 (fully detach direction gradient) as a
        # workaround for the reflection-normal curvature explosion. That is now
        # fixed at the normal level inside photon_iteration_update_factors, so
        # direction gradient can flow all K bounces.
        return {'track': self.K, 'calibration': 2, 'data': 0}.get(self.mode, 0)

    @property
    def is_data(self) -> bool:
        return self.mode == 'data'

    @property
    def is_calibration(self) -> bool:
        return self.mode == 'calibration'
