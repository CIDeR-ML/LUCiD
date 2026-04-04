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
    n_grad_iters : int
        Iteration threshold for direction stop_gradient.
        Default derived from mode: track=0, calibration=2.
    """
    n_photons: int = 1_000_000
    K: int = 7
    mode: str = 'track'
    use_expected_value: bool = True
    apply_smearing: bool = True
    n_grad_iters: Optional[int] = None

    @property
    def effective_n_grad_iters(self) -> int:
        """Resolve n_grad_iters: explicit value or mode default."""
        if self.n_grad_iters is not None:
            return self.n_grad_iters
        return {'track': 0, 'calibration': 2, 'data': 0}.get(self.mode, 0)

    @property
    def is_data(self) -> bool:
        return self.mode == 'data'

    @property
    def is_calibration(self) -> bool:
        return self.mode == 'calibration'
