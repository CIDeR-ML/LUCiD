"""Loss-landscape sweeps: 1D and 2D scans of a loss and its autodiff gradient over a parameter,
with finite-difference checks (``sweep``) and their plots (``plotting``)."""

from .plotting import plot_sweep_1d, plot_sweep_2d, plot_sweep_2d_single
from .sweep import (
    CALIB_SWEEPS,
    RECO_SWEEPS,
    SweepParam,
    SweepResult1D,
    SweepResult2D,
    find_zero_crossing,
    get_grad_component,
    get_param_value,
    numerical_gradient,
    set_param_value,
    sweep_1d,
    sweep_2d,
)
