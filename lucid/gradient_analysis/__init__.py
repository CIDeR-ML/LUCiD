"""Loss-landscape sweeps: 1D and 2D scans of a loss and its autodiff gradient over a parameter,
with finite-difference checks (``sweep``) and their plots (``plotting``)."""
from .sweep import (
    SweepParam,
    SweepResult1D,
    SweepResult2D,
    sweep_1d,
    sweep_2d,
    numerical_gradient,
    get_param_value,
    set_param_value,
    get_grad_component,
    find_zero_crossing,
    RECO_SWEEPS,
    CALIB_SWEEPS,
)
from .plotting import plot_sweep_1d, plot_sweep_2d, plot_sweep_2d_single
