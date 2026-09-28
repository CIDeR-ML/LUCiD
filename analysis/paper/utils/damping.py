"""The damped Gauss-Newton step used by the 2-D loss-geometry figure.

The streamlines (``calib_plots.py``) and the overlaid trajectory (``traj_2d_neyman.py``) must both
take the step the joint fit takes; one shared definition makes that hold by construction.

This is a numpy transcription of :func:`lucid.fitting.transforms.damped_matrix` (same convention:
the Levenberg median is taken over the diagonal entries carrying curvature), and
``tests/test_paper_damping.py`` pins the two as equal. It is not imported from the library because
importing any ``lucid.fitting`` submodule runs ``lucid/fitting/__init__.py``, which pulls in jax,
and ``calib_plots.py`` is plot-only and reads nothing but saved ``.npz``.
"""
import numpy as np

__all__ = ['damped_step']


def damped_step(F, g, lam, mu):
    """``-(F + lam*diag(F) + mu*median(diag F > 0)*I)^-1 g``, without the sign.

    Returns the SOLVE, so the caller applies the sign — the streamline field wants the descent
    direction and the trajectory wants the step, and they differ only in that.

    ``lam`` (Marquardt) scales with each direction's own curvature; ``mu`` (Levenberg) is
    isotropic and is the only guarantee of a non-singular solve, since for PSD ``F`` it shifts
    every eigenvalue by exactly ``mu*base``. Pass both from ``CALIB_RECIPE`` so the figure cannot
    describe a damping the run did not use.
    """
    dg = np.clip(np.diag(F), 0.0, None)
    pos = dg[dg > 0]
    base = np.median(pos) if pos.size else 1.0        # F is PSD -> lambda_min >= mu*base > 0
    return np.linalg.solve(F + lam * np.diag(dg) + mu * base * np.eye(F.shape[0]), g)
