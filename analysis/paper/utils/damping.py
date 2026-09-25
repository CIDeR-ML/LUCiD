"""The damped Gauss-Newton step used by the 2-D loss-geometry figure.

Both halves of that figure claim to take the step the joint fit takes. The streamlines are meant
to show the optimizer's actual field, and the trajectory overlaid on them is meant to be the same
arm — a claim each file used to state in a comment and back with its own copy of the arithmetic.
One definition, so the claim holds by construction.

Numpy only, and that is the reason this is its own module rather than a function in
``calibration.py``: ``calib_plots.py`` is plot-only by design and reads nothing but saved ``.npz``,
so importing the calibration setup there would pull jax and the detector geometry into a rendering
module. ``traj_2d_neyman.py`` is a compute script and equally has no use for matplotlib.

The CONVENTION here is the library's: :func:`lucid.fitting.transforms.damped_matrix` filters the Levenberg
median to the entries carrying curvature, exactly as this does. That was not always true — the
library used to FLOOR the median at ``1e-12``, which on a 2x2 with one null direction halved the
base and doubled the step along it, and this module was kept separate to protect the figure from
that. The library adopted the filter (see its docstring for why flooring is wrong under model
extension), so the disagreement is gone and ``tests/test_paper_damping.py`` now pins the two as
EQUAL rather than as differing by two.

What remains is a packaging reason, not a numerical one: ``from lucid.fitting.gn import
damped_matrix`` runs ``lucid/fitting/__init__.py``, which imports the recon and calibration
modules and therefore jax. ``calib_plots.py`` is plot-only and reads nothing but saved ``.npz``, so
that import does not belong in its dependency chain. Hence a numpy transcription, gated against
the library rather than diverging from it.
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
