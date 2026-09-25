"""Carry the iterate in SCALED coordinates instead of preconditioning around a raw one.

``gauss_newton(..., scale=S)`` preconditions the step (``S·H·S``, ``S·g``, applied as ``S·du``) but
keeps the iterate in raw physical units, so in float32 the iterate's magnitude sets its resolution:
energy in MeV at ~1000 has float32 spacing 6.1e-5, and steps of 1e-6 leave it exactly unchanged,
a silently frozen parameter that looks like convergence. :class:`ScaledProblem` moves the scaling
into the parameterisation, ``u = (theta − origin)/S``, so every component is O(1) and float32
resolution is uniform across parameters (checked against raw float64 in
``tests/test_float32_is_adequate.py``).

``ScaledProblem(inner, S)`` driven with ``scale=None`` is the SAME optimisation as ``inner`` driven
with ``scale=S``, by the chain rule:

    u = (theta − origin)/S      theta = origin + S·u
    dL/du = S · dL/dtheta       H_u   = S · H_theta · S

which is exactly the ``S·g`` and ``S·H·S`` the driver forms, so the two agree to float64 round-off
on a deterministic problem (asserted in ``tests/test_fitting_scaled.py``).

It is an opt-in wrapper: ``ReconProblem`` and ``CalibrationProblem`` keep their raw iterates, and
``fit_track`` and ``calibrate`` do not use it.
"""
import numpy as np

__all__ = ['ScaledProblem']


class ScaledProblem:
    """Wrap a problem so its iterate is ``u = (theta − origin)/scale``.

    Parameters
    ----------
    inner
        Any object satisfying the problem protocol: ``grad_metric_loss(theta, step, refresh)``
        returning ``(g, H, loss)`` in RAW coordinates, and ``accumulate(theta, dtheta)``.
    scale
        Per-parameter scale, the same vector that would otherwise be passed to
        ``gauss_newton(scale=...)`` — e.g. :data:`lucid.fitting.recon.SCALE9`.
    origin
        Offset subtracted before scaling. Defaults to zeros. Supplying the start point makes the
        iterate begin at exactly ``0``, the representation float32 handles best.
    dtype
        dtype the iterate is carried in. ``None`` leaves this wrapper's own ``accumulate``
        unconverted, which is NOT the same as deferring to the inner problem -- ``inner.accumulate``
        is never called at all (see Notes), so the iterate follows whatever the step's dtype is
        (float32 from the JAX Gauss-Newton step). This argument exists because the iterate's dtype
        must be the PROBLEM's choice, not a consequence of which step happens to be configured.
        :class:`~lucid.fitting.calib.CalibrationProblem` is exactly that case: its ``accumulate``
        keeps a float32 jnp iterate on purpose, and wrapping it with ``dtype=None`` moves the
        calibration trajectory. Pass the inner problem's own dtype to preserve it, or
        ``np.float32`` to exercise the low-precision path.

    Notes
    -----
    ``inner.accumulate`` is NOT called. Step application in scaled coordinates is a plain add by
    construction, and delegating would re-apply whatever raw-coordinate policy the inner problem
    has (``ProjectedReconProblem.accumulate`` multiplies by ``S`` itself, which would scale twice).
    The wrapper owns the iterate; the inner problem is only ever handed raw ``theta``.

    ``ProjectedReconProblem`` is not merely an ``accumulate`` hazard -- do not wrap it at all.
    Its ``grad_metric_loss`` ALREADY returns scaled quantities (``g = S·gq + P(S·gt)``, and the
    metric sandwiched in ``S`` likewise), which is why the paper driver runs it with
    ``scale=None``. Wrapping it applies ``S`` a second time, giving ``S²g`` and ``S⁴F``. That is
    a NON-uniform rescale, and a non-uniform rescale genuinely changes the damped Gauss-Newton
    path rather than cancelling -- so the mis-preconditioned solve would be silent. Every
    equivalence test in this package uses ``ReconProblem``, which is raw; none covers this.
    """

    def __init__(self, inner, scale, origin=None, dtype=None):
        self.inner = inner
        self.S = np.asarray(scale, float)
        if np.any(self.S == 0):
            raise ValueError('scale has a zero entry; that direction could never move')
        self.origin = (np.zeros_like(self.S) if origin is None
                       else np.asarray(origin, float))
        self.dtype = dtype

    # -- coordinate maps -------------------------------------------------------------
    def to_raw(self, u):
        """Scaled iterate -> raw physical parameters."""
        return self.origin + self.S * np.asarray(u, float)

    def to_scaled(self, theta):
        """Raw physical parameters -> scaled iterate."""
        return (np.asarray(theta, float) - self.origin) / self.S

    # -- the problem protocol --------------------------------------------------------
    def grad_metric_loss(self, u, step, refresh=True):
        g, H, loss = self.inner.grad_metric_loss(self.to_raw(u), step, refresh)
        g = np.asarray(g)
        H = None if H is None else np.asarray(H)
        gs = self.S * g
        Hs = None if H is None else self.S[:, None] * H * self.S[None, :]
        return gs, Hs, loss

    def accumulate(self, u, du):
        out = np.asarray(u) + np.asarray(du)
        return out if self.dtype is None else np.asarray(out, dtype=self.dtype)
