"""Parameterisation: the map between a fit vector ``theta`` and physical detector parameters.

Every fit in this package needs one, and until now each wrote its own. Calibration's lived as
``build_dp``/``to_real`` inside a figure script; reconstruction's is ``track_from_vec9`` /
``SCALE9`` in :mod:`lucid.fitting.recon`. Giving the concept a name is what makes the choice of
coordinates reviewable instead of incidental — and the choice matters:

* the fit runs in **log** space for positive quantities, so steps are multiplicative and the
  parameters stay positive without a constraint;
* reflection is fitted as ``[log R, logit f]`` rather than as the two components ``R·f`` and
  ``R·(1−f)`` directly, because the direct basis leaves the diffuse component stuck through
  anti-correlation;
* reporting inverts the map, and that inversion is **not** neutral. At ``f_s = 0.90``,
  ``d log(1−f)/d log f = −f/(1−f) = −9``, so a parameter recovered to 0.4% is reported as −2.6%.
  Owning ``to_real`` here is what stops each analysis script re-deriving that factor — and
  getting a different answer.

``CalibrationParams`` below is a byte-for-byte extraction of the reference engine's
``build_dp``/``to_real`` (``analysis/paper/utils/calib_fit.py``). The arithmetic is preserved
exactly, expression for expression: the whole stack runs in float32 (``jax_enable_x64`` is never
enabled), so re-associating even an equivalent expression can move the result, and the engine is
pinned bit-exactly by ``tests/reconciliation/test_calib_engine_pin.py``.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import DetectorParams

__all__ = ['CalibrationParams']


class CalibrationParams:
    """``theta`` ↔ detector parameters for the per-wavelength optics + shared reflection estimand.

    Layout, with ``W`` wavelengths and ``NG = 3W``::

        theta[3*wl + 0]  log scatter_length    at wavelength wl   (free PER wavelength)
        theta[3*wl + 1]  log absorption_length at wavelength wl
        theta[3*wl + 2]  log qe                at wavelength wl
        theta[NG + 0..3] reflection                                (SHARED across wavelengths)

    ``basis`` selects the reflection coordinates:

    ``'rlogit'``   ``[log R_w, logit f_w, log R_s, logit f_s]`` — the published choice.
    ``'specdiff'`` ``log`` of the four ``(spec, diff)`` components directly; ``R`` and ``f`` are
                   then derived. Kept because it is what the earlier campaigns ran.

    Mie is frozen (``mie_scatter_length=1e6``, ``g=0.9``): the fitted ``scatter_length`` is
    Rayleigh-only and the asymmetric channel is dropped, not folded in. It is a constructor
    argument so that is visible at the call site rather than buried.
    """

    def __init__(self, n_wavelengths, basis='rlogit', mie_scatter_length=1e6, g=0.9):
        if basis not in ('rlogit', 'specdiff'):
            raise ValueError(f"basis must be 'rlogit' or 'specdiff', got {basis!r}")
        self.W = int(n_wavelengths)
        self.NG = 3 * self.W                     # index of the first reflection parameter
        self.P = self.NG + 4                     # total free parameters
        self.basis = basis
        self.mie_scatter_length = mie_scatter_length
        self.g = g

    # -- fit space -> the forward model -------------------------------------------------------
    def to_dp(self, theta, wl, gains):
        """DetectorParams for wavelength index ``wl`` at fit vector ``theta``.

        Differentiable: this is called inside ``jacfwd``, so it must stay a pure JAX expression.
        """
        NG = self.NG
        p = jnp.exp(theta)
        if self.basis == 'rlogit':
            Rw = jnp.exp(theta[NG]); fw = jax.nn.sigmoid(theta[NG + 1])
            Rs = jnp.exp(theta[NG + 2]); fs = jax.nn.sigmoid(theta[NG + 3])
        else:
            Rw = p[NG] + p[NG + 1]; fw = p[NG] / Rw
            Rs = p[NG + 2] + p[NG + 3]; fs = p[NG + 2] / Rs
        return DetectorParams.from_flat(
            scatter_length=p[3 * wl], mie_scatter_length=self.mie_scatter_length, g=self.g,
            wall_reflection_rate=Rw, sensor_reflection_rate=Rs, wall_fspec=fw, sensor_fspec=fs,
            absorption_length=p[3 * wl + 1], qe=p[3 * wl + 2], qe_corrections=gains)

    # -- fit space -> reported quantities -----------------------------------------------------
    def to_real(self, theta):
        """Fit vector -> the reported vector: per-λ optics, then the four spec/diff reflectivities.

        The reflection half is where the reporting basis amplifies (see the module docstring);
        quote ``(R, f)`` alongside, or state the lever.
        """
        NG = self.NG
        r = np.exp(np.asarray(theta[:NG]))
        if self.basis == 'rlogit':
            Rw = np.exp(theta[NG]); fw = 1 / (1 + np.exp(-theta[NG + 1]))
            Rs = np.exp(theta[NG + 2]); fs = 1 / (1 + np.exp(-theta[NG + 3]))
            refl = [Rw * fw, Rw * (1 - fw), Rs * fs, Rs * (1 - fs)]
        else:
            refl = list(np.exp(np.asarray(theta[NG:])))
        return np.concatenate([r, refl])

    # -- construction --------------------------------------------------------------------------
    def theta_from_physical(self, per_wavelength, wall_r, wall_fspec, sensor_r, sensor_fspec):
        """Build ``theta`` from physical values.

        ``per_wavelength`` is a sequence of ``W`` mappings with keys ``scatter_length``,
        ``absorption_length`` and ``qe``.
        """
        if len(per_wavelength) != self.W:
            raise ValueError(f'expected {self.W} wavelength entries, got {len(per_wavelength)}')
        opt = []
        for t in per_wavelength:
            opt += [t['scatter_length'], t['absorption_length'], t['qe']]
        if self.basis == 'rlogit':
            _logit = lambda f: np.log(f / (1 - f))
            refl = [np.log(wall_r), _logit(wall_fspec), np.log(sensor_r), _logit(sensor_fspec)]
        else:
            refl = list(np.log([wall_r * wall_fspec, wall_r * (1 - wall_fspec),
                                sensor_r * sensor_fspec, sensor_r * (1 - sensor_fspec)]))
        return np.array(list(np.log(opt)) + refl)
