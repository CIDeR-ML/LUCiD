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
  ``d log(1−f)/d log f = −f/(1−f) = −9``, so a small fractional error on ``f`` is amplified
  ninefold in the reported diffuse component. (The campaign's measured pair is 0.4% on the fitted
  parameter against −2.6% reported — smaller than −9 × 0.4% because the two are not the same
  perturbation.) Owning ``to_real`` here is what stops
  each analysis script re-deriving that factor and getting a different answer.

Three parameterisations ship, and they answer different questions.

``CalibrationParams`` is the published estimand: a fixed layout of per-wavelength optics plus
shared reflection, in the coordinates the campaign chose. It is a byte-for-byte extraction of the
paper's reference calibration engine's ``build_dp``/``to_real``, preserved expression for
expression — the whole stack runs in float32 (``jax_enable_x64`` is never enabled), so
re-associating even an equivalent expression can move the result. Its bit-exact agreement with that
engine was verified when it was extracted; the engine is not in this repository and the check is not
shipped.

``FieldParams`` is the general one: name any leaf fields of a ``DetectorParams`` and fit those.
It is what lets somebody calibrate a quantity the paper never fitted without writing a new
forward model, and it is the parameterisation ``build_calibration_problem`` builds.

``LogParams`` is the degenerate case, for a caller who supplies the forward already parameterised
and needs only the log-space convention named.

All three answer the same four questions, which is all the forward model and the fitter ever ask
of a parameterisation:

``W``          number of wavelength slices the forward evaluates (1 if the fit has no λ axis)
``P``          number of free parameters
``to_dp(theta, wl, gains)``  fit vector -> ``DetectorParams``; must be traceable, it runs
               inside ``jacfwd``
``to_real(theta)``           fit vector -> the reported vector

Anything answering all four can build its own forward and be fitted, which is the point of naming
it. ``LogParams`` answers only three: its ``to_dp`` raises, because the map into detector
parameters is inside the forward closure supplied alongside it. It is fittable, but only by a
caller that brings that forward.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import (
    DetectorParams, _flatten_detector_params, _nest_flat_kwargs,
)

__all__ = ['CalibrationParams', 'FieldParams', 'LogParams']


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


class FieldParams:
    """Fit an arbitrary set of ``DetectorParams`` leaf fields, in log space.

    Where :class:`CalibrationParams` fixes the estimand, this one takes it as an argument: give a
    truth ``DetectorParams`` and a list of flat leaf names, and everything not named is held at its
    truth value. Scalars and array-valued leaves (the λ-deviation curves) both work — an array
    field contributes its whole ravelled length to ``theta``.

    Log space throughout, so the parameters stay positive without a constraint and steps are
    multiplicative. That suits every optical length, rate and efficiency — but it is a real
    restriction, not a free choice: a field that can legitimately be negative cannot be
    represented. ``g``, the Henyey-Greenstein asymmetry, is such a field. It is listed among the
    trainable fields in :func:`lucid.fitting.problem.build_calibration_problem` and fitted by
    ``examples/hello_calibrate.py``, and in this parameterisation it is confined to ``g > 0``.

    There is no wavelength axis (``W = 1``). A λ-dependent quantity is expressed as an array-valued
    leaf that the forward consumes whole, rather than by evaluating the forward once per λ. Both
    are representable; this is the shape the field-based bridge has always used.

    ``per_pmt_field`` is the per-sensor multiplicative factor. It is NOT part of ``theta``: there is
    one per sensor, and profiling it in closed form each step is what keeps the handful of
    interesting parameters from being swamped (see :func:`lucid.fitting.calib.profile_gains`).
    ``to_dp`` writes the supplied ``gains`` into it.
    """

    W = 1

    def __init__(self, dp_true, fields, per_pmt_field='qe_corrections', n_sensors=None):
        flat = {k: np.asarray(v) for k, v in _flatten_detector_params(dp_true).items()}
        missing = [f for f in fields if f not in flat]
        if missing:
            raise KeyError(f'not DetectorParams leaf fields: {missing}')
        if per_pmt_field not in flat:
            raise KeyError(f'per-PMT field {per_pmt_field!r} is not a DetectorParams leaf')
        self.fields = list(fields)
        self.per_pmt_field = per_pmt_field
        self.shapes = [(f, flat[f].shape) for f in self.fields]
        self.NS = int(n_sensors) if n_sensors is not None else int(flat[per_pmt_field].shape[0])
        self._flat_true = flat
        self._flat_true_j = {k: jnp.asarray(v) for k, v in flat.items()}
        self.P = int(sum(int(np.prod(s)) if s else 1 for _, s in self.shapes))

    def to_dp(self, theta, wl, gains):
        """DetectorParams at fit vector ``theta``. ``wl`` is accepted and ignored (``W = 1``)."""
        vals = jnp.exp(jnp.asarray(theta))
        flat = dict(self._flat_true_j)
        i = 0
        for f, shp in self.shapes:
            size = int(np.prod(shp)) if shp else 1
            flat[f] = vals[i:i + size].reshape(shp)
            i += size
        flat[self.per_pmt_field] = jnp.asarray(gains) * jnp.ones(self.NS)
        return _nest_flat_kwargs(flat)

    def to_real(self, theta):
        """Fit vector -> physical values. Log space throughout, so this is just ``exp``."""
        return np.exp(np.asarray(theta))

    def theta_from_physical(self, dp=None):
        """``theta`` at the truth, or at another ``DetectorParams``: the natural start."""
        flat = self._flat_true if dp is None else {
            k: np.asarray(v) for k, v in _flatten_detector_params(dp).items()}
        parts = [np.asarray(flat[f], dtype=np.float64).ravel() for f in self.fields]
        vec = np.concatenate(parts) if parts else np.zeros(0)
        return np.log(np.clip(vec, 1e-12, None))


class LogParams:
    """``theta`` is the log of the reported quantity, and the forward is supplied by the caller.

    The degenerate parameterisation. It exists for the case where the map from ``theta`` to
    detector parameters is already inside the forward closure — which is what
    :func:`lucid.fitting.problem.build_calibration_problem` produces, and what
    :func:`lucid.fitting.calibrate.fit` consumes. There is then nothing left for a
    parameterisation to do except state the coordinate convention, which is worth stating: the fit
    runs in log space, so ``to_real`` is ``exp`` and every reported quantity is positive.

    ``to_dp`` raises. A caller reaching it has passed this class somewhere that needs a real
    parameterisation, and a silent wrong answer there would be worse than a stop.
    """

    W = 1

    def __init__(self, n_params):
        self.P = int(n_params)

    def to_dp(self, theta, wl, gains):
        raise TypeError(
            'LogParams has no DetectorParams map: the theta -> detector-parameter map lives in '
            'the forward closure supplied alongside it. Use FieldParams or CalibrationParams for '
            'a fit that builds its own forward.')

    def to_real(self, theta):
        return np.exp(np.asarray(theta))
