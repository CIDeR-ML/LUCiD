"""Minimal event-wise Poisson likelihood for JUNO laser PE times.

The forward model predicts expected PE mass in PMT-by-time cells.  A fixed
numerical Gaussian kernel converts those masses into a continuous intensity
evaluated at every exact PE timestamp.  The kernel is not yet a claim about
the physical TTS or electronics response.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from .time_grid import TimeGrid


@dataclass(frozen=True)
class EventwisePeObservation:
    """PE event IDs, PMT IDs and times for one fixed source position."""

    event_ids: np.ndarray
    pmt_ids: np.ndarray
    pe_times_ns: np.ndarray
    n_events: int
    n_sensors: int
    time_grid: TimeGrid

    def __post_init__(self):
        """Normalize dtypes and validate the event-wise data contract."""
        events = np.asarray(self.event_ids, dtype=np.int32)
        sensors = np.asarray(self.pmt_ids, dtype=np.int32)
        times = np.asarray(self.pe_times_ns, dtype=np.float32)
        if not (events.ndim == sensors.ndim == times.ndim == 1):
            raise ValueError("PE event, PMT and time arrays must be 1D")
        if not (events.shape == sensors.shape == times.shape):
            raise ValueError("PE event, PMT and time arrays must match")
        if self.n_events <= 0 or self.n_sensors <= 0:
            raise ValueError("event and sensor counts must be positive")
        if not isinstance(self.time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid")
        if events.size and np.any((events < 0) | (events >= self.n_events)):
            raise ValueError("event IDs are outside the event range")
        if sensors.size and np.any(
            (sensors < 0) | (sensors >= self.n_sensors)
        ):
            raise ValueError("PMT IDs are outside the detector range")
        outside = (
            (~np.isfinite(times))
            | (times < self.time_grid.time_min_ns)
            | (times >= self.time_grid.time_max_ns)
        )
        if times.size and np.any(outside):
            raise ValueError("PE times are outside the observation window")
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "pmt_ids", sensors)
        object.__setattr__(self, "pe_times_ns", times)

    @property
    def n_pe(self) -> int:
        """Return the number of retained PE timestamps."""
        return int(self.pe_times_ns.size)

    @property
    def time_min_ns(self) -> float:
        """Return the shared observation-window lower edge in ns."""
        return self.time_grid.time_min_ns

    @property
    def time_max_ns(self) -> float:
        """Return the shared observation-window upper edge in ns."""
        return self.time_grid.time_max_ns


class EventwisePoissonTerms(NamedTuple):
    """Separate integral and observed-event terms of the likelihood."""

    integral: jnp.ndarray
    negative_log_rate: jnp.ndarray
    nll: jnp.ndarray
    pe_rate: jnp.ndarray


def pe_support_summary(pe_rate, *, low_rate_threshold: float = 1.0e-12):
    """Return host-side diagnostics for rates at observed PE positions."""
    if low_rate_threshold <= 0.0:
        raise ValueError("low-rate threshold must be positive")
    rates = np.asarray(pe_rate, dtype=np.float64)
    if rates.ndim != 1:
        raise ValueError("PE rates must be one-dimensional")
    return {
        "n_pe": int(rates.size),
        "min_pe_rate_per_ns": (
            float(np.min(rates)) if rates.size else None
        ),
        "n_nonpositive_pe_rate": int(np.count_nonzero(rates <= 0.0)),
        "n_below_rate_threshold": int(
            np.count_nonzero(rates < low_rate_threshold)
        ),
        "low_rate_threshold_per_ns": float(low_rate_threshold),
    }


def require_positive_pe_support(
    pe_rate,
    *,
    context: str,
    low_rate_threshold: float = 1.0e-12,
):
    """Raise when an observed PE lies outside the predicted PPP support."""
    summary = pe_support_summary(
        pe_rate, low_rate_threshold=low_rate_threshold
    )
    if summary["n_nonpositive_pe_rate"]:
        raise ValueError(
            f"PPP support failure for {context}: "
            f"{summary['n_nonpositive_pe_rate']} observed PEs have "
            "non-positive predicted rate"
        )
    return summary


def gaussian_mixture_rate(
    bin_masses,
    pmt_ids,
    pe_times_ns,
    *,
    time_grid: TimeGrid,
    sigma_ns: float,
    truncate_sigma: float = 5.0,
):
    """Evaluate full-axis Gaussian density at each observed PE time.

    ``truncate_sigma`` limits only the numerical candidate stencil.  The
    retained density is the original full-axis Gaussian and is never
    renormalized against the observation window.
    """
    sigma_values = np.asarray(sigma_ns, dtype=np.float64)
    if (
        np.any(sigma_values <= 0.0)
        or truncate_sigma <= 0.0
    ):
        raise ValueError("time-kernel scales must be positive")
    masses = jnp.asarray(bin_masses)
    if masses.ndim != 2 or masses.shape[1] < 1:
        raise ValueError("bin masses must have shape (n_sensors, n_bins)")
    time_grid.validate_mass_shape(masses)
    sensors = jnp.asarray(pmt_ids, dtype=jnp.int32)
    times = jnp.asarray(pe_times_ns, dtype=masses.dtype)
    if sensors.ndim != 1 or sensors.shape != times.shape:
        raise ValueError("PMT IDs and PE times must be matching 1D arrays")
    if sigma_values.ndim == 0:
        selected_sigma = jnp.full(
            times.shape, float(sigma_values), dtype=masses.dtype
        )
    elif sigma_values.shape == (masses.shape[0],):
        selected_sigma = jnp.asarray(
            sigma_values, dtype=masses.dtype
        )[sensors]
    else:
        raise ValueError(
            "sigma must be scalar or have one value per sensor"
        )
    radius = int(
        math.ceil(
            truncate_sigma
            * float(np.max(sigma_values))
            / time_grid.bin_width_ns
        )
    ) + 1
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    coordinate = (
        (times - time_grid.time_min_ns) / time_grid.bin_width_ns - 0.5
    )
    nearest = jnp.floor(coordinate + 0.5).astype(jnp.int32)
    indices = nearest[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < masses.shape[1])
    safe_indices = jnp.clip(indices, 0, masses.shape[1] - 1)
    selected = masses[sensors[:, None], safe_indices]
    centers = time_grid.time_min_ns + (
        safe_indices.astype(masses.dtype) + 0.5
    ) * time_grid.bin_width_ns
    sigma = selected_sigma[:, None]
    z = (times[:, None] - centers) / sigma
    density = jnp.exp(-0.5 * z**2) / (
        math.sqrt(2.0 * math.pi) * sigma
    )
    terms = jnp.where(
        valid & (jnp.abs(z) <= truncate_sigma), selected * density, 0.0
    )
    return jnp.sum(terms, axis=1)


def gaussian_window_probability(
    *,
    time_grid: TimeGrid,
    sigma_ns,
    n_sensors: int,
    window_min_ns: float | None = None,
    window_max_ns: float | None = None,
    dtype=jnp.float32,
):
    """Return each Gaussian component's probability inside a time window."""
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    lower_edge = (
        time_grid.time_min_ns
        if window_min_ns is None
        else float(window_min_ns)
    )
    upper_edge = (
        time_grid.time_max_ns
        if window_max_ns is None
        else float(window_max_ns)
    )
    if upper_edge <= lower_edge:
        raise ValueError("integration window must be increasing")
    sigma_values = np.asarray(sigma_ns, dtype=np.float64)
    if sigma_values.ndim == 0:
        sigma_values = np.full(n_sensors, float(sigma_values))
    if sigma_values.shape != (n_sensors,) or np.any(sigma_values <= 0.0):
        raise ValueError("sigma must be positive and scalar or per-sensor")
    sigma = jnp.asarray(sigma_values, dtype=dtype)[:, None]
    centers = jnp.asarray(time_grid.centers, dtype=dtype)[None, :]
    upper = (upper_edge - centers) / sigma
    lower = (lower_edge - centers) / sigma
    return jax.scipy.special.ndtr(upper) - jax.scipy.special.ndtr(lower)


def gaussian_window_diagnostics(
    bin_masses,
    *,
    time_grid: TimeGrid,
    sigma_ns,
    expansion_ns: float = 50.0,
    observation: EventwisePeObservation | None = None,
):
    """Summarize Gaussian mass accepted by current and expanded windows."""
    if expansion_ns <= 0.0:
        raise ValueError("window expansion must be positive")
    masses = jnp.asarray(bin_masses)
    time_grid.validate_mass_shape(masses)
    if observation is None:
        current = gaussian_window_probability(
            time_grid=time_grid,
            sigma_ns=sigma_ns,
            n_sensors=masses.shape[0],
            dtype=masses.dtype,
        )
        expanded = gaussian_window_probability(
            time_grid=time_grid,
            sigma_ns=sigma_ns,
            n_sensors=masses.shape[0],
            window_min_ns=time_grid.time_min_ns - expansion_ns,
            window_max_ns=time_grid.time_max_ns + expansion_ns,
            dtype=masses.dtype,
        )
        event_count = 1
    else:
        if observation.time_grid != time_grid:
            raise ValueError("diagnostic observation must share the time grid")
        if observation.n_sensors != masses.shape[0]:
            raise ValueError("diagnostic observation must share the sensors")
        current = gaussian_window_probability(
            time_grid=time_grid,
            sigma_ns=sigma_ns,
            n_sensors=masses.shape[0],
            dtype=masses.dtype,
        )
        expanded = gaussian_window_probability(
            time_grid=time_grid,
            sigma_ns=sigma_ns,
            n_sensors=masses.shape[0],
            window_min_ns=time_grid.time_min_ns - expansion_ns,
            window_max_ns=time_grid.time_max_ns + expansion_ns,
            dtype=masses.dtype,
        )
        event_count = observation.n_events
    full_mass = jnp.sum(masses)
    current_mass = jnp.sum(masses * current)
    expanded_mass = jnp.sum(masses * expanded)
    denominator = jnp.maximum(
        event_count * full_mass, jnp.finfo(masses.dtype).tiny
    )
    return {
        "full_axis_mass": float(event_count * full_mass),
        "current_window_mass": float(event_count * current_mass),
        "current_window_fraction": float(
            event_count * current_mass / denominator
        ),
        "expanded_window_mass": float(event_count * expanded_mass),
        "expanded_window_fraction": float(
            event_count * expanded_mass / denominator
        ),
        "expanded_minus_current_fraction": float(
            event_count * (expanded_mass - current_mass) / denominator
        ),
        "expansion_ns": float(expansion_ns),
    }


def gaussian_eventwise_poisson_terms(
    observation: EventwisePeObservation,
    bin_masses,
    *,
    sigma_ns: float,
    truncate_sigma: float = 5.0,
):
    """Evaluate an unbinned PPP likelihood using a fixed Gaussian kernel."""
    masses = jnp.asarray(bin_masses)
    if masses.ndim != 2 or masses.shape[0] != observation.n_sensors:
        raise ValueError("bin masses must match the observation sensors")
    observation.time_grid.validate_mass_shape(masses)
    pe_rate = gaussian_mixture_rate(
        masses,
        observation.pmt_ids,
        observation.pe_times_ns,
        time_grid=observation.time_grid,
        sigma_ns=sigma_ns,
        truncate_sigma=truncate_sigma,
    )
    window_probability = gaussian_window_probability(
        time_grid=observation.time_grid,
        sigma_ns=sigma_ns,
        n_sensors=observation.n_sensors,
        dtype=masses.dtype,
    )
    integral = observation.n_events * jnp.sum(masses * window_probability)
    negative_log_rate = -jnp.sum(jnp.log(pe_rate))
    return EventwisePoissonTerms(
        integral=integral,
        negative_log_rate=negative_log_rate,
        nll=integral + negative_log_rate,
        pe_rate=pe_rate,
    )


def make_gaussian_eventwise_objective(
    ensemble_forward,
    observations: Sequence[EventwisePeObservation],
    *,
    sigma_ns: float,
    truncate_sigma: float = 5.0,
    normalize: bool = True,
):
    """Build a continuous-time fixed-kernel multi-source PE objective."""
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one event-wise observation is required")
    denominator = max(sum(item.n_pe for item in observations), 1)

    @jax.jit
    def objective(theta, keys):
        predictions = ensemble_forward(theta, keys)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for observation, prediction in zip(observations, predictions):
            terms = gaussian_eventwise_poisson_terms(
                observation,
                prediction,
                sigma_ns=sigma_ns,
                truncate_sigma=truncate_sigma,
            )
            total += terms.nll
        return total / denominator if normalize else total

    return objective


def make_chunked_gaussian_eventwise_objective(
    ensemble_forwards,
    observation_chunks,
    *,
    sigma_ns: float,
    truncate_sigma: float = 5.0,
):
    """Build one globally normalized PPP objective from source chunks.

    Each forward returns only the predictions in its own source chunk.  The
    chunk NLLs are summed before division by the global PE count, preserving
    the unchunked multi-source objective while limiting the largest returned
    prediction tuple.
    """
    forwards = tuple(ensemble_forwards)
    chunks = tuple(tuple(items) for items in observation_chunks)
    if not forwards or len(forwards) != len(chunks):
        raise ValueError(
            "one non-empty observation chunk is required per forward"
        )
    if any(not chunk for chunk in chunks):
        raise ValueError("observation chunks must not be empty")
    denominator = max(
        sum(item.n_pe for chunk in chunks for item in chunk), 1
    )
    objectives = tuple(
        make_gaussian_eventwise_objective(
            forward,
            chunk,
            sigma_ns=sigma_ns,
            truncate_sigma=truncate_sigma,
            normalize=False,
        )
        for forward, chunk in zip(forwards, chunks)
    )

    @jax.jit
    def objective(theta, keys):
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for chunk_objective in objectives:
            total += chunk_objective(theta, keys)
        return total / denominator

    return objective


class HostAccumulatedObjective:
    """Evaluate independent source objectives without an outer XLA graph.

    Each member returns an unnormalized source NLL.  Values and gradients are
    explicitly synchronized, copied to the host and summed before one global
    PE normalization.  This preserves the joint PPP while preventing XLA from
    fusing several source transports into one replayed ROCm executable.
    """

    def __init__(self, objectives, denominator):
        """Store source objectives and prepare their independent VJPs."""
        self.objectives = tuple(objectives)
        self.denominator = int(denominator)
        if not self.objectives or self.denominator <= 0:
            raise ValueError("host objective requires sources and PE mass")
        self._value_and_grads = tuple(
            jax.value_and_grad(objective)
            for objective in self.objectives
        )

    def __call__(self, theta, keys):
        """Return the globally normalized loss using host accumulation."""
        total = 0.0
        for objective in self.objectives:
            value = objective(theta, keys)
            value.block_until_ready()
            total += float(value)
        return total / self.denominator

    def value_and_grad(self, theta, keys):
        """Return one joint loss/gradient without a multi-source XLA graph."""
        total = 0.0
        gradient_total = None
        for value_and_grad in self._value_and_grads:
            value, gradient = value_and_grad(theta, keys)
            jax.block_until_ready((value, gradient))
            total += float(value)
            host_gradient = np.asarray(gradient, dtype=np.float64)
            gradient_total = (
                host_gradient
                if gradient_total is None
                else gradient_total + host_gradient
            )
        return (
            total / self.denominator,
            gradient_total / self.denominator,
        )


def make_host_accumulated_gaussian_eventwise_objective(
    ensemble_forwards,
    observation_chunks,
    *,
    sigma_ns: float,
    truncate_sigma: float = 5.0,
):
    """Build a joint PPP evaluated as independent one-source modules."""
    forwards = tuple(ensemble_forwards)
    chunks = tuple(tuple(items) for items in observation_chunks)
    if not forwards or len(forwards) != len(chunks):
        raise ValueError(
            "one non-empty observation chunk is required per forward"
        )
    if any(len(chunk) != 1 for chunk in chunks):
        raise ValueError("host accumulation requires one source per chunk")
    denominator = max(
        sum(item.n_pe for chunk in chunks for item in chunk), 1
    )
    objectives = tuple(
        make_gaussian_eventwise_objective(
            forward,
            chunk,
            sigma_ns=sigma_ns,
            truncate_sigma=truncate_sigma,
            normalize=False,
        )
        for forward, chunk in zip(forwards, chunks)
    )
    return HostAccumulatedObjective(objectives, denominator)


def sample_gaussian_mixture_events(
    bin_masses,
    *,
    n_events: int,
    time_grid: TimeGrid,
    sigma_ns: float,
    seed: int,
    truncate_sigma: float = 5.0,
) -> EventwisePeObservation:
    """Sample fixed-exposure, zero-offset signal-only event-wise PEs."""
    masses = np.asarray(bin_masses, dtype=np.float64)
    if masses.ndim != 2 or masses.shape[1] < 1:
        raise ValueError("bin masses must have shape (n_sensors, n_bins)")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("bin masses must be finite and non-negative")
    sigma_values = np.asarray(sigma_ns, dtype=np.float64)
    if sigma_values.ndim not in (0, 1):
        raise ValueError("sigma must be scalar or one-dimensional")
    if sigma_values.ndim == 1 and sigma_values.shape != (masses.shape[0],):
        raise ValueError("sigma must have one value per sensor")
    if (
        n_events <= 0
        or np.any(sigma_values <= 0.0)
        or truncate_sigma <= 0.0
    ):
        raise ValueError("event count and time-kernel scales must be positive")
    time_grid.validate_mass_shape(masses)
    rng = np.random.default_rng(seed)
    sigma_per_sensor = (
        np.full(masses.shape[0], float(sigma_values))
        if sigma_values.ndim == 0
        else sigma_values
    )
    total_full_mass = float(np.sum(masses))
    counts = rng.poisson(total_full_mass, size=n_events)
    candidate_events = np.repeat(
        np.arange(n_events, dtype=np.int32), counts
    )
    n_candidates = int(np.sum(counts))
    if n_candidates and total_full_mass > 0.0:
        flat_bins = rng.choice(
            masses.size,
            size=n_candidates,
            p=masses.ravel() / total_full_mass,
        )
        n_bins = masses.shape[1]
        pmt_ids = (flat_bins // n_bins).astype(np.int32)
        time_bins = flat_bins % n_bins
        selected_sigma = (
            np.full(n_candidates, float(sigma_values), dtype=np.float64)
            if sigma_values.ndim == 0
            else sigma_values[pmt_ids]
        )
        centers = time_grid.time_min_ns + (
            time_bins + 0.5
        ) * time_grid.bin_width_ns
        times = rng.normal(
            centers, selected_sigma
        )
        retained = (
            (times >= time_grid.time_min_ns)
            & (times < time_grid.time_max_ns)
        )
        event_ids = candidate_events[retained]
        pmt_ids = pmt_ids[retained]
        times = times[retained]
    else:
        event_ids = np.empty(0, dtype=np.int32)
        pmt_ids = np.empty(0, dtype=np.int32)
        times = np.empty(0, dtype=np.float64)
    times = np.asarray(times, dtype=np.float32)
    return EventwisePeObservation(
        event_ids=event_ids,
        pmt_ids=pmt_ids,
        pe_times_ns=times,
        n_events=n_events,
        n_sensors=masses.shape[0],
        time_grid=time_grid,
    )
