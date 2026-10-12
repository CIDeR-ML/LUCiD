"""Paired PE samplers for the JUNO laser sampling diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .eventwise import EventwisePeObservation
from .time_grid import TimeGrid


def sample_direct_weighted_hits(
    weights,
    pmt_ids,
    hit_times_ns,
    *,
    n_events: int,
    n_sensors: int,
    time_grid: TimeGrid,
    sigma_ns: float,
    seed: int,
) -> EventwisePeObservation:
    """Sample the legacy per-hit Poisson construction with time jitter.

    The legacy construction has independent ``Poisson(weight_i)`` counts.
    This function uses the exactly equivalent Poisson-superposition identity:
    draw the total from ``Poisson(sum_i weight_i)`` and then draw raw-hit
    categories with probabilities ``weight_i / sum_i weight_i``.  This avoids
    allocating one Poisson variate per raw candidate for every pseudo-event,
    without changing the joint count distribution.  Gaussian jitter is
    applied only after the PE sample has been made.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sensors = np.asarray(pmt_ids, dtype=np.int32)
    times = np.asarray(hit_times_ns, dtype=np.float64)
    if not (weights.ndim == sensors.ndim == times.ndim == 1):
        raise ValueError("weights, PMT IDs and times must be one-dimensional")
    if not (weights.shape == sensors.shape == times.shape):
        raise ValueError("weights, PMT IDs and times must have matching shape")
    if n_events <= 0 or n_sensors <= 0 or sigma_ns <= 0.0:
        raise ValueError("event count, sensor count and sigma must be positive")
    valid = (
        np.isfinite(weights)
        & (weights >= 0.0)
        & np.isfinite(times)
        & (sensors >= 0)
        & (sensors < n_sensors)
        & (times >= time_grid.time_min_ns)
        & (times < time_grid.time_max_ns)
    )
    weights = weights[valid]
    sensors = sensors[valid]
    times = times[valid]
    rng = np.random.default_rng(seed)
    total_mass = float(np.sum(weights))
    counts = rng.poisson(total_mass, size=n_events)
    event_ids = np.repeat(np.arange(n_events, dtype=np.int32), counts)
    if event_ids.size and total_mass > 0.0:
        selected = rng.choice(
            weights.size,
            size=event_ids.size,
            p=weights / total_mass,
        )
        sampled_sensors = sensors[selected]
        sampled_times = times[selected] + rng.normal(
            0.0, sigma_ns, size=event_ids.size
        )
        retained = (
            (sampled_times >= time_grid.time_min_ns)
            & (sampled_times < time_grid.time_max_ns)
        )
        event_ids = event_ids[retained]
        sampled_sensors = sampled_sensors[retained]
        sampled_times = sampled_times[retained]
    else:
        sampled_sensors = np.empty(0, dtype=np.int32)
        sampled_times = np.empty(0, dtype=np.float64)
    return EventwisePeObservation(
        event_ids=event_ids,
        pmt_ids=sampled_sensors,
        pe_times_ns=sampled_times.astype(np.float32),
        n_events=n_events,
        n_sensors=n_sensors,
        time_grid=time_grid,
    )


def histogram_distance(first_times, second_times, edges):
    """Return normalized histogram and CDF distances for two PE samples."""
    first, _ = np.histogram(first_times, bins=edges)
    second, _ = np.histogram(second_times, bins=edges)
    first_mass = first / max(np.sum(first), 1)
    second_mass = second / max(np.sum(second), 1)
    delta = second_mass - first_mass
    width = np.diff(edges)
    return {
        "total_variation": float(0.5 * np.sum(np.abs(delta))),
        "kolmogorov_smirnov_histogram": float(
            np.max(np.abs(np.cumsum(delta)))
        ),
        "wasserstein_histogram_ns": float(
            np.sum(np.abs(np.cumsum(delta)) * width)
        ),
    }


def summarize_observation(observation: EventwisePeObservation):
    """Return transparent count and PE-time summaries for one sampler."""
    times = np.asarray(observation.pe_times_ns, dtype=np.float64)
    counts = np.bincount(
        observation.event_ids, minlength=observation.n_events
    )
    return {
        "n_events": int(observation.n_events),
        "n_pe": int(observation.n_pe),
        "event_pe_mean": float(np.mean(counts)),
        "event_pe_variance": float(np.var(counts)),
        "time_mean_ns": float(np.mean(times)),
        "time_std_ns": float(np.std(times)),
        "time_quantiles_ns": {
            str(value): float(np.quantile(times, value))
            for value in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
    }


def write_sampling_comparison(
    output_path,
    *,
    raw_weights,
    raw_pmt_ids,
    raw_times_ns,
    bin_masses,
    direct: EventwisePeObservation,
    binned: EventwisePeObservation,
    metadata,
):
    """Write raw basis, both sampled PE streams and metadata to HDF5."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = "lucid_pe_sampling_comparison_v1"
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        raw = handle.create_group("raw_hits")
        raw.create_dataset(
            "weights", data=np.asarray(raw_weights, dtype=np.float32),
            compression="gzip",
        )
        raw.create_dataset(
            "pmt_ids", data=np.asarray(raw_pmt_ids, dtype=np.int32),
            compression="gzip",
        )
        raw.create_dataset(
            "times_ns", data=np.asarray(raw_times_ns, dtype=np.float32),
            compression="gzip",
        )
        handle.create_dataset(
            "bin_masses", data=np.asarray(bin_masses, dtype=np.float32),
            compression="gzip",
        )
        for name, observation in (("direct", direct), ("binned", binned)):
            group = handle.create_group(name)
            group.create_dataset(
                "event_ids", data=observation.event_ids,
                compression="gzip",
            )
            group.create_dataset(
                "pmt_ids", data=observation.pmt_ids,
                compression="gzip",
            )
            group.create_dataset(
                "times_ns", data=observation.pe_times_ns,
                compression="gzip",
            )
