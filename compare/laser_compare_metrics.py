"""Metrics and histogram construction for laser-data comparisons."""

import numpy as np
from tqdm import tqdm

from . import compare_sim as base

# J26.3.2 QE_shape(532 nm) times effective CE, weighted by installed CD-LPMT
# categories, is 0.0967717 (the QE_shape curve is unchanged in J26.5.0).
# Keep three significant figures for this temporary common factor until
# LUCiD supplies per-PMT QE.
JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM = 0.097

estimate_peak_from_density = base.estimate_peak_from_density
estimate_peak_from_times = base.estimate_peak_from_times
density_hist = base.density_hist
l1_distance = base.l1_distance
wasserstein_distance = base.wasserstein_distance
convolve_density = base.convolve_density
convolve_kernel = base.convolve_kernel


def simulation_qe_factors(
    sim_data_list,
    apply_lucid_qe=True,
    lucid_qe_eff=JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
):
    """Return per-source signal factors, applying QE only to LUCiD.

    Input: simulation source caches, the LUCiD QE toggle and the common
    532 nm equivalent QE.  Output: float64 ndarray carrying lucid_qe_eff
    on LUCiD sources when the correction is enabled, 1 elsewhere.
    """
    if not np.isfinite(lucid_qe_eff) or not 0.0 < lucid_qe_eff <= 1.0:
        raise ValueError("lucid_qe_eff must be finite and in (0, 1]")
    factors = np.ones(len(sim_data_list), dtype=np.float64)
    if apply_lucid_qe:
        kinds = np.asarray(
            [
                str(np.asarray(item.get("kind", "")).item())
                for item in sim_data_list
            ]
        )
        factors[np.char.lower(kinds) == "lucid"] = lucid_qe_eff
    return factors


def processing_string(
    use_tts,
    irf,
    add_dark_to_simulation,
    target_pe=None,
    prompt_window=(-10.0, 20.0),
    late_window=(20.0, 600.0),
    alignment="l1",
    source_intensity_scale=None,
):
    """Serialize all response-processing settings into a cache key.

    Input: the effective TTS toggle and the IRF descriptor ("0" when
    the measured IRF is off, otherwise the file-based descriptor).
    Output: the aligned-cache processing key.
    """
    value = (
        # aligned-cache schema version; bump when result fields change
        "metrics=7,"
        f"tts={int(use_tts)},"
        f"irf={irf},"
        f"dark_to_sim={int(add_dark_to_simulation)},fsmp_dark=kept,"
        f"alignment=global_median_consensus_{alignment},"
        f"prompt={prompt_window[0]:g}:{prompt_window[1]:g},"
        f"late={late_window[0]:g}:{late_window[1]:g}"
    )
    if target_pe is not None:
        value += f",fsmp_target={int(target_pe)}"
    if source_intensity_scale is not None:
        scales = ":".join(f"{scale:g}" for scale in source_intensity_scale)
        value += f",source_scale={scales}"
    return value


def _records_by_channel(data):
    """Group FSMP cache arrays while retaining event and PE metadata."""
    channel = np.asarray(data["fsmp_ch"], dtype=np.int64)
    order = np.argsort(channel, kind="mergesort")
    sorted_channel = channel[order]
    unique, starts = np.unique(sorted_channel, return_index=True)
    stops = np.r_[starts[1:], len(order)]
    eid = np.asarray(
        data.get("fsmp_eid", np.arange(len(channel))), dtype=np.int64
    )
    count = np.asarray(
        data.get("fsmp_pe_count", np.ones(len(channel))), dtype=np.int64
    )
    return {
        int(value): {
            "time": np.asarray(data["fsmp_time"])[order[start:stop]],
            "eid": eid[order[start:stop]],
            "pe_count": count[order[start:stop]],
        }
        for value, start, stop in zip(unique, starts, stops)
    }


def estimate_dark_window(times, peak, margin_ns):
    """Return the finite pre-peak interval used for dark-rate estimation."""
    values = np.asarray(times, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(peak):
        return None
    low, high = float(np.min(values)), float(peak - margin_ns)
    return None if high <= low else (low, high)


def _empty_dark_stats():
    return {
        "prepeak_pe": 0,
        "rate_per_ns": 0.0,
        "rate_pe_per_event_per_ns": 0.0,
        "expected_pe_per_event_in_histogram": 0.0,
        "window_ns": 0.0,
        "window_low_ns": np.nan,
        "window_high_ns": np.nan,
        "event_samples": 0,
    }


def estimate_dark_noise(times, peak, margin_ns, edges, event_samples):
    """Estimate the pre-peak FSMP dark rate without modifying FSMP.

    Input: FSMP PE times, the FSMP peak, the pre-peak sideband margin,
    histogram edges and the event-sample exposure.
    Output: dark statistics; the per-event rate is added to simulation
    sources by the caller while the FSMP spectrum stays untouched.
    """
    stats = _empty_dark_stats()
    stats["event_samples"] = int(event_samples)
    window = estimate_dark_window(times, peak, margin_ns)
    if window is None or event_samples <= 0:
        return stats
    low, high = window
    width = high - low
    prepeak = int(np.count_nonzero((times >= low) & (times < high)))
    rate = prepeak / width
    rate_per_event = rate / event_samples
    stats.update(
        {
            "prepeak_pe": prepeak,
            "rate_per_ns": float(rate),
            "rate_pe_per_event_per_ns": float(rate_per_event),
            "expected_pe_per_event_in_histogram": float(
                rate_per_event * (edges[-1] - edges[0])
            ),
            "window_ns": float(width),
            "window_low_ns": low,
            "window_high_ns": high,
        }
    )
    return stats


def _source_arrays(data):
    """Normalize source cache fields to ``[source, channel, ...]``."""
    density = np.asarray(data["simulation_density"], dtype=np.float64)
    raw = np.asarray(
        data.get("simulation_raw_density", density), dtype=np.float64
    )
    entries = np.asarray(data["simulation_entries"], dtype=np.int64)
    peaks = np.asarray(
        data.get("simulation_peak", np.empty(0)), dtype=np.float64
    )
    if density.ndim == 2:
        density, raw, entries = density[None], raw[None], entries[None]
        if peaks.size:
            peaks = peaks[None]
    if not peaks.size:
        peaks = np.full(entries.shape, np.nan)
    labels = data.get("simulation_labels")
    if labels is None:
        labels = np.asarray(
            [str(np.asarray(data.get("label", "Simulation")).item())]
        )
    labels = np.asarray(labels).astype(str)
    if len(labels) != density.shape[0]:
        labels = np.asarray(
            [f"Simulation {i + 1}" for i in range(density.shape[0])]
        )
    return labels, raw, density, entries, peaks


def _metric_arrays(data):
    """Return labels, densities and source-by-channel distance arrays."""
    labels, _, density, _, _ = _source_arrays(data)
    run = np.asarray(data["run_density"], dtype=np.float64)
    if run.ndim == 2:
        run = run[None]
    width = float(data["edges"][1] - data["edges"][0])
    l1 = np.asarray(data.get("l1_distance", []), dtype=np.float64)
    wasserstein = np.asarray(
        data.get("wasserstein_distance", []), dtype=np.float64
    )
    if l1.size == 0:
        l1 = np.asarray(
            [
                [l1_distance(a, b, width) for a, b in zip(run[i], density[i])]
                for i in range(len(labels))
            ]
        )
    if wasserstein.size == 0:
        wasserstein = np.asarray(
            [
                [
                    wasserstein_distance(a, b, width)
                    for a, b in zip(run[i], density[i])
                ]
                for i in range(len(labels))
            ]
        )
    if l1.ndim == 1:
        l1 = l1[None]
    if wasserstein.ndim == 1:
        wasserstein = wasserstein[None]
    return labels, run, density, l1, wasserstein


def _source_counts(data, labels):
    """Normalize count fields to source-by-channel arrays."""
    _, _, density, entries, _ = _source_arrays(data)
    shape = density.shape[:2]

    def get(name, fallback, fill=0):
        value = np.asarray(data.get(name, data.get(fallback)), dtype=np.int64)
        if value.ndim == 1:
            value = value[None]
        if value.size == 0:
            return np.full(shape, fill, dtype=np.int64)
        if value.shape != shape:
            value = np.broadcast_to(value, shape).copy()
        return value

    return (
        get("run_entries", "simulation_entries"),
        get("run_events", "run_entries"),
        get("run_total_pe", "run_entries"),
        entries,
        get("simulation_events", "simulation_entries", -1),
        get("simulation_total_pe", "simulation_entries"),
    )


def _normalize_with_factor(density, width):
    """Return a unit-area density and its multiplicative coefficient."""
    area = float(np.sum(density) * width)
    factor = 1.0 / area if area > 0.0 else 0.0
    return np.asarray(density, dtype=np.float64) * factor, factor


def consensus_l1_alignment(
    times,
    simulation_densities,
    edges,
    coarse_shift,
    window=10.0,
):
    """Estimate one channel's candidate shift against equal-weight sources."""
    densities = np.asarray(simulation_densities, dtype=np.float64)
    if densities.ndim == 1:
        densities = densities[None, :]
    valid = np.all(np.isfinite(densities), axis=1)
    valid &= np.any(densities > 0.0, axis=1)
    densities = densities[valid]
    width = float(edges[1] - edges[0])
    if not np.isfinite(coarse_shift):
        coarse_shift = 0.0
    if not len(densities):
        aligned = density_hist(times + coarse_shift, edges)
        return float(coarse_shift), aligned, np.nan

    def objective(candidate):
        """Return aligned density and order-invariant mean source L1."""
        aligned = density_hist(times + candidate, edges)
        distances = np.asarray(
            [l1_distance(aligned, density, width) for density in densities]
        )
        return aligned, float(np.mean(np.sort(distances)))

    step = max(width / 10.0, 0.01)
    offsets = np.arange(-window, window + step / 2.0, step)
    best_shift = float(coarse_shift)
    best_density, best_l1 = objective(best_shift)
    for offset in offsets:
        candidate_shift = float(coarse_shift + offset)
        candidate_density, candidate_l1 = objective(candidate_shift)
        if np.isnan(best_l1) or candidate_l1 < best_l1:
            best_shift = candidate_shift
            best_density = candidate_density
            best_l1 = candidate_l1
    return best_shift, best_density, best_l1


def _median_simulation_peak(centers, simulation_densities):
    """Return the source-order-independent median of valid peak positions."""
    peaks = np.asarray(
        [
            estimate_peak_from_density(centers, density)
            for density in simulation_densities
        ],
        dtype=np.float64,
    )
    finite = peaks[np.isfinite(peaks)]
    return (float(np.median(finite)) if finite.size else np.nan), peaks


def _window_integral(density, centers, width, window, peak):
    """Integrate a PE/event density in a window relative to ``peak``."""
    if not np.isfinite(peak):
        return np.nan
    low, high = peak + window[0], peak + window[1]
    mask = (centers >= low) & (centers < high)
    return float(np.sum(np.asarray(density)[mask]) * width)


def _prompt_late_ratio(prompt, late):
    """Return prompt/late or NaN when the late integral is not positive."""
    return float(prompt / late) if late > 0.0 else np.nan


def build_channel_histograms_multi(
    fsmp_data,
    sim_data_list,
    alignment="l1",
    use_tts=True,
    irf_kernels=None,
    irf_fwhm_by_channel=None,
    irf_description="0",
    add_dark_to_simulation=True,
    dark_window=30.0,
    prompt_window=(-10.0, 20.0),
    late_window=(20.0, 600.0),
    simulation_source_scale=None,
):
    """Build diagnostics after applying one robust shift to every channel.

    Input: FSMP/simulation caches; when irf_kernels (channel ->
    unit-sum kernel) is given, the per-PMT TTS convolution is disabled
    because the measured IRF already contains the TTS.  Output: the
    aligned per-channel metrics dictionary.
    """
    if alignment not in {"l1", "peak"} or not sim_data_list:
        raise ValueError(
            "alignment must be l1/peak and at least one source is required"
        )
    for name, window in (("prompt", prompt_window), ("late", late_window)):
        if len(window) != 2 or window[1] <= window[0]:
            raise ValueError(f"{name} window must satisfy LOW < HIGH")
    channels = np.asarray(fsmp_data["channels"], dtype=np.int64)
    pmt_type = np.asarray(fsmp_data["pmt_type"], dtype=np.int16)
    tts = np.asarray(fsmp_data["tts"], dtype=np.float64)
    if irf_kernels is not None:
        # The measured IRF kernels already contain the TTS.
        use_tts = False
    fwhm_by_channel = irf_fwhm_by_channel or {}
    irf = np.asarray(
        [fwhm_by_channel.get(int(value), 0.0) for value in channels],
        dtype=np.float64,
    )
    edges = np.asarray(sim_data_list[0]["edges"], dtype=np.float64)
    centers = np.asarray(sim_data_list[0]["centers"], dtype=np.float64)
    if any(
        not np.array_equal(edges, source["edges"])
        for source in sim_data_list[1:]
    ):
        raise ValueError("simulation sources use different histogram edges")
    width = float(edges[1] - edges[0])
    records = _records_by_channel(fsmp_data)
    n_sources = len(sim_data_list)
    source_scale = np.ones(n_sources, dtype=np.float64)
    if simulation_source_scale is not None:
        source_scale = np.asarray(
            simulation_source_scale, dtype=np.float64
        )
        if (
            source_scale.shape != (n_sources,)
            or not np.all(np.isfinite(source_scale))
            or np.any(source_scale <= 0.0)
        ):
            raise ValueError(
                "simulation_source_scale must be positive per source"
            )
    physical_events = int(
        np.asarray(
            fsmp_data.get(
                "fsmp_physical_events",
                len(np.unique(fsmp_data.get("fsmp_eid", []))),
            )
        ).item()
    )
    event_samples = int(
        np.asarray(fsmp_data.get("fsmp_event_samples", physical_events)).item()
    )
    values = {
        name: []
        for name in (
            "run_density",
            "raw_density",
            "sim_density",
            "run_pe_density",
            "sim_pe_density",
            "run_peak",
            "aligned_run_peak",
            "sim_peak",
            "peak_difference",
            "run_entries",
            "sim_entries",
            "run_events",
            "run_event_samples",
            "run_total",
            "sim_events",
            "sim_total",
            "run_norm_pe",
            "sim_norm_pe",
            "run_renorm",
            "sim_renorm",
            "run_mean",
            "sim_mean",
            "run_prompt",
            "sim_prompt",
            "run_late",
            "sim_late",
            "run_ratio",
            "sim_ratio",
            "gate_peak",
            "prompt_effective_low",
            "prompt_effective_high",
            "late_effective_low",
            "late_effective_high",
            "l1",
            "wasserstein",
        )
    }
    dark_stats = []
    empty_record = {
        "time": np.empty(0),
        "eid": np.empty(0),
        "pe_count": np.empty(0),
    }
    prepared = []
    unsmoothed = []
    for channel, channel_tts, channel_irf in tqdm(
        list(zip(channels, tts, irf)),
        desc="Prepare channel models",
    ):
        record = records.get(int(channel), empty_record)
        times = np.asarray(record["time"], dtype=np.float64)
        peak_run = estimate_peak_from_times(times, len(edges) - 1)
        kernel = None
        if not use_tts and irf_kernels is not None:
            kernel = irf_kernels.get(int(channel))
            if kernel is None:
                unsmoothed.append(int(channel))
        source_results = []
        for model_index, source in enumerate(sim_data_list):
            source_channels = np.asarray(source["channels"], dtype=np.int64)
            source_index = np.searchsorted(source_channels, channel)
            if (
                source_index >= len(source_channels)
                or source_channels[source_index] != channel
            ):
                raw = np.zeros(len(edges) - 1)
                entries = total = event_count = 0
            else:
                raw = np.asarray(
                    source["simulation_raw_density"], dtype=np.float64
                )[source_index]
                entries = int(
                    np.asarray(source["simulation_entries"])[source_index]
                )
                total = int(
                    np.asarray(
                        source.get(
                            "simulation_total_pe",
                            source["simulation_entries"],
                        )
                    )[source_index]
                )
                event_count = int(
                    np.asarray(source["simulation_events"])[source_index]
                )
            observed = raw
            if use_tts:
                observed = convolve_density(observed, width, channel_tts)
            elif kernel is not None:
                observed = convolve_kernel(observed, kernel)
            shape, renorm = _normalize_with_factor(observed, width)
            if event_count > 0:
                pe_density = (
                    observed
                    * entries
                    / event_count
                    * source_scale[model_index]
                )
            else:
                pe_density = np.full_like(observed, np.nan)
            source_results.append(
                {
                    "raw": raw,
                    "shape": shape,
                    "pe_density": pe_density,
                    "entries": entries,
                    "total": total,
                    "events": event_count,
                    "renorm": renorm,
                }
            )
        reference_peak, simulation_peaks = _median_simulation_peak(
            centers,
            [item["shape"] for item in source_results],
        )
        if not np.isfinite(peak_run) or not np.isfinite(reference_peak):
            candidate_shift = np.nan
        elif alignment == "l1":
            coarse_shift = float(reference_peak - peak_run)
            candidate_shift, _, _ = consensus_l1_alignment(
                times,
                [item["shape"] for item in source_results],
                edges,
                coarse_shift,
            )
        else:
            candidate_shift = float(reference_peak - peak_run)
        prepared.append(
            {
                "channel": channel,
                "tts": channel_tts,
                "irf": channel_irf,
                "record": record,
                "times": times,
                "peak_run": peak_run,
                "source_results": source_results,
                "reference_peak": reference_peak,
                "simulation_peaks": simulation_peaks,
                "candidate_shift": candidate_shift,
            }
        )

    if irf_kernels is not None and unsmoothed:
        print(
            f"IRF: {len(unsmoothed)} channels without a measured "
            "kernel left unsmoothed"
        )
    candidate_shifts = np.asarray(
        [item["candidate_shift"] for item in prepared], dtype=np.float64
    )
    finite_candidates = candidate_shifts[np.isfinite(candidate_shifts)]
    global_shift = (
        float(np.median(finite_candidates))
        if finite_candidates.size
        else 0.0
    )
    pmt_shift_types = np.unique(pmt_type)
    pmt_shift_medians = np.asarray(
        [
            np.median(candidate_shifts[(pmt_type == value) & np.isfinite(
                candidate_shifts
            )])
            if np.any((pmt_type == value) & np.isfinite(candidate_shifts))
            else np.nan
            for value in pmt_shift_types
        ],
        dtype=np.float64,
    )

    for item in tqdm(prepared, desc="Build channel histograms"):
        channel = item["channel"]
        channel_tts = item["tts"]
        channel_irf = item["irf"]
        record = item["record"]
        times = item["times"]
        peak_run = item["peak_run"]
        source_results = item["source_results"]
        reference_peak = item["reference_peak"]
        simulation_peaks = item["simulation_peaks"]
        shift = global_shift
        aligned = density_hist(times + shift, edges)
        normalization = int(np.histogram(times + shift, bins=edges)[0].sum())
        stats = estimate_dark_noise(
            times, peak_run, dark_window, edges, event_samples
        )
        dark_density = (
            stats["rate_pe_per_event_per_ns"]
            if add_dark_to_simulation
            else 0.0
        )
        final_sources = []
        for source in source_results:
            pe_density = source["pe_density"] + dark_density
            shape, renorm = _normalize_with_factor(pe_density, width)
            final_sources.append(
                {
                    **source,
                    "pe_density": pe_density,
                    "shape": shape,
                    "renorm": renorm,
                }
            )
        source_results = final_sources
        if event_samples > 0:
            run_pe_density = aligned * normalization / event_samples
        else:
            run_pe_density = np.full_like(aligned, np.nan)
        run_shape, run_renorm = _normalize_with_factor(run_pe_density, width)
        aligned_run_peak = estimate_peak_from_density(centers, run_shape)
        run_mean = float(np.sum(run_pe_density) * width)
        run_prompt = _window_integral(
            run_pe_density, centers, width, prompt_window, reference_peak
        )
        run_late = _window_integral(
            run_pe_density, centers, width, late_window, reference_peak
        )
        sim_mean = [
            float(np.sum(item["pe_density"]) * width)
            for item in source_results
        ]
        sim_prompt = [
            _window_integral(
                item["pe_density"],
                centers,
                width,
                prompt_window,
                reference_peak,
            )
            for item in source_results
        ]
        sim_late = [
            _window_integral(
                item["pe_density"],
                centers,
                width,
                late_window,
                reference_peak,
            )
            for item in source_results
        ]
        if np.isfinite(reference_peak):
            prompt_low = max(edges[0], reference_peak + prompt_window[0])
            prompt_high = min(edges[-1], reference_peak + prompt_window[1])
            late_low = max(edges[0], reference_peak + late_window[0])
            late_high = min(edges[-1], reference_peak + late_window[1])
        else:
            prompt_low = prompt_high = np.nan
            late_low = late_high = np.nan
        dark_stats.append(stats)
        values["run_density"].append([run_shape] * n_sources)
        values["raw_density"].append(
            [item["raw"] for item in source_results]
        )
        values["sim_density"].append(
            [item["shape"] for item in source_results]
        )
        values["run_pe_density"].append([run_pe_density] * n_sources)
        values["sim_pe_density"].append(
            [item["pe_density"] for item in source_results]
        )
        values["run_peak"].append([peak_run] * n_sources)
        values["aligned_run_peak"].append(
            [aligned_run_peak] * n_sources
        )
        values["sim_peak"].append(simulation_peaks)
        values["peak_difference"].append(
            simulation_peaks - aligned_run_peak
        )
        values["run_entries"].append([len(times)] * n_sources)
        values["sim_entries"].append(
            [item["entries"] for item in source_results]
        )
        values["run_events"].append([physical_events] * n_sources)
        values["run_event_samples"].append([event_samples] * n_sources)
        values["run_total"].append(
            [int(np.sum(record["pe_count"]))] * n_sources
        )
        values["sim_events"].append(
            [item["events"] for item in source_results]
        )
        values["sim_total"].append(
            [item["total"] for item in source_results]
        )
        values["run_norm_pe"].append([normalization] * n_sources)
        values["sim_norm_pe"].append(
            [item["entries"] for item in source_results]
        )
        values["run_renorm"].append([run_renorm] * n_sources)
        values["sim_renorm"].append(
            [item["renorm"] for item in source_results]
        )
        values["run_mean"].append([run_mean] * n_sources)
        values["sim_mean"].append(sim_mean)
        values["run_prompt"].append([run_prompt] * n_sources)
        values["sim_prompt"].append(sim_prompt)
        values["run_late"].append([run_late] * n_sources)
        values["sim_late"].append(sim_late)
        values["run_ratio"].append(
            [_prompt_late_ratio(run_prompt, run_late)] * n_sources
        )
        values["sim_ratio"].append(
            [
                _prompt_late_ratio(prompt, late)
                for prompt, late in zip(sim_prompt, sim_late)
            ]
        )
        values["gate_peak"].append([reference_peak] * n_sources)
        values["prompt_effective_low"].append([prompt_low] * n_sources)
        values["prompt_effective_high"].append([prompt_high] * n_sources)
        values["late_effective_low"].append([late_low] * n_sources)
        values["late_effective_high"].append([late_high] * n_sources)
        values["l1"].append(
            [
                l1_distance(run_shape, item["shape"], width)
                for item in source_results
            ]
        )
        values["wasserstein"].append(
            [
                wasserstein_distance(run_shape, item["shape"], width)
                for item in source_results
            ]
        )

    def transpose(name, dtype=float):
        """Convert one channel-major list to a source-major array."""
        array = np.asarray(values[name], dtype=dtype)
        return array.transpose(1, 0, 2) if array.ndim == 3 else array.T

    result = {
        "channels": channels,
        "pmt_type": pmt_type,
        "tts": tts,
        "irf_fwhm_by_channel": irf,
        "z": np.asarray(fsmp_data["z"], dtype=float),
        "theta": np.asarray(fsmp_data["theta"], dtype=float),
        "edges": edges,
        "centers": centers,
        "run_density": transpose("run_density"),
        "simulation_raw_density": transpose("raw_density"),
        "simulation_density": transpose("sim_density"),
        "run_pe_per_event_density": transpose("run_pe_density"),
        "simulation_pe_per_event_density": transpose("sim_pe_density"),
        "run_peak": transpose("run_peak"),
        "fsmp_peak_unaligned_ns": transpose("run_peak"),
        "aligned_fsmp_peak_ns": transpose("aligned_run_peak"),
        "alignment_shift_ns": np.asarray(global_shift),
        "channel_alignment_candidate_shift_ns": candidate_shifts,
        "alignment_pmt_types": pmt_shift_types,
        "alignment_pmt_type_candidate_median_ns": pmt_shift_medians,
        "simulation_peak": transpose("sim_peak"),
        "simulation_minus_fsmp_aligned_peak_ns": transpose(
            "peak_difference"
        ),
        "run_entries": transpose("run_entries", np.int64),
        "simulation_entries": transpose("sim_entries", np.int64),
        "run_events": transpose("run_events", np.int64),
        "run_event_samples": transpose("run_event_samples", np.int64),
        "run_total_pe": transpose("run_total", np.int64),
        "simulation_events": transpose("sim_events", np.int64),
        "simulation_total_pe": transpose("sim_total", np.int64),
        "run_shape_normalization_pe": transpose("run_norm_pe", np.int64),
        "simulation_shape_normalization_pe": transpose(
            "sim_norm_pe", np.int64
        ),
        "run_shape_renormalization": transpose("run_renorm"),
        "simulation_shape_renormalization": transpose("sim_renorm"),
        "run_shape_c_event_per_pe": transpose("run_renorm"),
        "simulation_shape_c_event_per_pe": transpose("sim_renorm"),
        "run_mean_pe_per_event": transpose("run_mean"),
        "simulation_mean_pe_per_event": transpose("sim_mean"),
        "run_prompt_pe_per_event": transpose("run_prompt"),
        "simulation_prompt_pe_per_event": transpose("sim_prompt"),
        "run_late_pe_per_event": transpose("run_late"),
        "simulation_late_pe_per_event": transpose("sim_late"),
        "run_prompt_late_ratio": transpose("run_ratio"),
        "simulation_prompt_late_ratio": transpose("sim_ratio"),
        "gate_reference_peak_ns": transpose("gate_peak"),
        "prompt_window_effective_low_ns": transpose(
            "prompt_effective_low"
        ),
        "prompt_window_effective_high_ns": transpose(
            "prompt_effective_high"
        ),
        "late_window_effective_low_ns": transpose("late_effective_low"),
        "late_window_effective_high_ns": transpose("late_effective_high"),
        "l1_distance": transpose("l1"),
        "wasserstein_distance": transpose("wasserstein"),
        "prompt_window_relative_ns": np.asarray(prompt_window, dtype=float),
        "late_window_relative_ns": np.asarray(late_window, dtype=float),
        "alignment_method": np.asarray(
            f"global_median_consensus_{alignment}"
        ),
        "simulation_labels": np.asarray(
            [
                str(np.asarray(source.get("label", "Simulation")).item())
                for source in sim_data_list
            ]
        ),
        "simulation_kinds": np.asarray(
            [
                str(np.asarray(source.get("kind", "simulation")).item())
                for source in sim_data_list
            ]
        ),
        "simulation_applied_source_scale": source_scale,
        "simulation_dark_added": np.asarray(add_dark_to_simulation),
        "processing": np.asarray(
            processing_string(
                use_tts,
                irf_description,
                add_dark_to_simulation,
                target_pe=fsmp_data.get("fsmp_target_pe"),
                prompt_window=prompt_window,
                late_window=late_window,
                alignment=alignment,
                source_intensity_scale=source_scale,
            )
        ),
    }
    for key in (
        "prepeak_pe",
        "rate_per_ns",
        "rate_pe_per_event_per_ns",
        "expected_pe_per_event_in_histogram",
        "window_ns",
        "window_low_ns",
        "window_high_ns",
        "event_samples",
    ):
        item_values = np.asarray([item[key] for item in dark_stats])
        integer = key in {"prepeak_pe", "event_samples"}
        dtype = np.int64 if integer else float
        result[f"run_dark_{key}"] = np.repeat(
            item_values.astype(dtype)[None, :], n_sources, axis=0
        )
    return result


def merge_smearing_variants(tts_data, irf_data, processing=None):
    """Merge TTS/IRF builds of the same sources into one dictionary.

    Input: two results of build_channel_histograms_multi computed on
    the same FSMP data and simulation sources, the first with TTS and
    the second with measured-IRF smearing; and the processing string
    to stamp on the merged result.
    Output: merged metrics dictionary; every simulation-side source
    axis carries, per original source, the TTS variant then the IRF
    variant; run-side and channel-level fields come from the TTS
    build.
    """
    tts_labels = [
        str(np.asarray(label).item())
        for label in np.atleast_1d(tts_data["simulation_labels"])
    ]
    irf_labels = [
        str(np.asarray(label).item())
        for label in np.atleast_1d(irf_data["simulation_labels"])
    ]
    if tts_labels != irf_labels:
        raise ValueError("TTS and IRF builds describe different sources")
    if not np.array_equal(tts_data["channels"], irf_data["channels"]):
        raise ValueError("TTS and IRF builds use different channels")
    if not np.array_equal(tts_data["edges"], irf_data["edges"]):
        raise ValueError("TTS and IRF builds use different edges")
    n_sources = len(tts_labels)

    def _interleave(first, second):
        """Stack [first_i, second_i] pairs along a new source axis.

        Input: two arrays whose first axis counts the sources.
        Output: one array with 2 * n_sources rows, TTS then IRF.
        """
        parts = []
        for index in range(n_sources):
            parts.append(np.asarray(first[index]))
            parts.append(np.asarray(second[index]))
        return np.asarray(parts)

    variant_keys = {
        "l1_distance",
        "wasserstein_distance",
        "gate_reference_peak_ns",
        "fsmp_peak_unaligned_ns",
        "aligned_fsmp_peak_ns",
        "prompt_window_effective_low_ns",
        "prompt_window_effective_high_ns",
        "late_window_effective_low_ns",
        "late_window_effective_high_ns",
    }
    copied_arrays = {
        "channels",
        "pmt_type",
        "z",
        "theta",
        "tts",
        "irf_fwhm_by_channel",
        "edges",
        "centers",
        "channel_alignment_candidate_shift_ns",
        "alignment_pmt_types",
        "alignment_pmt_type_candidate_median_ns",
        "prompt_window_relative_ns",
        "late_window_relative_ns",
    }
    merged = {}
    for key, value in tts_data.items():
        array = np.asarray(value)
        is_variant_axis = (
            (key.startswith(("simulation_", "run_")) or key in variant_keys)
            and array.ndim >= 1
            and array.shape[0] == n_sources
        )
        if key == "simulation_labels":
            merged[key] = np.asarray(
                [
                    f"{label} ({variant})"
                    for label in tts_labels
                    for variant in ("TTS", "IRF")
                ]
            )
        elif key == "processing":
            merged[key] = np.asarray(
                processing if processing is not None else value
            )
        elif is_variant_axis:
            merged[key] = _interleave(array, np.asarray(irf_data[key]))
        else:
            if (
                array.ndim >= 1
                and array.shape[0] == n_sources
                and key not in copied_arrays
            ):
                raise ValueError(
                    "merge_smearing_variants: no merge rule for "
                    f"source-axis key {key!r}"
                )
            merged[key] = value
    return merged
