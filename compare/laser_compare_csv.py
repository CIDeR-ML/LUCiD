"""Machine-readable CSV outputs for laser-data comparisons."""

import csv

import numpy as np

from . import compare_sim as base
from .laser_compare_metrics import (
    _metric_arrays,
    _source_counts,
)

PMT_TYPE_NAME = base.PMT_TYPE_NAME


def _shape_coefficient(normalization_pe, bin_width_ns, renormalization):
    """Return the final count-to-unit-area-density coefficient."""
    if normalization_pe <= 0 or bin_width_ns <= 0.0:
        return np.nan
    return float(renormalization / (normalization_pe * bin_width_ns))


def _processing_title(data):
    raw = str(np.asarray(data.get("processing", "")).item())
    flags = dict(item.split("=", 1) for item in raw.split(",") if "=" in item)
    irf = flags.get("irf", "unknown")
    if irf.startswith("file:"):
        parts = irf.split(":")
        irf_title = (
            f"measured per-channel IRF ({parts[1]}, "
            f"{parts[2][2:]} channels)"
        )
    elif "|" in irf:
        values = dict(item.split(":", 1) for item in irf.split("|"))
        irf_title = (
            f"IRF sigma: Hamamatsu={values.get('hamamatsu', '?')} ns, "
            f"NNVT={values.get('nnvt', '?')} ns"
        )
    else:
        irf_title = f"IRF sigma={irf} ns"
    tts = flags.get("tts", "unknown")
    dark = flags.get("dark_to_sim", "unknown")
    alignment = flags.get("alignment", "unknown").replace("_", " ")
    tts_text = "on" if tts == "1" else "off" if tts == "0" else tts
    dark_text = "on" if dark == "1" else "off" if dark == "0" else dark
    return (
        f"simulation TTS={tts_text}; {irf_title}; "
        f"FSMP dark kept; MC dark addition={dark_text}; alignment={alignment}"
    )


def write_channel_manifest(
    path,
    data,
    run,
    source_z,
    irf_fwhm_by_channel=None,
):
    """Write selected channel metadata, including the applied IRF FWHM.

    Input: output path, FSMP data, run/source identifiers and the
    per-channel kernel FWHM mapping (empty when the IRF is off).
    Output: none (CSV written).
    """
    channels = np.asarray(data["channels"], dtype=np.int64)
    fwhm = irf_fwhm_by_channel or {}
    irf = np.asarray(
        [fwhm.get(int(channel), 0.0) for channel in channels],
        dtype=np.float64,
    )
    fields = [
        "source_z_m",
        "run",
        "channel",
        "pmt_type",
        "pmt_type_name",
        "z_mm",
        "theta_deg",
        "tts_ns",
        "irf_fwhm_ns",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, channel in enumerate(channels):
            writer.writerow(
                {
                    "source_z_m": float(source_z),
                    "run": int(run),
                    "channel": int(channel),
                    "pmt_type": int(data["pmt_type"][index]),
                    "pmt_type_name": PMT_TYPE_NAME.get(
                        int(data["pmt_type"][index]), "Unknown"
                    ),
                    "z_mm": float(data["z"][index]),
                    "theta_deg": float(data["theta"][index]),
                    "tts_ns": float(data["tts"][index]),
                    "irf_fwhm_ns": float(irf[index]),
                }
            )
    print(f"Wrote {path}")


def _source_slug(label, index):
    slug = "".join(
        char.lower() if char.isalnum() else "_" for char in str(label)
    ).strip("_")
    return slug or f"simulation_{index + 1}"


def write_comparison_stats_csv(path, data, run, source_z):
    """Write per-channel metrics, counts and dark-noise audit fields."""
    labels, _, _, l1, wasserstein = _metric_arrays(data)
    entries, events, total_pe, sim_entries, sim_events, sim_total = (
        _source_counts(data, labels)
    )
    processing = _processing_title(data)
    width = float(data["edges"][1] - data["edges"][0])
    prompt_window = np.asarray(data["prompt_window_relative_ns"])
    late_window = np.asarray(data["late_window_relative_ns"])
    run_norm = np.asarray(data["run_shape_normalization_pe"])
    sim_norm = np.asarray(data["simulation_shape_normalization_pe"])
    run_renorm = np.asarray(data["run_shape_renormalization"])
    sim_renorm = np.asarray(data["simulation_shape_renormalization"])
    fields = [
        "run",
        "source_z_m",
        "channel",
        "pmt_type",
        "z_mm",
        "theta_deg",
        "tts_ns",
        "irf_fwhm_ns",
        "processing",
        "alignment_method",
        "alignment_shift_ns",
        "channel_alignment_candidate_shift_ns",
        "fsmp_peak_unaligned_ns",
        "fsmp_peak_aligned_ns",
        "pe_expectation_definition",
        "fsmp_events",
        "fsmp_event_samples",
        "fsmp_pe",
        "fsmp_time_samples",
        "fsmp_mean_pe_per_event",
        "fsmp_prompt_pe_per_event",
        "fsmp_late_pe_per_event",
        "fsmp_prompt_late_ratio",
        "fsmp_shape_normalization_pe",
        "fsmp_shape_normalization_coefficient_per_pe_ns",
        "prompt_window_low_relative_ns",
        "prompt_window_high_relative_ns",
        "late_window_low_relative_ns",
        "late_window_high_relative_ns",
        "gate_reference_peak_ns",
        "prompt_window_effective_low_ns",
        "prompt_window_effective_high_ns",
        "late_window_effective_low_ns",
        "late_window_effective_high_ns",
    ]
    for index, label in enumerate(labels):
        prefix = _source_slug(label, index)
        fields.extend(
            [
                f"{prefix}_events",
                f"{prefix}_pe",
                f"{prefix}_time_samples",
                f"{prefix}_mean_pe_per_event",
                f"{prefix}_mean_pe_ratio_to_fsmp",
                f"{prefix}_prompt_pe_per_event",
                f"{prefix}_late_pe_per_event",
                f"{prefix}_prompt_late_ratio",
                f"{prefix}_shape_normalization_pe",
                f"{prefix}_shape_normalization_coefficient_per_pe_ns",
                f"{prefix}_l1",
                f"{prefix}_wasserstein_ns",
                f"{prefix}_peak_ns",
                f"{prefix}_minus_fsmp_aligned_peak_ns",
                f"{prefix}_applied_source_scale",
            ]
        )
    fields.extend(
        f"fsmp_dark_{key}"
        for key in (
            "prepeak_pe",
            "rate_per_ns",
            "rate_pe_per_event_per_ns",
            "expected_pe_per_event_in_histogram",
            "window_ns",
            "window_low_ns",
            "window_high_ns",
            "event_samples",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for channel_index, channel in enumerate(data["channels"]):
            row = {
                "run": int(run),
                "source_z_m": float(source_z),
                "channel": int(channel),
                "pmt_type": int(data["pmt_type"][channel_index]),
                "z_mm": float(data["z"][channel_index]),
                "theta_deg": float(data["theta"][channel_index]),
                "tts_ns": float(data["tts"][channel_index]),
                "irf_fwhm_ns": float(
                    data["irf_fwhm_by_channel"][channel_index]
                ),
                "processing": processing,
                "alignment_method": str(
                    np.asarray(data["alignment_method"]).item()
                ),
                "alignment_shift_ns": float(
                    np.asarray(data["alignment_shift_ns"]).item()
                ),
                "channel_alignment_candidate_shift_ns": float(
                    data["channel_alignment_candidate_shift_ns"][
                        channel_index
                    ]
                ),
                "fsmp_peak_unaligned_ns": float(
                    data["fsmp_peak_unaligned_ns"][0, channel_index]
                ),
                "fsmp_peak_aligned_ns": float(
                    data["aligned_fsmp_peak_ns"][0, channel_index]
                ),
                "pe_expectation_definition": (
                    "empirical mean PE in histogram range per exposure"
                ),
                "fsmp_events": int(events[0, channel_index]),
                "fsmp_event_samples": int(
                    data["run_event_samples"][0, channel_index]
                ),
                "fsmp_pe": int(total_pe[0, channel_index]),
                "fsmp_time_samples": int(entries[0, channel_index]),
                "fsmp_mean_pe_per_event": float(
                    data["run_mean_pe_per_event"][0, channel_index]
                ),
                "fsmp_prompt_pe_per_event": float(
                    data["run_prompt_pe_per_event"][0, channel_index]
                ),
                "fsmp_late_pe_per_event": float(
                    data["run_late_pe_per_event"][0, channel_index]
                ),
                "fsmp_prompt_late_ratio": float(
                    data["run_prompt_late_ratio"][0, channel_index]
                ),
                "fsmp_shape_normalization_pe": int(
                    run_norm[0, channel_index]
                ),
                "fsmp_shape_normalization_coefficient_per_pe_ns": float(
                    _shape_coefficient(
                        run_norm[0, channel_index],
                        width,
                        run_renorm[0, channel_index],
                    )
                ),
                "prompt_window_low_relative_ns": float(prompt_window[0]),
                "prompt_window_high_relative_ns": float(prompt_window[1]),
                "late_window_low_relative_ns": float(late_window[0]),
                "late_window_high_relative_ns": float(late_window[1]),
                "gate_reference_peak_ns": float(
                    data["gate_reference_peak_ns"][0, channel_index]
                ),
                "prompt_window_effective_low_ns": float(
                    data["prompt_window_effective_low_ns"][0, channel_index]
                ),
                "prompt_window_effective_high_ns": float(
                    data["prompt_window_effective_high_ns"][0, channel_index]
                ),
                "late_window_effective_low_ns": float(
                    data["late_window_effective_low_ns"][0, channel_index]
                ),
                "late_window_effective_high_ns": float(
                    data["late_window_effective_high_ns"][0, channel_index]
                ),
            }
            for index, label in enumerate(labels):
                prefix = _source_slug(label, index)
                row.update(
                    {
                        f"{prefix}_events": int(
                            sim_events[index, channel_index]
                        ),
                        f"{prefix}_pe": int(sim_total[index, channel_index]),
                        f"{prefix}_time_samples": int(
                            sim_entries[index, channel_index]
                        ),
                        f"{prefix}_mean_pe_per_event": float(
                            data["simulation_mean_pe_per_event"][
                                index, channel_index
                            ]
                        ),
                        f"{prefix}_mean_pe_ratio_to_fsmp": float(
                            data["simulation_mean_pe_per_event"][
                                index, channel_index
                            ]
                            / data["run_mean_pe_per_event"][
                                0, channel_index
                            ]
                        )
                        if data["run_mean_pe_per_event"][
                            0, channel_index
                        ]
                        > 0.0
                        else np.nan,
                        f"{prefix}_prompt_pe_per_event": float(
                            data["simulation_prompt_pe_per_event"][
                                index, channel_index
                            ]
                        ),
                        f"{prefix}_late_pe_per_event": float(
                            data["simulation_late_pe_per_event"][
                                index, channel_index
                            ]
                        ),
                        f"{prefix}_prompt_late_ratio": float(
                            data["simulation_prompt_late_ratio"][
                                index, channel_index
                            ]
                        ),
                        f"{prefix}_shape_normalization_pe": int(
                            sim_norm[index, channel_index]
                        ),
                        (
                            f"{prefix}_shape_normalization_"
                            "coefficient_per_pe_ns"
                        ): float(
                            _shape_coefficient(
                                sim_norm[index, channel_index],
                                width,
                                sim_renorm[index, channel_index],
                            )
                        ),
                        f"{prefix}_l1": float(l1[index, channel_index]),
                        f"{prefix}_wasserstein_ns": float(
                            wasserstein[index, channel_index]
                        ),
                        f"{prefix}_peak_ns": float(
                            data["simulation_peak"][index, channel_index]
                        ),
                        f"{prefix}_minus_fsmp_aligned_peak_ns": float(
                            data[
                                "simulation_minus_fsmp_aligned_peak_ns"
                            ][index, channel_index]
                        ),
                        f"{prefix}_applied_source_scale": float(
                            np.asarray(
                                data["simulation_applied_source_scale"]
                            )[index]
                        ),
                    }
                )
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
                row[f"fsmp_dark_{key}"] = np.asarray(data[f"run_dark_{key}"])[
                    0, channel_index
                ]
            writer.writerow(row)
    print(f"Wrote {path}")
