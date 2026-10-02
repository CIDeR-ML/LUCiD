"""PDF plotting for laser-data comparisons."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import compare_sim as base
from .laser_compare_metrics import (
    JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
    _metric_arrays,
    _normalize_with_factor,
    _source_counts,
)

PMT_TYPE_NAME = base.PMT_TYPE_NAME
FIGURE_SIZE = base.FIGURE_SIZE
configure_plot_style = base.configure_plot_style
l1_distance = base.l1_distance
wasserstein_distance = base.wasserstein_distance
plot_metric_distribution = base.plot_metric_distribution

FSMP_COLOR = "tab:blue"
SOURCE_COLORS = {"lucid": "tab:orange", "junosw": "tab:green"}
FALLBACK_SOURCE_COLORS = ("tab:red", "tab:purple", "tab:brown", "tab:pink")
DIFFERENCE_AXIS_COLOR = "0.25"
RATIO_AXIS_COLOR = "#7f3c8d"


def amplitude_plot_arrays(
    data,
    apply_lucid_qe=True,
    lucid_qe_eff=JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
):
    """Return amplitude arrays with an optional LUCiD-only QE factor.

    Input: the aligned-data mapping.  When dark noise was added to the
    simulation densities (``simulation_dark_added``), the dark plateau
    is subtracted before the QE scaling and added back afterwards,
    because dark counts are a detector-side rate that never crosses
    the optical chain.  Output: plotting arrays for the amplitude
    pages.
    """
    if not np.isfinite(lucid_qe_eff) or not 0.0 < lucid_qe_eff <= 1.0:
        raise ValueError("lucid_qe_eff must be finite in (0, 1]")
    labels = np.asarray(data["simulation_labels"]).astype(str)
    kinds = np.asarray(
        data.get("simulation_kinds", ["simulation"] * len(labels))
    ).astype(str)
    factors = np.ones(len(labels), dtype=np.float64)
    if apply_lucid_qe:
        lowered = np.array([value.lower() for value in kinds])
        factors[lowered == "lucid"] = lucid_qe_eff
    density = np.asarray(data["simulation_pe_per_event_density"])
    mean = np.asarray(data["simulation_mean_pe_per_event"])
    prompt = np.asarray(data["simulation_prompt_pe_per_event"])
    late = np.asarray(data["simulation_late_pe_per_event"])
    dark_added = bool(
        np.asarray(data.get("simulation_dark_added", False)).item()
    )
    if dark_added:
        dark = np.asarray(
            data["run_dark_rate_pe_per_event_per_ns"], dtype=np.float64
        )
        if dark.ndim == 2:
            dark = dark[0]
        span = float(data["edges"][-1] - data["edges"][0])
        prompt_length = np.asarray(
            data["prompt_window_effective_high_ns"], dtype=np.float64
        ) - np.asarray(
            data["prompt_window_effective_low_ns"], dtype=np.float64
        )
        late_length = np.asarray(
            data["late_window_effective_high_ns"], dtype=np.float64
        ) - np.asarray(
            data["late_window_effective_low_ns"], dtype=np.float64
        )
        dark_mean = dark * span
        dark_prompt = dark * prompt_length
        dark_late = dark * late_length
        scale = factors[:, None]
        density = (density - dark[None, :, None]) * factors[:, None, None]
        density = density + dark[None, :, None]
        mean = (mean - dark_mean[None, :]) * scale + dark_mean[None, :]
        prompt = (
            (prompt - dark_prompt[None, :]) * scale
            + dark_prompt[None, :]
        )
        late = (late - dark_late[None, :]) * scale + dark_late[None, :]
    else:
        density = density * factors[:, None, None]
        mean = mean * factors[:, None]
        prompt = prompt * factors[:, None]
        late = late * factors[:, None]
    return {
        "source_factors": factors,
        "simulation_density": density,
        "simulation_mean": mean,
        "simulation_prompt": prompt,
        "simulation_late": late,
    }


def _lucid_qe_note(data, apply_lucid_qe, lucid_qe_eff):
    """Describe the active common LUCiD QE approximation."""
    kinds = np.asarray(data.get("simulation_kinds", [])).astype(str)
    lowered = {value.lower() for value in kinds}
    if apply_lucid_qe and "lucid" in lowered:
        return (
            "LUCiD-only equivalent QE at 532 nm="
            f"{lucid_qe_eff:.6g} (JUNOsw J26.5.0)"
        )
    return "LUCiD QE correction disabled"


def _shape_scale_note(data, channel_index, labels):
    """Describe c=1/mu for every unit-area curve on one channel page.

    Input: the aligned-data mapping, the channel index and source labels.
    Output: multi-line annotation text listing the coefficient converting
    each PE/event amplitude density to its unit-area shape.
    """
    run_c = np.asarray(data["run_shape_c_event_per_pe"])[0, channel_index]
    simulation_c = np.asarray(data["simulation_shape_c_event_per_pe"])[
        :, channel_index
    ]
    lines = [r"$c=1/\mu$ [event/PE]", f"FSMP: {run_c:.4g}"]
    lines.extend(
        f"{label}: {value:.4g}"
        for label, value in zip(labels, simulation_c)
    )
    return "\n".join(lines)


def _simulation_colors(data, labels):
    kinds = np.asarray(
        data.get("simulation_kinds", ["simulation"] * len(labels))
    ).astype(str)
    colors = []
    lucid_count = 0  # 记录 LUCiD 源的数量
    junosw_count = 0  # 记录 JUNOsw 源的数量
    fallback_index = 0  # 同类第二个及之后的源依次使用备用颜色
    for index, (kind, label) in enumerate(zip(kinds, labels)):
        key, text = kind.lower(), str(label).lower()
        if key == "lucid" or "lucid" in text:
            if lucid_count == 0:
                colors.append(SOURCE_COLORS["lucid"])
                lucid_count += 1
                continue
            lucid_count += 1
        elif key == "junosw" or "juno" in text:
            if junosw_count == 0:
                colors.append(SOURCE_COLORS["junosw"])
                junosw_count += 1
                continue
            junosw_count += 1
        else:
            colors.append(
                FALLBACK_SOURCE_COLORS[index % len(FALLBACK_SOURCE_COLORS)]
            )
            continue
        colors.append(
            FALLBACK_SOURCE_COLORS[
                fallback_index % len(FALLBACK_SOURCE_COLORS)
            ]
        )
        fallback_index += 1
    return colors


def plot_data(
    data,
    output,
    plot_range,
    run,
    time_difference_panels=True,
    channel_indices=None,
):
    """Write per-channel three-way density, residual and metric pages.

    Input: metrics dictionary, output path, time axis range, run number
    and the difference-panel switch; time_difference_panels=False skips
    the per-channel difference/ratio page and every L1/Wasserstein
    display (annotation and distribution pages).
    Output: none (PDF written).
    """
    configure_plot_style()
    labels, run_density, sim_density, l1, wasserstein = _metric_arrays(data)
    colors = _simulation_colors(data, labels)
    alignment = str(
        np.asarray(data.get("alignment_method", "unknown")).item()
    ).replace("_", " ")
    shift = float(np.asarray(data.get("alignment_shift_ns", 0.0)).item())
    indices = (
        range(len(data["channels"]))
        if channel_indices is None
        else channel_indices
    )
    with PdfPages(output) as pdf:
        for index in indices:
            channel = data["channels"][index]
            pmt = PMT_TYPE_NAME.get(int(data["pmt_type"][index]), "Unknown")
            processing = str(
                np.asarray(data.get("processing", "")).item()
            )
            smear = f"TTS={data['tts'][index]:.2f} ns; "
            irf_fwhm = data.get("irf_fwhm_by_channel")
            if "mode=tts_vs_irf" in processing:
                smear = "TTS vs measured per-channel IRF; "
            elif "irf=file:" in processing and irf_fwhm is not None:
                smear = (
                    f"IRF FWHM={irf_fwhm[index]:.2f} ns (TTS off); "
                )
            title = (
                f"Channel {int(channel)} | {pmt}\n"
                f"{smear}"
                f"z={data['z'][index]:.2f} mm; "
                f"theta={data['theta'][index]:.2f} deg\n"
                # f"Alignment={alignment}; FSMP shift="
                # f"{shift:.4g} ns"
            )
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                data["centers"],
                run_density[0, index],
                where="mid",
                color=FSMP_COLOR,
                linestyle="--",
                label=f"FSMP RUN {run}",
            )
            for source, (label, color) in enumerate(zip(labels, colors)):
                ax.step(
                    data["centers"],
                    sim_density[source, index],
                    where="mid",
                    color=color,
                    label=str(label),
                )
            if time_difference_panels:
                ax.text(
                    0.98,
                    0.98,
                    "\n".join(
                        f"{label}: L1={l1[i,index]:.4g}, "
                        f"W={wasserstein[i,index]:.4g} ns"
                        for i, label in enumerate(labels)
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
            ax.text(
                0.98,
                0.03,
                _shape_scale_note(data, index, labels),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize="xx-small",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.85,
                },
            )
            ax.set(
                title=title,
                xlim=plot_range,
                xlabel="Time [ns]",
                ylabel="Normalized density",
                yscale="log",
            )
            ax.set_ylim(bottom=1e-6)
            ax.legend(frameon=False, loc="upper left", fontsize="x-small")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            if not time_difference_panels:
                continue

            fig, difference = plt.subplots(figsize=FIGURE_SIZE)
            lines = []
            for source, (label, color) in enumerate(zip(labels, colors)):
                lines.append(
                    difference.step(
                        data["centers"],
                        sim_density[source, index] - run_density[0, index],
                        where="mid",
                        color=color,
                        label=f"difference: {label} - FSMP",
                    )[0]
                )
            difference.axhline(0.0, color=DIFFERENCE_AXIS_COLOR, linestyle="--")
            ratio_ax = difference.twinx()
            ratios = []
            for source, (label, color) in enumerate(zip(labels, colors)):
                ratio = np.divide(
                    sim_density[source, index],
                    run_density[0, index],
                    out=np.full_like(run_density[0, index], np.nan),
                    where=run_density[0, index] > 0,
                )
                ratios.append(
                    ratio_ax.step(
                        data["centers"],
                        ratio,
                        where="mid",
                        color=color,
                        linestyle=":",
                        label=f"ratio: {label} / FSMP",
                    )[0]
                )
            ratio_ax.axhline(1.0, color=RATIO_AXIS_COLOR, linestyle="--")
            ratio_ax.set_yscale("log")
            difference.set(
                title=title,
                xlim=plot_range,
                xlabel="Time [ns]",
                ylabel="Simulation - FSMP density",
            )
            ratio_ax.set_ylabel("Simulation / FSMP ratio")
            difference.legend(
                lines + ratios,
                [line.get_label() for line in lines + ratios],
                frameon=False,
                fontsize="x-small",
                loc="lower right",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        for name, mask in (
            ("Hamamatsu PMT", data["pmt_type"] == 1),
            ("NNVT PMT", data["pmt_type"] == 2),
            ("All PMTs", np.ones(len(data["channels"]), dtype=bool)),
        ):
            for source, label in enumerate(labels):
                if not time_difference_panels:
                    continue
                plot_metric_distribution(
                    pdf, l1[source, mask], f"{name} {label} L1", "L1 distance"
                )
                plot_metric_distribution(
                    pdf,
                    wasserstein[source, mask],
                    f"{name} {label} Wasserstein",
                    "Wasserstein distance [ns]",
                )
    print(f"Wrote {output}")


def pooled_group_shape(pe_density, mask, width):
    """Pool selected-channel PE spectra, then normalize the total shape."""
    return pooled_group_shape_and_c(pe_density, mask, width)[0]


def pooled_group_shape_and_c(pe_density, mask, width):
    """Return a pooled unit-area shape and the coefficient used for it.

    Input: per-channel PE/event density, the channel mask and the bin
    width.
    Output: (shape, c) with shape = pooled density times c, so
    c = 1/mu_pooled in event/PE.
    """
    selected = np.asarray(pe_density, dtype=np.float64)[mask]
    if not len(selected):
        return (
            np.zeros(np.asarray(pe_density).shape[-1], dtype=np.float64),
            0.0,
        )
    pooled = np.nansum(selected, axis=0)
    shape, coefficient = _normalize_with_factor(pooled, width)
    return shape, float(coefficient)


def _pooled_density_view(data):
    """Return per-source PE/event densities for the summary pages.

    Input: the aligned-data mapping.  Output: the QE-corrected density
    recovered as ``shape / renormalization`` -- the same convention as
    the time pages, so the dark plateau is not scaled by the
    LUCiD-only QE factor.  Caches without the renormalization field
    fall back to the stored (uncorrected) PE/event density.
    """
    renorm = data.get("simulation_shape_renormalization")
    if renorm is None:
        return np.asarray(data["simulation_pe_per_event_density"])
    return (
        np.asarray(data["simulation_density"])
        / np.asarray(renorm)[:, :, None]
    )


def summary_theta_groups(theta, source_z, bin_deg=10.0):
    """Return (title, mask) pairs grouping channels by theta bin.

    Input: per-channel theta values in degrees and the source z
    position; a central source (z = 0) is not split.  Output: one
    group per populated ``bin_deg``-wide bin, titled
    ``theta in [LOW, HIGH) deg``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if np.isclose(source_z, 0.0):
        return [("", np.ones(len(theta), dtype=bool))]
    bin_low = np.floor(theta / bin_deg) * bin_deg
    return [
        (
            f"theta in [{value:.0f}, {value + bin_deg:.0f}) deg",
            np.isclose(bin_low, value),
        )
        for value in np.unique(bin_low[np.isfinite(theta)])
    ]


def plot_summary_data(data, output, plot_range, run, source_z):
    """Write PMT-type/theta summary pages for all simulation sources."""
    configure_plot_style()
    labels, _, _, _, _ = _metric_arrays(data)
    _, _, _, sim_entries, _, _ = _source_counts(data, labels)
    run_pe_density = np.asarray(data["run_pe_per_event_density"])
    sim_pe_density = _pooled_density_view(data)
    colors = _simulation_colors(data, labels)
    width = float(data["edges"][1] - data["edges"][0])
    alignment = str(
        np.asarray(data.get("alignment_method", "unknown")).item()
    ).replace("_", " ")
    theta_groups = summary_theta_groups(data["theta"], source_z)
    with PdfPages(output) as pdf:
        for theta_title, theta_mask in theta_groups:
            for pmt_title, pmt_id in (
                ("Hamamatsu PMT pooled timing shape", 1),
                ("NNVT PMT pooled timing shape", 2),
            ):
                mask = theta_mask & (data["pmt_type"] == pmt_id)
                if not np.any(mask):
                    continue
                pooled_run, pooled_run_c = pooled_group_shape_and_c(
                    run_pe_density[0], mask, width
                )
                pooled_sources = []
                pooled_source_c = []
                for index in range(len(labels)):
                    if np.any(mask & (sim_entries[index] > 0)):
                        pooled, coefficient = pooled_group_shape_and_c(
                            sim_pe_density[index], mask, width
                        )
                    else:
                        pooled = np.zeros(len(data["centers"]))
                        coefficient = 0.0
                    pooled_sources.append(pooled)
                    pooled_source_c.append(coefficient)
                title = (
                    f"{theta_title + ', ' if theta_title else ''}{pmt_title}"
                )
                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                ax.step(
                    data["centers"],
                    pooled_run,
                    where="mid",
                    color=FSMP_COLOR,
                    linestyle="--",
                    label=f"FSMP RUN {run}",
                )
                for label, pooled, color in zip(
                    labels, pooled_sources, colors
                ):
                    ax.step(
                        data["centers"],
                        pooled,
                        where="mid",
                        color=color,
                        label=str(label),
                    )
                ax.text(
                    0.98,
                    0.98,
                    "\n".join(
                        f"{label}: "
                        f"L1={l1_distance(pooled_run, pooled, width):.4g}, "
                        "W={:.4g} ns".format(
                            wasserstein_distance(
                                pooled_run, pooled, width
                            )
                        )
                        for label, pooled in zip(labels, pooled_sources)
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
                scale_lines = [
                    r"$c=1/\mu_{\mathrm{pooled}}$ [event/PE]",
                    f"N PMTs={np.count_nonzero(mask)}",
                    f"FSMP: {pooled_run_c:.4g}",
                ]
                scale_lines.extend(
                    f"{label}: {coefficient:.4g}"
                    for label, coefficient in zip(labels, pooled_source_c)
                )
                ax.text(
                    0.98,
                    0.03,
                    "\n".join(scale_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize="xx-small",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.8",
                        "alpha": 0.85,
                    },
                )
                short_labels = ", ".join(
                    short_source_label(label) for label in labels
                )
                ax.set(
                    title=(
                        f"{title}\nFSMP RUN {run} vs {short_labels}; "
                        # f"alignment={alignment}"
                    ),
                    xlim=plot_range,
                    xlabel="Time [ns]",
                    ylabel="Normalized density",
                    yscale="log",
                )
                ax.set_ylim(bottom=1e-6)
                ax.legend(
                    frameon=False, loc="upper left", fontsize="x-small"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                fig, difference = plt.subplots(figsize=FIGURE_SIZE)
                lines = []
                ratios = []
                for label, pooled, color in zip(
                    labels, pooled_sources, colors
                ):
                    lines.append(
                        difference.step(
                            data["centers"],
                            pooled - pooled_run,
                            where="mid",
                            color=color,
                            label=f"difference: {label} - FSMP",
                        )[0]
                    )
                difference.axhline(
                    0.0, color=DIFFERENCE_AXIS_COLOR, linestyle="--"
                )
                ratio_ax = difference.twinx()
                for label, pooled, color in zip(
                    labels, pooled_sources, colors
                ):
                    ratio = np.divide(
                        pooled,
                        pooled_run,
                        out=np.full_like(pooled_run, np.nan),
                        where=pooled_run > 0,
                    )
                    ratios.append(
                        ratio_ax.step(
                            data["centers"],
                            ratio,
                            where="mid",
                            color=color,
                            linestyle=":",
                            label=f"ratio: {label} / FSMP",
                        )[0]
                    )
                ratio_ax.axhline(1.0, color=RATIO_AXIS_COLOR, linestyle="--")
                ratio_ax.set_yscale("log")
                difference.set(
                    title=title,
                    xlim=plot_range,
                    xlabel="Time [ns]",
                    ylabel="Simulation - FSMP density",
                )
                ratio_ax.set_ylabel("Simulation / FSMP ratio")
                difference.legend(
                    lines + ratios,
                    [line.get_label() for line in lines + ratios],
                    frameon=False,
                    fontsize="x-small",
                    loc="lower right",
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Wrote {output}")


def short_source_label(label):
    """Compact a source label for dense title and annotation text.

    Input: the full source label (e.g. "LUCiD (chimney off)" or
    "JUNOsw (J26.5.0)").  Output: the compact form used in titles;
    legends keep the full label.  The JUNOsw version tag is dropped
    and the space before a variant parenthesis is removed.
    """
    text = str(label)
    if text.startswith("JUNOsw"):
        return "JUNOsw"
    return text.replace(" (", "(")


def _amplitude_channel_title(channel, pmt, dark_rate, labels, scales,
                             qe_note):
    """Return the per-channel amplitude-page title.

    Input: channel copy number, PMT type name, this channel's dark rate
    added to MC in PE/event/ns, simulation labels, the applied source
    intensity scales and the LUCiD QE note.  Output: the title string
    carrying the dark rate and the per-source intensity scales with
    shortened labels; the scale list wraps to a second aligned line
    when a single line would exceed the page width.
    """
    pairs = [
        f"{short_source_label(label)}={scale:.6g}"
        for label, scale in zip(labels, scales)
    ]
    prefix = "Source intensity scale: "
    scale_line = prefix + ", ".join(pairs)
    if len(scale_line) > 68 and len(pairs) > 1:
        mid = (len(pairs) + 1) // 2
        scale_line = (
            prefix + ", ".join(pairs[:mid]) + ",\n"
            + " " * len(prefix) + ", ".join(pairs[mid:])
        )
    return (
        f"Channel {int(channel)} | {pmt} | PE/event timing spectrum\n"
        f"Dark added to MC={dark_rate:.5g} PE/event/ns\n"
        f"{scale_line}\n"
        f"{qe_note}"
    )


def plot_amplitude_data(
    data,
    output,
    plot_range,
    run,
    apply_lucid_qe=True,
    lucid_qe_eff=JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
    channel_indices=None,
):
    """Write per-channel PE/event spectra and channel-rate summaries."""
    configure_plot_style()
    labels = np.asarray(data["simulation_labels"]).astype(str)
    colors = _simulation_colors(data, labels)
    run_density = np.asarray(data["run_pe_per_event_density"])
    run_mean = np.asarray(data["run_mean_pe_per_event"])
    amplitude = amplitude_plot_arrays(
        data,
        apply_lucid_qe=apply_lucid_qe,
        lucid_qe_eff=lucid_qe_eff,
    )
    sim_density = amplitude["simulation_density"]
    sim_mean = amplitude["simulation_mean"]
    qe_note = _lucid_qe_note(data, apply_lucid_qe, lucid_qe_eff)
    prompt_low = np.asarray(data["prompt_window_effective_low_ns"])
    prompt_high = np.asarray(data["prompt_window_effective_high_ns"])
    late_low = np.asarray(data["late_window_effective_low_ns"])
    late_high = np.asarray(data["late_window_effective_high_ns"])
    indices = (
        range(len(data["channels"]))
        if channel_indices is None
        else channel_indices
    )
    with PdfPages(output) as pdf:
        for index in indices:
            channel = data["channels"][index]
            pmt = PMT_TYPE_NAME.get(int(data["pmt_type"][index]), "Unknown")
            dark_rate = data["run_dark_rate_pe_per_event_per_ns"][
                0, index
            ]
            title = _amplitude_channel_title(
                channel,
                pmt,
                dark_rate,
                labels,
                data["simulation_applied_source_scale"],
                qe_note,
            )
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                data["centers"],
                run_density[0, index],
                where="mid",
                color=FSMP_COLOR,
                linestyle="--",
                label=(
                    f"FSMP RUN {run}: mean={run_mean[0,index]:.6g}"
                    " PE/event"
                ),
            )
            for source, (label, color) in enumerate(zip(labels, colors)):
                ax.step(
                    data["centers"],
                    sim_density[source, index],
                    where="mid",
                    color=color,
                    label=(
                        f"{label}: mean={sim_mean[source,index]:.6g}"
                        " PE/event"
                    ),
                )
            boundaries = np.unique(
                [
                    prompt_low[0, index],
                    prompt_high[0, index],
                    late_low[0, index],
                    late_high[0, index],
                ]
            )
            for boundary in boundaries:
                if np.isfinite(boundary):
                    ax.axvline(
                        boundary,
                        color="0.6",
                        linestyle=":",
                        linewidth=0.8,
                    )
            ax.set(
                title=title,
                xlim=plot_range,
                xlabel="Time [ns]",
                ylabel="PE / event / ns",
                yscale="log",
            )
            ax.set_ylim(bottom=1e-8)
            ax.legend(frameon=False, loc="upper left", fontsize="x-small")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        channels = np.asarray(data["channels"])
        fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
        axes[0].plot(
            channels,
            run_mean[0],
            ".",
            color=FSMP_COLOR,
            label=f"FSMP RUN {run}",
        )
        for source, (label, color) in enumerate(zip(labels, colors)):
            axes[0].plot(
                channels,
                sim_mean[source],
                ".",
                color=color,
                label=str(label),
            )
            ratio = np.divide(
                sim_mean[source],
                run_mean[0],
                out=np.full_like(sim_mean[source], np.nan),
                where=run_mean[0] > 0.0,
            )
            finite_ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
            median_ratio = (
                float(np.median(finite_ratio))
                if finite_ratio.size
                else np.nan
            )
            axes[1].plot(
                channels,
                ratio,
                ".",
                color=color,
                label=(
                    f"{label} / FSMP "
                    f"(median={median_ratio:.6g})"
                ),
            )
        axes[0].set(
            title=f"Mean PE per event by channel\n{qe_note}",
            ylabel="Mean PE / event",
            yscale="log",
        )
        axes[1].axhline(1.0, color=DIFFERENCE_AXIS_COLOR, linestyle="--")
        axes[1].set(
            xlabel="Channel",
            ylabel="Simulation / FSMP",
            yscale="log",
        )
        for axis in axes:
            axis.legend(frameon=False, fontsize="x-small")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Wrote {output}")
