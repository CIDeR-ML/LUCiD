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
    """Return amplitude arrays with an optional LUCiD-only QE factor."""
    if not np.isfinite(lucid_qe_eff) or not 0.0 < lucid_qe_eff <= 1.0:
        raise ValueError("lucid_qe_eff must be finite and in (0, 1]")
    labels = np.asarray(data["simulation_labels"]).astype(str)
    kinds = np.asarray(
        data.get("simulation_kinds", ["simulation"] * len(labels))
    ).astype(str)
    factors = np.ones(len(labels), dtype=np.float64)
    if apply_lucid_qe:
        factors[np.char.lower(kinds) == "lucid"] = lucid_qe_eff
    return {
        "source_factors": factors,
        "simulation_density": np.asarray(
            data["simulation_pe_per_event_density"]
        )
        * factors[:, None, None],
        "simulation_mean": np.asarray(
            data["simulation_mean_pe_per_event"]
        )
        * factors[:, None],
        "simulation_prompt": np.asarray(
            data["simulation_prompt_pe_per_event"]
        )
        * factors[:, None],
        "simulation_late": np.asarray(
            data["simulation_late_pe_per_event"]
        )
        * factors[:, None],
    }


def _lucid_qe_note(data, apply_lucid_qe, lucid_qe_eff):
    """Describe the active common LUCiD QE approximation."""
    kinds = np.asarray(data.get("simulation_kinds", [])).astype(str)
    if apply_lucid_qe and np.any(np.char.lower(kinds) == "lucid"):
        return (
            "LUCiD-only equivalent QE at 532 nm="
            f"{lucid_qe_eff:.6g} (JUNOsw J26.3.2)"
        )
    return "LUCiD QE correction disabled"


def _simulation_colors(data, labels):
    kinds = np.asarray(
        data.get("simulation_kinds", ["simulation"] * len(labels))
    ).astype(str)
    colors = []
    lucid_count = 0  # 记录 LUCiD 源的数量
    for index, (kind, label) in enumerate(zip(kinds, labels)):
        key, text = kind.lower(), str(label).lower()
        if key == "lucid" or "lucid" in text:
            if lucid_count == 0:
                colors.append(SOURCE_COLORS["lucid"])
            else:
                # 第2个及之后的LUCiD使用备用颜色
                colors.append(
                    FALLBACK_SOURCE_COLORS[
                        (lucid_count - 1) % len(FALLBACK_SOURCE_COLORS)
                    ]
                )
            lucid_count += 1
        elif key == "junosw" or "juno" in text:
            colors.append(SOURCE_COLORS["junosw"])
        else:
            colors.append(
                FALLBACK_SOURCE_COLORS[index % len(FALLBACK_SOURCE_COLORS)]
            )
    return colors


def plot_data(data, output, plot_range, run):
    """Write per-channel three-way density, residual and metric pages."""
    configure_plot_style()
    labels, run_density, sim_density, l1, wasserstein = _metric_arrays(data)
    colors = _simulation_colors(data, labels)
    alignment = str(
        np.asarray(data.get("alignment_method", "unknown")).item()
    ).replace("_", " ")
    shift = float(np.asarray(data.get("alignment_shift_ns", 0.0)).item())
    with PdfPages(output) as pdf:
        for index, channel in enumerate(data["channels"]):
            pmt = PMT_TYPE_NAME.get(int(data["pmt_type"][index]), "Unknown")
            title = (
                f"Channel {int(channel)} | {pmt}\n"
                f"TTS={data['tts'][index]:.2f} ns; "
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
    selected = np.asarray(pe_density, dtype=np.float64)[mask]
    if not len(selected):
        return np.zeros(np.asarray(pe_density).shape[-1], dtype=np.float64)
    pooled = np.nansum(selected, axis=0)
    return _normalize_with_factor(pooled, width)[0]


def plot_summary_data(data, output, plot_range, run, source_z):
    """Write PMT-type/theta summary pages for all simulation sources."""
    configure_plot_style()
    labels, _, _, _, _ = _metric_arrays(data)
    _, _, _, sim_entries, _, _ = _source_counts(data, labels)
    run_pe_density = np.asarray(data["run_pe_per_event_density"])
    sim_pe_density = np.asarray(data["simulation_pe_per_event_density"])
    colors = _simulation_colors(data, labels)
    width = float(data["edges"][1] - data["edges"][0])
    alignment = str(
        np.asarray(data.get("alignment_method", "unknown")).item()
    ).replace("_", " ")
    if np.isclose(source_z, 0.0):
        theta_groups = [("", np.ones(len(data["theta"]), dtype=bool))]
    else:
        theta_groups = [
            (f"theta={value:.2f} deg", np.isclose(data["theta"], value))
            for value in np.unique(data["theta"][np.isfinite(data["theta"])])
        ]
    with PdfPages(output) as pdf:
        for theta_title, theta_mask in theta_groups:
            for pmt_title, pmt_id in (
                ("Hamamatsu PMT pooled timing shape", 1),
                ("NNVT PMT pooled timing shape", 2),
            ):
                mask = theta_mask & (data["pmt_type"] == pmt_id)
                if not np.any(mask):
                    continue
                pooled_run = pooled_group_shape(
                    run_pe_density[0], mask, width
                )
                pooled_sources = [
                    (
                        pooled_group_shape(sim_pe_density[i], mask, width)
                        if np.any(mask & (sim_entries[i] > 0))
                        else np.zeros(len(data["centers"]))
                    )
                    for i in range(len(labels))
                ]
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
                ax.set(
                    title=(
                        f"{title}\nFSMP RUN {run} vs {', '.join(labels)}; "
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


def plot_amplitude_data(
    data,
    output,
    plot_range,
    run,
    apply_lucid_qe=True,
    lucid_qe_eff=JUNOSW_J26_3_2_EQUIVALENT_QE_532_NM,
):
    """Write per-channel PE/event spectra and channel-rate summaries."""
    configure_plot_style()
    labels = np.asarray(data["simulation_labels"]).astype(str)
    colors = _simulation_colors(data, labels)
    run_density = np.asarray(data["run_pe_per_event_density"])
    run_mean = np.asarray(data["run_mean_pe_per_event"])
    run_prompt = np.asarray(data["run_prompt_pe_per_event"])
    run_late = np.asarray(data["run_late_pe_per_event"])
    amplitude = amplitude_plot_arrays(
        data,
        apply_lucid_qe=apply_lucid_qe,
        lucid_qe_eff=lucid_qe_eff,
    )
    sim_density = amplitude["simulation_density"]
    sim_mean = amplitude["simulation_mean"]
    sim_prompt = amplitude["simulation_prompt"]
    sim_late = amplitude["simulation_late"]
    qe_note = _lucid_qe_note(data, apply_lucid_qe, lucid_qe_eff)
    prompt_low = np.asarray(data["prompt_window_effective_low_ns"])
    prompt_high = np.asarray(data["prompt_window_effective_high_ns"])
    late_low = np.asarray(data["late_window_effective_low_ns"])
    late_high = np.asarray(data["late_window_effective_high_ns"])
    with PdfPages(output) as pdf:
        for index, channel in enumerate(data["channels"]):
            pmt = PMT_TYPE_NAME.get(int(data["pmt_type"][index]), "Unknown")
            title = (
                f"Channel {int(channel)} | {pmt} | PE/event timing spectrum\n"
                "Effective gates: "
                f"prompt=[{prompt_low[0,index]:.4g}, "
                f"{prompt_high[0,index]:.4g}) ns; "
                f"late=[{late_low[0,index]:.4g}, "
                f"{late_high[0,index]:.4g}) ns\n"
                f"{qe_note}"
            )
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.step(
                data["centers"],
                run_density[0, index],
                where="mid",
                color=FSMP_COLOR,
                linestyle="--",
                label=(
                    f"FSMP RUN {run}: mean={run_mean[0,index]:.6g}, "
                    f"prompt={run_prompt[0,index]:.6g}, "
                    f"late={run_late[0,index]:.6g} PE/event"
                ),
            )
            for source, (label, color) in enumerate(zip(labels, colors)):
                ax.step(
                    data["centers"],
                    sim_density[source, index],
                    where="mid",
                    color=color,
                    label=(
                        f"{label}: mean={sim_mean[source,index]:.6g}, "
                        f"prompt={sim_prompt[source,index]:.6g}, "
                        f"late={sim_late[source,index]:.6g} PE/event"
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
