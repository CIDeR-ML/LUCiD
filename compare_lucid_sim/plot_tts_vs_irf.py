#!/usr/bin/env python
"""TTS-versus-IRF comparison check on top of the three-way pipeline.

Every simulation source is built twice -- once with the per-channel TTS
convolution and once with the measured per-channel IRF kernels (which
already contain the TTS) -- and the resulting variant curves are drawn
together with the FSMP data, with TTS/IRF suffixes in the legend.  The
merged metrics are cached under a ``*_tts_vs_irf`` filename suffix so
re-runs only re-plot.

This is a cross-check tool: the production comparison
(compare_sim_vs_real.py) convolves each source with either TTS or IRF,
never both.  Reuses the shared pipeline stages (prepare_inputs,
expected_source_labels, emit_results) from compare_sim_vs_real.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_lucid_sim import compare_sim_vs_real as pipeline
from compare_lucid_sim.laser_compare_metrics import (
    build_channel_histograms_multi,
    merge_smearing_variants,
    processing_string,
)
from compare_lucid_sim.laser_compare_irf import characterized_channels


def parse_args(argv=None):
    """Parse the check CLI: the pipeline parser plus the panel switch.

    Input: optional argv list.  Output: the finalized namespace; the
    check requires --irf-h5 (the measured per-channel IRF kernels).
    """
    parser = pipeline.build_parser()
    parser.prog = "plot_tts_vs_irf.py"
    parser.description = __doc__
    parser.add_argument(
        "--time-difference-panels",
        dest="time_difference_panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw per-channel difference/ratio pages and the "
        "L1/Wasserstein displays; "
        "disable with --no-time-difference-panels",
    )
    args = pipeline.parse_args(
        ensure_channel_selection(argv), parser=parser
    )
    if args.irf_h5 is None:
        parser.error("the TTS-vs-IRF check requires --irf-h5")
    return args


def ensure_channel_selection(argv):
    """Inject a derived --channel-selection into argv when absent.

    Input: optional argv list.  Output: the argv list; when no explicit
    selection is given, the characterized channels (IRF h5 serials
    mapped to CD copy numbers via the PMT map) are written to
    ``<output_root>/irf125.csv`` and appended as --channel-selection,
    keeping the ``irf125`` cache filename suffix.  Runs before the full
    parse because the cache suffix is finalized during parsing.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--channel-selection", type=Path, default=None)
    pre.add_argument("--irf-h5", type=Path, default=None)
    pre.add_argument(
        "--pmt-map", type=Path, default=Path(pipeline.DEFAULT_PMT_MAP)
    )
    pre.add_argument(
        "--output-root", type=Path, default=Path("compare_sim")
    )
    known, _ = pre.parse_known_args(argv)
    if known.channel_selection is not None or known.irf_h5 is None:
        return argv
    channels = characterized_channels(known.irf_h5, known.pmt_map)
    if not channels:
        raise ValueError(
            f"no characterized CD channels found in {known.irf_h5}"
        )
    known.output_root.mkdir(parents=True, exist_ok=True)
    selection = known.output_root / "irf125.csv"
    with selection.open("w") as stream:
        stream.write("channel\n")
        for channel in channels:
            stream.write(f"{channel}\n")
    print(
        f"Selected {len(channels)} characterized channels "
        f"-> {selection}"
    )
    base = list(sys.argv[1:] if argv is None else argv)
    return base + ["--channel-selection", str(selection)]


def main(argv=None):
    """Build both smearing variants of every source and render them.

    Input: optional argv list.  Output: none (figures and CSVs written).
    """
    args = parse_args(argv)
    prepared = pipeline.prepare_inputs(args)
    processing = processing_string(
        True,
        prepared.irf_description,
        args.add_dark_to_simulation,
        target_pe=getattr(
            args,
            "fsmp_target_pe",
            prepared.fsmp_data.get("fsmp_target_pe"),
        ),
        prompt_window=args.prompt_window,
        late_window=args.late_window,
        alignment=args.alignment,
        source_intensity_scale=prepared.source_scale,
        lucid_qe=(
            args.lucid_qe_effective if args.apply_lucid_qe else None
        ),
    )
    processing += ",mode=tts_vs_irf"
    aligned_path = args.aligned_data.with_name(
        args.aligned_data.stem + "_tts_vs_irf.npz"
    )
    data = None
    if aligned_path.exists() and not args.force:
        candidate = pipeline.load_data(aligned_path)
        labels, kinds = pipeline.expected_source_labels(
            prepared.sources, smearing_variants=True
        )
        if (
            str(candidate.get("processing", "")) == processing
            and np.array_equal(
                candidate.get("channels"), prepared.fsmp_data["channels"]
            )
            and np.array_equal(
                candidate.get("edges"), prepared.sources[0]["edges"]
            )
            and np.array_equal(candidate.get("simulation_labels"), labels)
            and np.array_equal(candidate.get("simulation_kinds"), kinds)
        ):
            data = candidate
    if data is None:
        common = {
            "add_dark_to_simulation": args.add_dark_to_simulation,
            "dark_window": args.dark_window,
            "prompt_window": args.prompt_window,
            "late_window": args.late_window,
            "alignment": args.alignment,
            "simulation_source_scale": prepared.source_scale,
            "lucid_qe_factors": prepared.qe_factors,
        }
        tts_data = build_channel_histograms_multi(
            prepared.fsmp_data, prepared.sources, use_tts=True, **common
        )
        irf_data = build_channel_histograms_multi(
            prepared.fsmp_data,
            prepared.sources,
            use_tts=False,
            irf_kernels=prepared.irf_kernels,
            irf_fwhm_by_channel=prepared.irf_fwhm_by_channel,
            irf_description=prepared.irf_description,
            **common,
        )
        data = merge_smearing_variants(tts_data, irf_data, processing)
        np.savez_compressed(aligned_path, **data)
        print(f"Wrote {aligned_path}")
    pipeline.emit_results(
        args, data, time_difference_panels=args.time_difference_panels
    )


if __name__ == "__main__":
    main()