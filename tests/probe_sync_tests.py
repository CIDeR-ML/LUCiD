"""Probe restored-pipeline values for syncing the tracked test file."""
import csv
import importlib.util

import numpy as np

from compare import compare_sim_vs_real as comparison
from compare import laser_compare_plotting as plotting

spec = importlib.util.spec_from_file_location(
    "t", "tests/test_compare_sim_vs_real_metrics.py"
)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

fsmp, sim = t._synthetic_inputs()
result = comparison.build_channel_histograms_multi(
    fsmp, [sim], alignment="l1", use_tts=False,
    prompt_window=(-1.0, 1.0), late_window=(1.0, 4.0),
)
print("== shape keys:", [k for k in result if "shape" in k])
print("== run_shape_c:", result["run_shape_c_event_per_pe"][0, 0],
      "| sim_shape_c:", result["simulation_shape_c_event_per_pe"][0, 0])
comparison.write_comparison_stats_csv("/tmp/probe.csv", result, 15040, 0.0)
with open("/tmp/probe.csv", newline="") as f:
    row = next(csv.DictReader(f))
for k, v in row.items():
    print(f"CSV {k} = {v}")


class CapturePdf:
    figures = []

    def __init__(self, _):
        CapturePdf.figures = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def savefig(self, figure):
        self.figures.append(figure)


plotting.PdfPages = CapturePdf
plotting.plt.switch_backend("Agg")
comparison.plot_amplitude_data(
    result, "/tmp/probe_amp.pdf", (0.0, 4.0), 15040
)
axis = CapturePdf.figures[0].axes[0]
print("== TITLE:", repr(axis.get_title()))
print("== LEGEND:", [x.get_text() for x in axis.get_legend().get_texts()])
print("== NLINES:", len(axis.lines))
print("== windows:",
      result["prompt_window_effective_low_ns"][0, 0],
      result["prompt_window_effective_high_ns"][0, 0],
      result["late_window_effective_low_ns"][0, 0],
      result["late_window_effective_high_ns"][0, 0])

comparison.resolve_run = lambda *_: 15040
args = comparison.parse_args(["--sim-dir", "simulation"])
print("== CLI dark_window:", args.dark_window)
print("== CLI amplitude_opt:", args.amplitude_opt.name)
print("== CLI apply_lucid_qe:", args.apply_lucid_qe,
      "| qe:", args.lucid_qe_effective)
print("== CLI has channel_mode:", hasattr(args, "channel_mode"))
print("== CLI scale:", args.source_intensity_scale)
