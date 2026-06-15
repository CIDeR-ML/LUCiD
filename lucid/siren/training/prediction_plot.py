"""
In-training prediction-vs-truth plot callback for SIREN.

Drops PDFs at `<output_dir>/prediction_plots/step_NNNNNN.pdf` every N training
steps, each one comparing the trained-so-far model against the source lookup
table at a small set of anchor points. Two views per figure:

  * View A — intensity vs angle (or dE/dx for the dedx variant) at four fixed
    `s/s_max` slices (defaults 0.1, 0.25, 0.5, 0.75). One panel per slice; one
    curve per representative energy.
  * View B — intensity vs `s/s_max` summed over the 2nd axis (angle or dE/dx).

Truth curves are dashed, model predictions solid, matched colours per energy.

Works for both `photon_lookup_table.h5` and `dedx_lookup_table.h5` (auto-
detects via the h5 keys, mirroring `dataset.py`'s detection).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List

import h5py
import jax.numpy as jnp
import numpy as np

# Render PDFs via an explicit PDF canvas instead of pyplot, so the global
# matplotlib backend (e.g. Jupyter's inline backend driving the live loss/LR
# monitor) is left alone.
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.cm import viridis
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Energy axis label for the marginal panel
_AXIS_LABELS = {
    "photon": ("Opening angle [rad]", r"$s / s_{\max}$"),
    "dedx":   (r"dE/dx [keV/mm]",     r"$s / s_{\max}$"),
}
_INTENSITY_LABELS = {
    "photon": "photons / event / bin",
    "dedx":   "entries / event / bin",
}


class PredictionComparisonCallback:
    """Trainer callback that emits one PNG per refresh interval."""

    def __init__(
        self,
        *,
        h5_path: Path,
        dataset,                      # PhotonSimDataset
        output_dir: Path,
        energies_mev: Iterable[int],
        distance_slices: Iterable[float],
        xaxis_slices: Iterable[float],
        every: int,
        resume: bool = False,
    ):
        self.h5_path = Path(h5_path)
        self.dataset = dataset
        self.every = int(every)
        self.energies_req = [int(e) for e in energies_mev]
        self.distance_slices_req = [float(s) for s in distance_slices]
        # xaxis_slices is in the variant's native units — radians for the
        # photon (angle) variant, keV/mm for the dedx variant.
        self.xaxis_slices_req = [float(x) for x in xaxis_slices]

        self.plot_dir = Path(output_dir) / "prediction_plots"
        if not resume and self.plot_dir.is_dir():
            shutil.rmtree(self.plot_dir, ignore_errors=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.table_type = getattr(dataset, "table_type", None)
        if self.table_type not in ("photon", "dedx"):
            raise ValueError(
                f"dataset.table_type must be 'photon' or 'dedx' "
                f"(got {self.table_type!r})"
            )

        # Used to draw a horizontal "zero-suppression floor" on each panel —
        # below this y-value, the trainer can't distinguish samples from zero
        # (everything ≤ zero_threshold maps to log10(zero_threshold) after
        # the offset+log transform). Fallback for older datasets without it.
        self.zero_threshold = float(getattr(dataset, "zero_threshold", 1e-2))

        # Sanity-assert the h5 file matches the dataset table_type. Guards
        # against `--h5-path foo_photon.h5 --data-type dedx` foot-guns.
        with h5py.File(self.h5_path, "r") as f:
            wanted_key = (
                "data/dedx_table_average" if self.table_type == "dedx"
                else "data/photon_table_average"
            )
            if wanted_key not in f:
                raise ValueError(
                    f"{self.h5_path} does not contain {wanted_key} — does it "
                    f"match --data-type {self.table_type!r}?"
                )

        self._load_truth_and_grids()

        # Cache normalisation bounds for cheap np ops in __call__.
        self.input_min = np.asarray(dataset.normalized_bounds["input_min"])
        self.input_max = np.asarray(dataset.normalized_bounds["input_max"])

        # Warm-up JIT compile so the first callback firing doesn't stall.
        self._warmup_done = False

    # ---- one-time setup ----------------------------------------------------

    def _load_truth_and_grids(self) -> None:
        """Open the .h5, locate nearest grid points, cache truth + model inputs."""
        with h5py.File(self.h5_path, "r") as f:
            energy_values = np.asarray(f["coordinates/energy_values"][:])
            distance_centers = np.asarray(f["coordinates/distance_centers"][:])
            if self.table_type == "photon":
                x_centers = np.asarray(f["coordinates/angle_centers"][:])
                avg_key = "data/photon_table_average"
            else:
                x_centers = np.asarray(f["coordinates/dedx_centers"][:])
                avg_key = "data/dedx_table_average"
            avg_table = np.asarray(f[avg_key][:])  # (n_E, n_x, n_d)

        # Nearest-neighbor energy lookup. Warn if any choice is >5% off.
        e_idx = []
        for req in self.energies_req:
            j = int(np.argmin(np.abs(energy_values - req)))
            actual = int(energy_values[j])
            if abs(actual - req) > 0.05 * max(abs(req), 1):
                logger.warning(
                    "prediction_plot: requested E=%d MeV → grid neighbour "
                    "%d MeV (%.1f%% off)",
                    req, actual, 100 * (actual - req) / max(abs(req), 1),
                )
            e_idx.append(j)
        self.energy_indices = np.asarray(e_idx, dtype=int)
        self.energy_values = energy_values
        self.energies_grid = energy_values[self.energy_indices]    # (n_E_req,)

        # Nearest-neighbor s/s_max slices.
        d_idx = [int(np.argmin(np.abs(distance_centers - s)))
                 for s in self.distance_slices_req]
        self.distance_slice_indices = np.asarray(d_idx, dtype=int)
        self.distance_slices_grid = distance_centers[self.distance_slice_indices]

        # Nearest-neighbor x-axis slices (angle in rad for photon,
        # dE/dx in keV/mm for dedx).
        x_idx = [int(np.argmin(np.abs(x_centers - xv)))
                 for xv in self.xaxis_slices_req]
        self.xaxis_slice_indices = np.asarray(x_idx, dtype=int)
        self.xaxis_slices_grid = x_centers[self.xaxis_slice_indices]

        self.x_centers = x_centers              # (n_x,)
        self.distance_centers = distance_centers  # (n_d,)
        n_E = len(self.energy_indices)
        n_x = len(x_centers)
        n_d = len(distance_centers)
        n_slices = len(self.distance_slice_indices)
        n_xslices = len(self.xaxis_slice_indices)

        # Truth caches.
        # View A: shape (n_E, n_slices, n_x). truth_A[k, m, j] = avg[i_E, j, i_d_m].
        self.truth_A = np.zeros((n_E, n_slices, n_x), dtype=np.float64)
        for k, iE in enumerate(self.energy_indices):
            for m, iD in enumerate(self.distance_slice_indices):
                self.truth_A[k, m, :] = avg_table[iE, :, iD]
        # View B: shape (n_E, n_d). truth_B[k, l] = sum_j avg[i_E, j, l].
        self.truth_B = np.zeros((n_E, n_d), dtype=np.float64)
        for k, iE in enumerate(self.energy_indices):
            self.truth_B[k, :] = avg_table[iE, :, :].sum(axis=0)
        # View C: shape (n_E, n_xslices, n_d). truth_C[k, p, l] = avg[i_E, i_x_p, l].
        self.truth_C = np.zeros((n_E, n_xslices, n_d), dtype=np.float64)
        for k, iE in enumerate(self.energy_indices):
            for p, iX in enumerate(self.xaxis_slice_indices):
                self.truth_C[k, p, :] = avg_table[iE, iX, :]

        # Pre-build the *unnormalised* model-input grids.
        # View A inputs: (n_E, n_slices, n_x, 3) → flatten to (N_A, 3).
        Ev_A = np.broadcast_to(self.energies_grid[:, None, None],
                               (n_E, n_slices, n_x))
        Xc_A = np.broadcast_to(x_centers[None, None, :],
                               (n_E, n_slices, n_x))
        Dc_A = np.broadcast_to(self.distance_slices_grid[None, :, None],
                               (n_E, n_slices, n_x))
        self.inputs_A_phys = np.stack(
            [Ev_A.ravel(), Xc_A.ravel(), Dc_A.ravel()], axis=-1,
        ).astype(np.float32)
        self.inputs_A_shape = (n_E, n_slices, n_x)

        # View B inputs: (n_E, n_x, n_d, 3) for summing later → flatten too.
        Ev_B = np.broadcast_to(self.energies_grid[:, None, None],
                               (n_E, n_x, n_d))
        Xc_B = np.broadcast_to(x_centers[None, :, None],
                               (n_E, n_x, n_d))
        Dc_B = np.broadcast_to(distance_centers[None, None, :],
                               (n_E, n_x, n_d))
        self.inputs_B_phys = np.stack(
            [Ev_B.ravel(), Xc_B.ravel(), Dc_B.ravel()], axis=-1,
        ).astype(np.float32)
        self.inputs_B_shape = (n_E, n_x, n_d)

        # View C inputs: (n_E, n_xslices, n_d, 3) → flatten.
        Ev_C = np.broadcast_to(self.energies_grid[:, None, None],
                               (n_E, n_xslices, n_d))
        Xc_C = np.broadcast_to(self.xaxis_slices_grid[None, :, None],
                               (n_E, n_xslices, n_d))
        Dc_C = np.broadcast_to(distance_centers[None, None, :],
                               (n_E, n_xslices, n_d))
        self.inputs_C_phys = np.stack(
            [Ev_C.ravel(), Xc_C.ravel(), Dc_C.ravel()], axis=-1,
        ).astype(np.float32)
        self.inputs_C_shape = (n_E, n_xslices, n_d)

    # ---- inference helpers -------------------------------------------------

    def _normalize_inputs(self, x_phys: np.ndarray) -> np.ndarray:
        x01 = (x_phys - self.input_min) / (self.input_max - self.input_min)
        return (2.0 * x01 - 1.0).astype(np.float32)

    def _predict(self, state, inputs_phys: np.ndarray) -> np.ndarray:
        normed = self._normalize_inputs(inputs_phys)
        out = state.apply_fn(state.params, jnp.asarray(normed))
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        # Targets in the dataset are log-normalised to [0, 1]; convert back.
        return self.dataset.denormalize_targets_from_normalized(out)

    # ---- trainer callback contract ----------------------------------------

    def __call__(self, trainer, step: int) -> None:
        if step <= 0 or (step % self.every) != 0:
            if not self._warmup_done:
                # Warm-up: one tiny forward pass on the very first invocation
                # to amortise the JIT compile cost.
                _ = self._predict(trainer.state, self.inputs_A_phys[:1])
                self._warmup_done = True
            return

        if not self._warmup_done:
            self._warmup_done = True

        try:
            pred_A_flat = self._predict(trainer.state, self.inputs_A_phys)
            pred_B_flat = self._predict(trainer.state, self.inputs_B_phys)
            pred_C_flat = self._predict(trainer.state, self.inputs_C_phys)
        except Exception as exc:
            logger.warning("prediction_plot: inference failed at step %d: %s",
                           step, exc)
            return

        pred_A = pred_A_flat.reshape(self.inputs_A_shape)   # (n_E, n_slices, n_x)
        pred_B_full = pred_B_flat.reshape(self.inputs_B_shape)  # (n_E, n_x, n_d)
        pred_B = pred_B_full.sum(axis=1)                    # (n_E, n_d) — sum over angle/dedx
        pred_C = pred_C_flat.reshape(self.inputs_C_shape)   # (n_E, n_xslices, n_d)

        self._save_figure(step, pred_A, pred_B, pred_C)

    # ---- rendering ---------------------------------------------------------

    def _save_figure(self, step: int, pred_A: np.ndarray,
                     pred_B: np.ndarray, pred_C: np.ndarray) -> None:
        n_E = len(self.energy_indices)
        n_slices = len(self.distance_slice_indices)
        n_xslices = len(self.xaxis_slice_indices)

        x_label, smax_label = _AXIS_LABELS[self.table_type]
        y_label = _INTENSITY_LABELS[self.table_type]
        colors = [viridis(t) for t in np.linspace(0.05, 0.95, n_E)]

        n_cols = max(n_slices, n_xslices, 4)
        fig = Figure(figsize=(4.0 * n_cols, 10.0))
        canvas = FigureCanvasPdf(fig)
        gs = fig.add_gridspec(3, n_cols, height_ratios=[1.0, 1.0, 1.1])

        floor = 1e-6  # log-y floor so zeros don't blow up

        # --- Row 1 (view A): one panel per s/s_max slice, x = angle/dedx ---
        for m in range(n_slices):
            ax = fig.add_subplot(gs[0, m])
            for k in range(n_E):
                c = colors[k]
                ax.plot(self.x_centers, np.maximum(self.truth_A[k, m], floor),
                        "--", color=c, lw=1.0, alpha=0.85)
                ax.plot(self.x_centers, np.maximum(pred_A[k, m], floor),
                        "-", color=c, lw=1.4,
                        label=f"{int(self.energies_grid[k])} MeV")
            ax.axhline(self.zero_threshold, color="red", linestyle=":",
                       lw=0.9, alpha=0.6,
                       label=(f"zero threshold ({self.zero_threshold:g})"
                              if m == n_slices - 1 else None))
            ax.set_yscale("log")
            ax.set_title(f"{smax_label} = {self.distance_slices_grid[m]:.3f}")
            ax.set_xlabel(x_label)
            if m == 0:
                ax.set_ylabel(y_label)
            ax.grid(True, which="both", alpha=0.25)
            if m == n_slices - 1:
                ax.legend(fontsize=8, loc="upper right", frameon=False)

        # --- Row 2 (view C): one panel per x-axis slice, x = s/s_max -------
        for p in range(n_xslices):
            ax = fig.add_subplot(gs[1, p])
            for k in range(n_E):
                c = colors[k]
                ax.plot(self.distance_centers,
                        np.maximum(self.truth_C[k, p], floor),
                        "--", color=c, lw=1.0, alpha=0.85)
                ax.plot(self.distance_centers,
                        np.maximum(pred_C[k, p], floor),
                        "-", color=c, lw=1.4,
                        label=f"{int(self.energies_grid[k])} MeV")
            ax.axhline(self.zero_threshold, color="red", linestyle=":",
                       lw=0.9, alpha=0.6)
            ax.set_yscale("log")
            ax.set_title(_format_xaxis_title(self.table_type,
                                             self.xaxis_slices_grid[p]))
            ax.set_xlabel(smax_label)
            if p == 0:
                ax.set_ylabel(y_label)
            ax.grid(True, which="both", alpha=0.25)

        # --- Row 3 (view B): marginal over angle/dedx ----------------------
        # No zero-threshold line here: the threshold is a per-bin concept and
        # this panel sums over 500 bins, so the cutoff doesn't carry over.
        ax_b = fig.add_subplot(gs[2, :])
        for k in range(n_E):
            c = colors[k]
            ax_b.plot(self.distance_centers,
                      np.maximum(self.truth_B[k], floor),
                      "--", color=c, lw=1.0, alpha=0.85)
            ax_b.plot(self.distance_centers,
                      np.maximum(pred_B[k], floor),
                      "-", color=c, lw=1.6,
                      label=f"{int(self.energies_grid[k])} MeV")
        ax_b.set_yscale("log")
        ax_b.set_title(f"Marginal — summed over {x_label.split('[')[0].strip()}")
        ax_b.set_xlabel(smax_label)
        ax_b.set_ylabel(f"summed {y_label}")
        ax_b.grid(True, which="both", alpha=0.25)
        ax_b.legend(fontsize=8, loc="upper right", frameon=False, ncol=2)

        # Single super-title with the training step.
        fig.suptitle(
            f"SIREN ({self.table_type}) — step {step}   "
            f"(truth: dashed, prediction: solid)",
            fontsize=12, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        out_path = self.plot_dir / f"step_{step:06d}.pdf"
        canvas.print_pdf(str(out_path))


def _format_xaxis_title(table_type: str, value_native: float) -> str:
    """Subpanel title for the fixed-x-axis row.

    Photon: convert rad → deg for readability.
    dEdx:   show keV/mm directly.
    """
    if table_type == "photon":
        deg = np.degrees(value_native)
        return f"angle = {deg:.1f}°"
    return f"dE/dx = {value_native:g} keV/mm"


def parse_int_list(s: str) -> List[int]:
    """CLI helper: parse 'a,b,c' → [int(a), int(b), int(c)]."""
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]
