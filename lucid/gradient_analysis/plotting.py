
"""Gradient analysis plotting utilities for LUCiD.

Standardized style: LogNorm + viridis + deeppink markers + white streamlines.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# LUCiD plot style
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'font.size': 10,
})


# ---------------------------------------------------------------------------
# 1D plotting
# ---------------------------------------------------------------------------

def plot_sweep_1d(results, title=None, show_numerical=True, save_path=None):
    """Plot 1D sweep results: loss curves and gradient comparison.

    Args:
        results: Dict[str, SweepResult1D] from sweep_1d
        title: optional figure title
        show_numerical: overlay finite-difference gradients
        save_path: save figure if provided

    Returns:
        (fig, axes)
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)

    for col, (name, res) in enumerate(results.items()):
        ax_loss = axes[0, col]
        ax_grad = axes[1, col]

        # Loss curve
        ax_loss.plot(res.values, res.losses, 'b-', linewidth=1.5)
        ax_loss.axvline(
            res.param.center, color='deeppink', ls='--', lw=1.2, label='True',
        )
        ax_loss.set_xlabel(res.param.label)
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(alpha=0.3)
        if col == 0:
            ax_loss.legend()

        # Gradient comparison
        ax_grad.plot(
            res.values, res.gradients, 'r-', linewidth=1.5, label='Autodiff',
        )
        if show_numerical:
            ax_grad.plot(
                res.values, res.numerical_grads, 'k--', linewidth=1.0,
                label='Finite diff',
            )
        ax_grad.axhline(0, color='gray', ls=':', lw=0.8)
        ax_grad.set_xlabel(res.param.label)
        ax_grad.set_ylabel('Gradient')
        ax_grad.grid(alpha=0.3)
        if col == 0:
            ax_grad.legend()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ---------------------------------------------------------------------------
# 2D plotting helpers
# ---------------------------------------------------------------------------

def _plot_2d_panel(ax, result, vmin, vmax):
    """Render a single 2D loss-surface panel with streamlines."""
    # Delta-loss for LogNorm
    delta = result.losses - np.nanmin(result.losses) + 1e-6

    extent = [
        float(result.x_values[0]), float(result.x_values[-1]),
        float(result.y_values[0]), float(result.y_values[-1]),
    ]

    im = ax.imshow(
        delta.T, extent=extent, origin='lower', aspect='auto',
        cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    # Streamlines with grad_scale applied at plot time
    U = -result.grad_x * result.param_x.grad_scale
    V = -result.grad_y * result.param_y.grad_scale
    norm = np.sqrt(U ** 2 + V ** 2) + 1e-10
    U_norm, V_norm = U / norm, V / norm

    x_stream = np.linspace(
        float(result.x_values[0]), float(result.x_values[-1]),
        len(result.x_values),
    )
    y_stream = np.linspace(
        float(result.y_values[0]), float(result.y_values[-1]),
        len(result.y_values),
    )

    ax.streamplot(
        x_stream, y_stream, U_norm.T, V_norm.T,
        color='white', density=1.0, linewidth=0.7, arrowsize=0.5,
    )

    # True-value marker
    ax.plot(
        result.param_x.center, result.param_y.center,
        '*', color='deeppink', markersize=20,
        markeredgecolor='white', markeredgewidth=0.5,
    )

    ax.set_xlabel(result.param_x.label)
    ax.set_ylabel(result.param_y.label)
    ax.set_title(f'{result.param_x.name} vs {result.param_y.name}')

    return im


# ---------------------------------------------------------------------------
# 2D grid plot
# ---------------------------------------------------------------------------

def plot_sweep_2d(results_list, title=None, save_path=None):
    """Plot 2D sweep results in a grid with unified colorbar.

    Args:
        results_list: list of SweepResult2D
        title: optional figure title
        save_path: save figure if provided

    Returns:
        (fig, axes)
    """
    n = len(results_list)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols + 0.8, 3.5 * n_rows),
        squeeze=False,
    )

    # Global vmin/vmax from delta-losses
    all_deltas = []
    for res in results_list:
        delta = res.losses - np.nanmin(res.losses) + 1e-6
        all_deltas.append(delta)
    all_flat = np.concatenate([d.ravel() for d in all_deltas])
    vmax = float(np.nanpercentile(all_flat, 95))
    vmin = min(1e-1, vmax * 0.01)

    im = None
    for idx, res in enumerate(results_list):
        row, col = divmod(idx, n_cols)
        im = _plot_2d_panel(axes[row, col], res, vmin, vmax)

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Unified colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='\u0394Loss')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ---------------------------------------------------------------------------
# 2D single-panel plot
# ---------------------------------------------------------------------------

def plot_sweep_2d_single(result, title=None, save_path=None):
    """Plot a single 2D sweep result.

    Args:
        result: SweepResult2D
        title: optional panel title
        save_path: save figure if provided

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    delta = result.losses - np.nanmin(result.losses) + 1e-6
    vmax = float(np.nanpercentile(delta, 95))
    vmin = min(1e-1, vmax * 0.01)

    im = _plot_2d_panel(ax, result, vmin, vmax)

    if title:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, pad=0.02, label='\u0394Loss')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
