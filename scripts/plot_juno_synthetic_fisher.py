#!/usr/bin/env python
"""Plot per-source Jacobian and numerical-basis Fisher diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_result(path):
    """Load and validate a numerical-basis Fisher result."""
    with Path(path).open() as stream:
        result = json.load(stream)
    expected = "lucid_per_source_numerical_basis_fisher_v1"
    if result.get("schema") != expected or not result.get("sources"):
        raise ValueError("input is not a populated JUNO Fisher result")
    return result


def _plot_source_page(pdf, result):
    """Plot response mass and Jacobian norms for every source."""
    sources = result["sources"]
    names = result["optimizer_coordinate_names"]
    labels = [
        f"z={source['position_m'][2]:g} m" for source in sources
    ]
    mass = [source["response_mass"] for source in sources]
    jacobian = np.asarray([
        source["jacobian_summary"]["l2_norm"] for source in sources
    ])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].bar(labels, mass, color="#4c78a8")
    axes[0].set_ylabel("Expected response mass [PE/event]")
    axes[0].set_title("Per-source response mass")
    x = np.arange(len(sources))
    for index, name in enumerate(names):
        axes[1].plot(x, jacobian[:, index], marker="o", label=name)
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Jacobian L2 norm")
    axes[1].set_title("Per-source response sensitivity")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Numerical-time-basis Jacobian diagnostics")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def _plot_fisher_page(pdf, result):
    """Plot the total Fisher correlation and eigenspectrum."""
    names = result["optimizer_coordinate_names"]
    total = result["total"]
    correlation = np.asarray(total["pseudoinverse_correlation"])
    eigenvalues = np.asarray(total["eigenvalues"])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    image = axes[0].imshow(
        correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm"
    )
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set_title("Pseudoinverse correlation")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].plot(
        np.arange(1, len(eigenvalues) + 1),
        np.sort(eigenvalues)[::-1],
        marker="o",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Ordered eigenvalue index")
    axes[1].set_ylabel("Fisher eigenvalue")
    axes[1].set_title(
        f"rank={total['rank']}, condition={total['condition_number']:.3g}"
    )
    axes[1].grid(alpha=0.25)
    figure.suptitle("Expected Poisson Fisher on the numerical time basis")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def render(result, output):
    """Write a two-page vector diagnostic PDF."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _plot_source_page(pdf, result)
        _plot_fisher_page(pdf, result)


def main():
    """Parse Fisher JSON and output PDF paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_result(args.result), args.output)


if __name__ == "__main__":
    main()
