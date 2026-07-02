#!/usr/bin/env python3
"""Plot the cached S-yield loss scan: NLL(S) with autodiff-gradient tangents, and
dNLL/dS(S) with its zero crossing. Reads data/juno_S_loss_scan.npz."""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"   # match serif text in math labels
plt.rcParams["font.size"] = 13

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "data" / "juno_S_loss_scan.npz")
S, S0, loss, grad = z["S"], float(z["S0"]), z["loss"], z["grad"]
r = S / S0
step = r[1] - r[0]

# zero crossing of the gradient (linear interp) -> best-fit S
i = np.where(np.diff(np.sign(grad)) != 0)[0]
r_min = None
if len(i):
    j = i[0]
    r_min = r[j] - grad[j] * (r[j + 1] - r[j]) / (grad[j + 1] - grad[j])

dloss = loss - loss.min()             # delta loss (subtract minimum)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08),
                               facecolor="white")

# --- delta loss -------------------------------------------------------------
ax1.plot(r, dloss, "-", color="0.6", lw=1.4, zorder=1)
ax1.plot(r, dloss, "o", color="#1f4e79", ms=6, zorder=3)
ax1.set_ylabel(r"$\Delta \mathcal{L}$")
if r_min is not None:
    ax1.axvline(r_min, color="0.5", ls=":", lw=1.2)

# --- gradient with zero crossing --------------------------------------------
ax2.axhline(0.0, color="0.5", lw=1.0)
ax2.plot(r, grad, "-", color="0.6", lw=1.4, zorder=1)
ax2.plot(r, grad, "s", color="#1f4e79", ms=6, zorder=3)
ax2.set_ylabel(r"$d\mathcal{L}/dS$")
ax2.set_xlabel(r"scintillation yield  $S / S_0$")
if r_min is not None:
    ax2.axvline(r_min, color="0.5", ls=":", lw=1.2)
    ax2.plot(r_min, 0.0, "*", color="#c0392b", ms=15, zorder=4,
             label=r"$\hat{S}=%.2f\,S_0$" % r_min)
    ax2.legend(frameon=False, loc="lower right")

base = HERE / "figures" / "juno_S_loss_scan"
for ext in ("pdf", "png"):
    fig.savefig(f"{base}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved {base}.pdf (+png); zero-crossing S_hat = "
      f"{r_min:.3f} S0" if r_min else "no zero crossing in range")


if __name__ == "__main__":
    pass
