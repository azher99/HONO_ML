"""
Figure S3 - Pairwise Pearson correlation matrix for all measured variables.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figureS3_correlation_matrix/run_figureS3.py

Output: outputs/figures/FigureS3_pearson_correlation_matrix.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as cm

LABELS = ["HONO", "NO$_x$", "O$_3$", "RH", "NH$_3$", "N$_2$O",
          "CO$_2$", "CH$_4$", "TVOCs", "Combustion", "Occupants", "Ventilation"]
REDUNDANCY_THRESHOLD = 0.75


def sig_star(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    dt_model = cm.preprocess(cm.load_raw())
    df = dt_model[cm.CORR_COLS].apply(pd.to_numeric, errors="coerce")

    n = len(cm.CORR_COLS)
    corr = pd.DataFrame(np.nan, index=cm.CORR_COLS, columns=cm.CORR_COLS)
    pvals = corr.copy()

    for i, a in enumerate(cm.CORR_COLS):
        for j, b in enumerate(cm.CORR_COLS):
            t = pd.concat([df[a], df[b]], axis=1).dropna()
            x, y = t.iloc[:, 0], t.iloc[:, 1]
            if len(t) >= 3 and x.nunique() > 1 and y.nunique() > 1:
                corr.iloc[i, j], pvals.iloc[i, j] = stats.pearsonr(x, y)

    cm.set_plot_style(font_size=11)
    cmap, norm = plt.cm.RdYlBu_r, mcolors.Normalize(-1, 1)
    fig, axes = plt.subplots(n, n, figsize=(13.5, 12.5))
    plt.subplots_adjust(hspace=0.04, wspace=0.04)

    for i in range(n):
        for j in range(n):
            ax, r, p = axes[i, j], corr.iloc[i, j], pvals.iloc[i, j]
            ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1),
                   facecolor="white")
            for sp in ax.spines.values():
                sp.set(linewidth=0.35, edgecolor="#D0D0D0")

            if i == j:
                ax.plot([0, 1], [1, 0], color="black", lw=1.15)

            elif i < j and pd.notna(r):
                if abs(r) > REDUNDANCY_THRESHOLD:
                    for sp in ax.spines.values():
                        sp.set(edgecolor="red", linewidth=1.8)
                ax.text(0.5, 0.58, f"{r:.3f}", ha="center", va="center",
                        fontsize=11.5, fontweight="bold")
                ax.text(0.5, 0.30, sig_star(p), ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color="#B2182B")

            elif i > j and pd.notna(r):
                size = 0.18 + 0.74 * abs(r)
                m = (1 - size) / 2
                ax.add_patch(mpatches.Rectangle(
                    (m, m), size, size, facecolor=cmap(norm(r)),
                    edgecolor="#222222", lw=0.45, alpha=0.92))

            if j == 0:
                ax.set_ylabel(LABELS[i], fontsize=11.5, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=6)
            if i == n - 1:
                ax.set_xlabel(LABELS[j], fontsize=11.5, fontweight="bold",
                              rotation=45, ha="right", va="top", labelpad=8)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.92, 0.18, 0.018, 0.62]))
    cbar.set_label("Pearson correlation coefficient (r)",
                   fontsize=12.5, fontweight="bold")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=10)

    cm.save_figure(fig, "FigureS3_pearson_correlation_matrix")
    plt.close(fig)


if __name__ == "__main__":
    main()
