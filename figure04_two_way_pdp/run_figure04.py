"""
Figure 4 - Two-way partial dependence surfaces for nine feature pairs (3x3)

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figure04_two_way_pdp/run_figure04.py

Output: outputs/figures/Figure04_two_way_PDP.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as cm

INTERACTION_PAIRS = [
    ("RH (%)", "NOx (ppb)"),
    ("RH (%)", "NH3 (ppb)"),
    ("RH (%)", "TVOCs (ppb)"),
    ("RH (%)", "O3 (ppb)"),
    ("NH3 (ppb)", "TVOCs (ppb)"),
    ("NOx (ppb)", "O3 (ppb)"),
    ("RH (%)", "Combustion"),
    ("RH (%)", "Occupancy"),
    ("RH (%)", "Ventilation"),
]

N_GRID = 40
N_REFERENCE = 350


def feature_grid(x, n_grid=N_GRID):
    """Evaluation grid: unique levels for binary/categorical variables,
    otherwise the 1st-99th percentile range split into n_grid points."""
    vals = np.sort(pd.Series(x).dropna().unique())
    if len(vals) <= 5:
        return vals
    return np.linspace(np.nanquantile(x, 0.01), np.nanquantile(x, 0.99), n_grid)


def compute_2d_pdp(model, X_ref, f1, f2, n_grid=N_GRID):
    """Two-dimensional partial dependence surface for the pair (f1, f2)."""
    g1, g2 = feature_grid(X_ref[f1], n_grid), feature_grid(X_ref[f2], n_grid)
    z = np.zeros((len(g2), len(g1)))
    for ii, v2 in enumerate(g2):
        for jj, v1 in enumerate(g1):
            X_tmp = X_ref.copy()
            X_tmp[f1], X_tmp[f2] = v1, v2
            z[ii, jj] = model.predict(X_tmp).mean()
    return g1, g2, z


def main():
    (ml_data, X, y, X_train, X_test, X_val,
     y_train, y_test, y_val) = cm.get_data()

    best_models, _ = cm.tune_models(X_train, y_train, X_test, y_test,
                                    X_val, y_val)
    final_model = best_models["CatBoost"]

    X_ref = (X_train[cm.PRIMARY_FEATURES]
             .apply(pd.to_numeric, errors="coerce")
             .sample(n=min(N_REFERENCE, len(X_train)),
                     random_state=cm.MAIN_SEED))

    pairs = [(a, b) for a, b in INTERACTION_PAIRS
             if a in cm.PRIMARY_FEATURES and b in cm.PRIMARY_FEATURES]

    results = {}
    for f1, f2 in pairs:
        print(f"[pdp] {f1} x {f2} ...")
        g1, g2, z = compute_2d_pdp(final_model, X_ref, f1, f2)
        results[(f1, f2)] = (g1, g2, z)

    vmin = min(z.min() for _, _, z in results.values())
    vmax = max(z.max() for _, _, z in results.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fill_levels = np.linspace(vmin, vmax, 21)

    cm.set_plot_style(font_size=14)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13))
    axes = axes.reshape(-1)
    panel_labels = list("abcdefghi")

    for i, ((f1, f2), (g1, g2, z)) in enumerate(results.items()):
        ax = axes[i]
        ax.contourf(g1, g2, z, levels=fill_levels, cmap="RdYlBu_r",
                    norm=norm, extend="both")

        if np.nanmax(z) > np.nanmin(z):
            label_levels = np.linspace(np.nanmin(z), np.nanmax(z), 10)
            cs = ax.contour(g1, g2, z, levels=label_levels, colors="black",
                            linewidths=0.7, alpha=0.75)
            ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.2f}",
                      fontsize=11, colors="black")

        for grid, setter, labeller in ((g1, ax.set_xticks, ax.set_xticklabels),
                                       (g2, ax.set_yticks, ax.set_yticklabels)):
            if len(grid) <= 5:
                setter(grid)
                labeller([str(int(v)) if float(v).is_integer() else f"{v:.2f}"
                          for v in grid])

        ax.text(-0.18, 1.08, f"({panel_labels[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold")
        ax.set_xlabel(cm.FEATURE_LABELS.get(f1, f1))
        ax.set_ylabel(cm.FEATURE_LABELS.get(f2, f2))
        ax.set_title(f"{cm.FEATURE_LABELS.get(f1, f1)} × "
                     f"{cm.FEATURE_LABELS.get(f2, f2)}",
                     fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(results), 9):
        axes[j].axis("off")

    cbar_ax = fig.add_axes([1.03, 0.13, 0.015, 0.85])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
                        cax=cbar_ax, orientation="vertical")
    cbar.set_label("Predicted HONO (ppb)", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1.02, 1.02], w_pad=1.5, h_pad=1.5)
    cm.save_figure(fig, "Figure04_two_way_PDP")
    plt.close(fig)


if __name__ == "__main__":
    main()
