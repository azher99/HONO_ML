"""
Figure S5 - One-way partial dependence and individual conditional expectation
(ICE) curves. Light curves are per-observation ICE trajectories, the solid
line is the partial dependence, and the shaded band spans the 5th-95th ICE
percentiles. Spread between ICE curves indicates heterogeneity in the
modelled response, consistent with the interactions examined in Figure 4.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figureS5_pdp_ice/run_figureS5.py

Output: outputs/figures/FigureS5_PDP_ICE.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as cm

PDP_FEATURES = ["RH (%)", "NOx (ppb)", "NH3 (ppb)", "TVOCs (ppb)",
                "O3 (ppb)", "N2O (ppm)", "Occupancy", "Combustion",
                "Ventilation"]
N_GRID = 65
N_REFERENCE = 350
N_ICE_DRAWN = 120

ICE_COLOR, PDP_COLOR, CI_COLOR = "#63C1A4", "#003366", "#BFD7EA"


def feature_grid(x, n_grid=N_GRID):
    vals = np.sort(pd.Series(x).dropna().unique())
    return vals if len(vals) <= 5 else np.linspace(np.nanmin(x),
                                                   np.nanmax(x), n_grid)


def compute_ice_pdp(model, X_ref, feature, n_grid=N_GRID):
    """ICE matrix (observations x grid) and its column mean (the PDP)."""
    grid = feature_grid(X_ref[feature], n_grid)
    ice = []
    for v in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature] = v
        ice.append(model.predict(X_tmp))
    ice = np.asarray(ice).T
    return grid, ice, ice.mean(axis=0)


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

    features = [f for f in PDP_FEATURES if f in cm.PRIMARY_FEATURES]

    cm.set_plot_style(font_size=11.5)
    plt.rcParams.update({"legend.fontsize": 8.7})
    fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
    axes = axes.reshape(-1)
    panel_labels = list("abcdefghi")
    rng = np.random.RandomState(cm.MAIN_SEED)

    legend_handles = [
        Line2D([0], [0], color=ICE_COLOR, lw=1.2, alpha=0.75, label="ICE"),
        Line2D([0], [0], color=PDP_COLOR, lw=2.8, label="PDP"),
        Patch(facecolor=CI_COLOR, edgecolor="none", alpha=0.45,
              label="5th-95th ICE percentile"),
    ]

    for i, feat in enumerate(features):
        print(f"[pdp/ice] {feat} ...")
        ax = axes[i]
        grid, ice, pdp = compute_ice_pdp(final_model, X_ref, feat)
        q05, q95 = np.percentile(ice, [5, 95], axis=0)

        draw = ice[rng.choice(ice.shape[0],
                              size=min(N_ICE_DRAWN, ice.shape[0]),
                              replace=False)]
        for row in draw:
            ax.plot(grid, row, color=ICE_COLOR, lw=0.75, alpha=0.18, zorder=1)

        ax.fill_between(grid, q05, q95, color=CI_COLOR, alpha=0.45,
                        linewidth=0, zorder=2)
        ax.plot(grid, pdp, color=PDP_COLOR, lw=2.8,
                marker="o" if len(grid) <= 5 else None,
                markersize=4.5, zorder=4)

        if len(grid) <= 5:
            ax.set_xticks(grid)
            ax.set_xticklabels(["No", "Yes"] if len(grid) == 2 else
                               [f"{v:.2f}" for v in grid])

        ax.text(-0.15, 1.10, f"({panel_labels[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=15, fontweight="bold")
        ax.set_title(cm.FEATURE_LABELS.get(feat, feat),
                     fontweight="bold", pad=8)
        ax.set_xlabel(cm.FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel("Partial dependence\nPredicted HONO (ppb)")
        ax.legend(handles=legend_handles, loc="best", frameon=True,
                  facecolor="white", edgecolor="gray", framealpha=0.82,
                  borderpad=0.35, handlelength=1.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(w_pad=1.20, h_pad=1.25)
    cm.save_figure(fig, "FigureS5_PDP_ICE")
    plt.close(fig)


if __name__ == "__main__":
    main()
