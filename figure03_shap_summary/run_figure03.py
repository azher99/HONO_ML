"""
Figure 3 - SHAP interpretation of the tuned CatBoost model, evaluated on the
validation subset using CatBoost native TreeSHAP.
(a) SHAP beeswarm summary; (b) mean absolute SHAP value per feature.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figure03_shap_summary/run_figure03.py

Output: outputs/figures/Figure03a_SHAP_summary_beeswarm.png
        outputs/figures/Figure03b_SHAP_mean_absolute.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool

import common as cm


def compute_shap(model, X_shap):
    """CatBoost native SHAP values; the final column is the base value."""
    raw = model.get_feature_importance(
        Pool(X_shap, feature_names=list(X_shap.columns)), type="ShapValues")
    return raw[:, :-1], raw[:, -1]


def main():
    (ml_data, X, y, X_train, X_test, X_val,
     y_train, y_test, y_val) = cm.get_data()

    best_models, _ = cm.tune_models(X_train, y_train, X_test, y_test,
                                    X_val, y_val)
    final_model = best_models["CatBoost"]

    X_shap = X_val[cm.PRIMARY_FEATURES].apply(pd.to_numeric, errors="coerce")
    shap_values, base_values = compute_shap(final_model, X_shap)

    importance = (pd.DataFrame({
        "Feature": cm.PRIMARY_FEATURES,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        "Mean_SHAP": shap_values.mean(axis=0),
        "SHAP_std": shap_values.std(axis=0)})
        .sort_values("Mean_abs_SHAP", ascending=False))

    labels = [cm.FEATURE_LABELS.get(c, c) for c in cm.PRIMARY_FEATURES]

    # ----------------------------------------------------------------
    # Panel (a): SHAP beeswarm summary
    # ----------------------------------------------------------------
    cm.set_plot_style(font_size=12)
    shap.summary_plot(shap_values, X_shap, feature_names=labels,
                      plot_type="dot", max_display=len(cm.PRIMARY_FEATURES),
                      show=False)
    fig_a = plt.gcf()
    fig_a.set_size_inches(7.4, 5.8)
    fig_a.gca().set_xlabel("SHAP value (impact on model output)")
    plt.tight_layout()
    cm.save_figure(fig_a, "Figure03a_SHAP_summary_beeswarm")
    plt.close(fig_a)

    # ----------------------------------------------------------------
    # Panel (b): mean absolute SHAP value
    # ----------------------------------------------------------------
    cm.set_plot_style(font_size=13)
    plot_imp = importance.copy()
    plot_imp["Label"] = plot_imp["Feature"].map(cm.FEATURE_LABELS).fillna(
        plot_imp["Feature"])

    fig_b, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.barh(plot_imp["Label"][::-1], plot_imp["Mean_abs_SHAP"][::-1],
            color="#2C7BB6", edgecolor="black", linewidth=0.8)
    for i, v in enumerate(plot_imp["Mean_abs_SHAP"][::-1]):
        ax.text(v + plot_imp["Mean_abs_SHAP"].max() * 0.015, i,
                f"{v:.3f}", va="center", fontsize=10.5)

    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_xlim(0, plot_imp["Mean_abs_SHAP"].max() * 1.15)
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.45)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    cm.save_figure(fig_b, "Figure03b_SHAP_mean_absolute")
    plt.close(fig_b)

    print("[info] feature ranking by mean absolute SHAP value:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
