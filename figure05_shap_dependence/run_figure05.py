"""
Figure 5 - SHAP dependence plots for the tuned CatBoost model. Continuous 
features (a-f) are summarised with a cubic polynomial trend 
and a bootstrap 95% band; binary operational features (g-i)
with the group mean and a 95% confidence interval. The fitted trends are
descriptive summaries of the model output rather than mechanistic
dose-response relationships.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figure05_shap_dependence/run_figure05.py

Output: outputs/figures/Figure05_SHAP_dependence.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

import common as cm

SELECTED_FEATURES = ["RH (%)", "NOx (ppb)", "NH3 (ppb)", "TVOCs (ppb)",
                     "O3 (ppb)", "N2O (ppm)", "Combustion", "Occupancy",
                     "Ventilation"]
BINARY_FEATURES = {"Combustion", "Occupancy", "Ventilation"}
N_BOOT = 120
POLY_DEGREE = 3


def plot_binary_shap(ax, x_vals, shap_y, rng):
    """Jittered points plus group mean with a 95% confidence interval."""
    temp = pd.DataFrame({"x": pd.Series(x_vals).round().astype(int),
                         "shap": shap_y}).dropna()
    temp = temp[temp["x"].isin([0, 1])]
    temp["jitter"] = temp["x"] + rng.uniform(-0.055, 0.055, len(temp))

    ax.scatter(temp["jitter"], temp["shap"], s=18, alpha=0.55,
               color="#2C7BB6", edgecolor="none")

    rows = []
    for v in (0, 1):
        g = temp.loc[temp["x"] == v, "shap"].values
        rows.append({"x": v,
                     "mean": np.mean(g) if len(g) else np.nan,
                     "ci": 1.96 * stats.sem(g) if len(g) > 1 else np.nan,
                     "n": len(g)})
    sdat = pd.DataFrame(rows)

    ax.errorbar(sdat["x"], sdat["mean"], yerr=sdat["ci"], fmt="o-",
                color="#B2182B", ecolor="#B2182B", capsize=4,
                lw=2.4, markersize=5.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    return sdat


def plot_continuous_shap(ax, x_vals, shap_y, rng, n_boot=N_BOOT):
    """Scatter plus a bootstrapped cubic polynomial trend with a 95% band."""
    temp = (pd.DataFrame({"x": x_vals, "shap": shap_y})
            .replace([np.inf, -np.inf], np.nan).dropna())
    ax.scatter(temp["x"], temp["shap"], c=temp["shap"], cmap="Reds",
               s=16, alpha=0.62, edgecolor="none")

    if temp["x"].nunique() < 6:
        return None

    x_arr, y_arr = temp["x"].values, temp["shap"].values
    x_grid = np.linspace(np.nanquantile(x_arr, 0.005),
                         np.nanquantile(x_arr, 0.995), 180).reshape(-1, 1)

    def poly_model():
        return Pipeline([("poly", PolynomialFeatures(degree=POLY_DEGREE,
                                                     include_bias=False)),
                         ("linear", LinearRegression())])

    preds = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(x_arr)), replace=True,
                       random_state=int(rng.integers(1, 10 ** 9)))
        m = poly_model().fit(x_arr[idx].reshape(-1, 1), y_arr[idx])
        preds.append(m.predict(x_grid))
    preds = np.asarray(preds)

    fitted = poly_model().fit(x_arr.reshape(-1, 1), y_arr).predict(x_grid)
    lo, hi = np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)

    ax.fill_between(x_grid.ravel(), lo, hi, color="#9ECAE1",
                    alpha=0.35, linewidth=0)
    ax.plot(x_grid.ravel(), fitted, color="#B2182B", lw=2.5)

    return pd.DataFrame({"x_grid": x_grid.ravel(), "trend": fitted,
                         "ci_low": lo, "ci_high": hi})


def main():
    (ml_data, X, y, X_train, X_test, X_val,
     y_train, y_test, y_val) = cm.get_data()

    best_models, _ = cm.tune_models(X_train, y_train, X_test, y_test,
                                    X_val, y_val)
    final_model = best_models["CatBoost"]

    X_shap = X_val[cm.PRIMARY_FEATURES].apply(pd.to_numeric, errors="coerce")
    raw = final_model.get_feature_importance(
        Pool(X_shap, feature_names=cm.PRIMARY_FEATURES), type="ShapValues")
    shap_values = raw[:, :-1]

    features = [f for f in SELECTED_FEATURES if f in cm.PRIMARY_FEATURES]

    cm.set_plot_style(font_size=14)
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = axes.reshape(-1)
    rng = np.random.default_rng(cm.MAIN_SEED)
    panel_labels = list("abcdefghi")
    trend_outputs = {}

    for i, feat in enumerate(features):
        ax = axes[i]
        j = cm.PRIMARY_FEATURES.index(feat)
        x_vals, shap_y = X_shap[feat].values, shap_values[:, j]

        trend_outputs[feat] = (
            plot_binary_shap(ax, x_vals, shap_y, rng)
            if feat in BINARY_FEATURES
            else plot_continuous_shap(ax, x_vals, shap_y, rng))

        ax.axhline(0, color="black", lw=0.75, linestyle="--", alpha=0.65)
        ax.text(-0.18, 1.08, f"({panel_labels[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold")
        ax.set_title(cm.FEATURE_LABELS.get(feat, feat),
                     fontweight="bold", pad=8)
        ax.set_xlabel(cm.FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel("SHAP value")
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.40)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(features), 9):
        axes[j].axis("off")

    plt.tight_layout(w_pad=1.2, h_pad=1.2)
    cm.save_figure(fig, "Figure05_SHAP_dependence")
    plt.close(fig)


if __name__ == "__main__":
    main()
