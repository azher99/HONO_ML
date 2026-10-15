"""
Figure 2 - Measured versus predicted indoor HONO for the five best tuned ensemble
models (panels a-e, training set inset) and the 10-fold cross-validation
distribution of R2, RMSE and MAE (panel f).

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figure02_model_performance/run_figure02.py

Output: outputs/figures/Figure02_model_performance.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

import common as cm

MODEL_ABBR = {"CatBoost": "CB", "Random Forest": "RF", "Bagging Regressor": "BR",
              "XGBoost": "XGB", "Gradient Boosting": "GB", "Extra Trees": "ET"}


def cross_validated_metrics(best_models, top_models, X_train, y_train):
    """True 10-fold CV metrics on the training set for each tuned model."""
    cv = KFold(n_splits=10, shuffle=True, random_state=cm.MAIN_SEED)
    scoring = {"R2": "r2",
               "RMSE": "neg_root_mean_squared_error",
               "MAE": "neg_mean_absolute_error"}

    rows = []
    for name in top_models:
        print(f"[cv] 10-fold cross-validation for {name} ...")
        sc = cross_validate(clone(best_models[name]), X_train, y_train,
                            cv=cv, scoring=scoring, n_jobs=-1)
        rows += [{"Model": name, "Fold": i + 1,
                  "R2": sc["test_R2"][i],
                  "RMSE": -sc["test_RMSE"][i],
                  "MAE": -sc["test_MAE"][i]} for i in range(10)]

    return pd.DataFrame(rows)


def add_regression_line(ax, y_true, y_pred, color, x_line, style="-"):
    """Ordinary least-squares fit of predicted on measured values."""
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        slope, intercept = np.polyfit(np.asarray(y_true), np.asarray(y_pred), 1)
        ax.plot(x_line, slope * x_line + intercept, color=color, lw=2, ls=style)


def main():
    (ml_data, X, y, X_train, X_test, X_val,
     y_train, y_test, y_val) = cm.get_data()

    best_models, model_results = cm.tune_models(
        X_train, y_train, X_test, y_test, X_val, y_val)

    results = cm.wide_results(model_results)
    top_models = (results.sort_values("Validation_R2", ascending=False)
                         .head(5).index.tolist())
    print("[info] models ordered by validation R2:", top_models)
    print(results.loc[top_models].round(3).to_string())

    pred = {name: {"Train": np.asarray(best_models[name].predict(X_train)),
                   "Test": np.asarray(best_models[name].predict(X_test)),
                   "Validation": np.asarray(best_models[name].predict(X_val))}
            for name in top_models}

    cv_metrics = cross_validated_metrics(best_models, top_models,
                                         X_train, y_train)

    # ----------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------
    cm.set_plot_style(font_size=15)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = {"Train": "#1b72b1", "Test": "#ff7f0e", "Validation": "#2ca02c"}
    axis_min, axis_max = 0, 7.35
    x_line = np.linspace(axis_min, axis_max, 100)

    for i, name in enumerate(top_models):
        ax = axes[i]
        yt, yp_t = np.asarray(y_test), pred[name]["Test"]
        yv, yp_v = np.asarray(y_val), pred[name]["Validation"]
        ytr, yp_tr = np.asarray(y_train), pred[name]["Train"]

        ax.scatter(yt, yp_t, color=colors["Test"], alpha=0.80,
                   edgecolor="k", linewidth=0.4, s=42, label="Test")
        ax.scatter(yv, yp_v, color=colors["Validation"], alpha=0.75,
                   edgecolor="k", linewidth=0.4, s=42, label="Validation")
        add_regression_line(ax, yt, yp_t, colors["Test"], x_line, "-")
        add_regression_line(ax, yv, yp_v, colors["Validation"], x_line, "-.")

        # Inset: training set
        ax_in = inset_axes(ax, width="40%", height="40%", loc="upper left")
        ax_in.scatter(ytr, yp_tr, color=colors["Train"], alpha=0.55,
                      edgecolor="k", linewidth=0.25, s=18)
        add_regression_line(ax_in, ytr, yp_tr, colors["Train"], x_line, "--")
        ax_in.set_xlim(axis_min, axis_max)
        ax_in.set_ylim(axis_min, axis_max)
        ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.grid(False)

        r = results.loc[name]
        ax_in.text(0.04, 0.95,
                   f"Train $R^2$ = {r['Train_R2']:.3f}\n"
                   f"RMSE = {r['Train_RMSE']:.3f}",
                   transform=ax_in.transAxes, va="top", ha="left", fontsize=12)
        ax.text(0.96, 0.04,
                f"Validation $R^2$ = {r['Validation_R2']:.3f}\n"
                f"Validation RMSE = {r['Validation_RMSE']:.3f}\n"
                f"Test $R^2$ = {r['Test_R2']:.3f}\n"
                f"Test RMSE = {r['Test_RMSE']:.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"))

        ax.set_xlim(axis_min, axis_max); ax.set_ylim(axis_min, axis_max)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel("Measured HONO (ppb)")
        ax.set_ylabel("Predicted HONO (ppb)")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(False)

    # Panel (f): 10-fold cross-validation distributions
    ax = axes[5]
    cv_melt = cv_metrics.melt(id_vars=["Model", "Fold"],
                              value_vars=["R2", "RMSE", "MAE"],
                              var_name="Metric", value_name="Value")
    cv_melt["Model"] = cv_melt["Model"].map(MODEL_ABBR).fillna(cv_melt["Model"])

    sns.boxplot(data=cv_melt, x="Model", y="Value", hue="Metric", ax=ax,
                width=0.62, linewidth=1.0, palette="Set2")
    ax.set_title("$R^2$, RMSE and MAE (CV = 10)", fontweight="bold")
    ax.set_xlabel("Models")
    ax.set_ylabel("Metric value")
    ax.legend(title=None, loc="upper left", frameon=True)
    ax.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    cm.save_figure(fig, "Figure02_model_performance")
    plt.close(fig)


if __name__ == "__main__":
    main()
