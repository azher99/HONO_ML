"""
Figure S4 - Stability of the SHAP feature ranking across 20 independent
70/15/15 splits. (a) Mean absolute SHAP value per feature averaged 
over splits, annotated with the mean rank; (b) distribution of each
feature's rank. Included as a check on whether the reported ordering is
sensitive to a particular data partition.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology

Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

Run from the repository root:
    python figureS4_shap_rank_stability/run_figureS4.py

Output: outputs/figures/FigureS4_SHAP_rank_stability.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import Pool
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import common as cm


def main():
    (ml_data, X, y, X_train, X_test, X_val,
     y_train, y_test, y_val) = cm.get_data()

    best_models, _ = cm.tune_models(X_train, y_train, X_test, y_test,
                                    X_val, y_val)

    X_model = ml_data[cm.PRIMARY_FEATURES].apply(pd.to_numeric, errors="coerce")
    y_model = pd.to_numeric(ml_data[cm.TARGET], errors="coerce")
    keep = X_model.notna().all(axis=1) & y_model.notna()
    X_model = X_model.loc[keep].reset_index(drop=True)
    y_model = y_model.loc[keep].reset_index(drop=True)

    rows = []
    for seed in cm.SEED_RANGE:
        print(f"[stability] seed {seed} ...")
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_model, y_model, test_size=0.30, random_state=seed, shuffle=True)
        _, X_va, _, _ = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=seed, shuffle=True)

        model = clone(best_models["CatBoost"])
        try:
            model.set_params(random_seed=seed, verbose=False)
        except Exception:
            pass
        model.fit(X_tr, y_tr)

        raw = model.get_feature_importance(
            Pool(X_va[cm.PRIMARY_FEATURES],
                 feature_names=cm.PRIMARY_FEATURES), type="ShapValues")
        shap_vals = raw[:, :-1]

        tmp = pd.DataFrame({"Feature": cm.PRIMARY_FEATURES,
                            "Mean_abs_SHAP": np.abs(shap_vals).mean(axis=0)})
        tmp["Rank"] = tmp["Mean_abs_SHAP"].rank(ascending=False,
                                                method="min").astype(int)
        tmp["Seed"] = seed
        rows.append(tmp)

    stability = pd.concat(rows, ignore_index=True)
    summary = (stability.groupby("Feature")
               .agg(Mean_abs_SHAP_mean=("Mean_abs_SHAP", "mean"),
                    Mean_rank=("Rank", "mean"),
                    Top3_frequency=("Rank", lambda s: 100 * (s <= 3).mean()),
                    Top5_frequency=("Rank", lambda s: 100 * (s <= 5).mean()))
               .sort_values("Mean_rank").round(4))

    print(summary.to_string())

    plot_sum = (summary.reset_index()
                .sort_values("Mean_abs_SHAP_mean", ascending=True))
    plot_sum["Label"] = plot_sum["Feature"].map(cm.FEATURE_LABELS).fillna(
        plot_sum["Feature"])
    plot_rank = stability.copy()
    plot_rank["Label"] = plot_rank["Feature"].map(cm.FEATURE_LABELS).fillna(
        plot_rank["Feature"])

    cm.set_plot_style(font_size=12)
    fig, axes = plt.subplots(2, 1, figsize=(7, 11))

    axes[0].barh(plot_sum["Label"], plot_sum["Mean_abs_SHAP_mean"],
                 color="#2C7BB6", edgecolor="black", linewidth=0.75)
    for i, row in plot_sum.reset_index(drop=True).iterrows():
        axes[0].text(row["Mean_abs_SHAP_mean"]
                     + plot_sum["Mean_abs_SHAP_mean"].max() * 0.025, i,
                     f"Rank = {row['Mean_rank']:.1f}", va="center", fontsize=9.5)

    sns.boxplot(data=plot_rank, x="Rank", y="Label",
                order=plot_sum["Label"], ax=axes[1], color="#9ECAE1",
                linewidth=0.8, fliersize=2.5)
    sns.stripplot(data=plot_rank, x="Rank", y="Label",
                  order=plot_sum["Label"], ax=axes[1], color="#B2182B",
                  size=3, alpha=0.55, jitter=0.18)

    axes[0].set_title("(a) Mean absolute SHAP stability",
                      fontweight="bold", loc="left")
    axes[0].set_xlabel("Mean absolute SHAP value across repeated splits")
    axes[1].set_title("(b) SHAP rank distribution",
                      fontweight="bold", loc="left")
    axes[1].set_xlabel("Feature rank across repeated splits")
    axes[1].set_xlim(0.5, len(cm.PRIMARY_FEATURES) + 0.5)

    for ax in axes:
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    cm.save_figure(fig, "FigureS4_SHAP_rank_stability")
    plt.close(fig)


if __name__ == "__main__":
    main()
