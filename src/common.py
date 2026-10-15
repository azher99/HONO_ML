"""
Shared configuration, preprocessing, data splitting and model definitions.

Paper: Explainable Machine Learning Analysis of Indoor HONO Formation:
       Impacts of Combustion, Occupancy and Ventilation
       Deng et al., under review at Environmental Science & Technology
Code author: Muhammad Azher Hassan (azher@tju.edu.cn)

All figure scripts import this module so that the data pipeline, the
70/15/15 split.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Data.xlsx"

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

for _folder in (OUTPUT_DIR, FIG_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Dataset structure
# ------------------------------------------------------------------
TARGET = "HONO (ppb)"

EXACT_COLUMNS = [
    "Date", "HONO (ppb)", "NOx (ppb)", "O3 (ppb)", "RH (%)", "NH3 (ppb)",
    "N2O (ppm)", "CO2 (ppm)", "CH4 (ppm)", "TVOCs (ppb)",
    "Combustion", "Occupants", "Ventilation", "Group",
]

# Predictors retained in the main model after correlation-based screening.
# "Occupancy" is the binary (0/1) recoding of the "Occupants" count.
PRIMARY_FEATURES = [
    "NOx (ppb)", "O3 (ppb)", "N2O (ppm)", "RH (%)", "NH3 (ppb)",
    "TVOCs (ppb)", "Combustion", "Occupancy", "Ventilation",
]

# Retained for correlation screening only; excluded from the main model
# because of high pairwise Pearson correlation (|r| > 0.75).
SECONDARY_FEATURES = ["CO2 (ppm)", "CH4 (ppm)"]

CORR_COLS = [
    "HONO (ppb)", "NOx (ppb)", "O3 (ppb)", "RH (%)", "NH3 (ppb)", "N2O (ppm)",
    "CO2 (ppm)", "CH4 (ppm)", "TVOCs (ppb)", "Combustion", "Occupants",
    "Ventilation",
]

FILL_COLS = [
    "HONO (ppb)", "NOx (ppb)", "O3 (ppb)", "RH (%)", "NH3 (ppb)", "N2O (ppm)",
    "CO2 (ppm)", "CH4 (ppm)", "TVOCs (ppb)", "Combustion", "Occupants",
    "Ventilation",
]
CONT_COLS = [
    "NOx (ppb)", "O3 (ppb)", "RH (%)", "NH3 (ppb)", "N2O (ppm)",
    "CO2 (ppm)", "CH4 (ppm)", "TVOCs (ppb)",
]
OP_COLS = ["Combustion", "Occupants", "Ventilation"]

FEATURE_LABELS = {
    "O3 (ppb)": "O$_3$ (ppb)",
    "NOx (ppb)": "NO$_x$ (ppb)",
    "N2O (ppm)": "N$_2$O (ppm)",
    "RH (%)": "RH (%)",
    "NH3 (ppb)": "NH$_3$ (ppb)",
    "TVOCs (ppb)": "TVOCs (ppb)",
    "Combustion": "Combustion",
    "Occupancy": "Occupancy",
    "Occupants": "Occupants",
    "Ventilation": "Ventilation",
    "CO2 (ppm)": "CO$_2$ (ppm)",
    "CH4 (ppm)": "CH$_4$ (ppm)",
    "HONO (ppb)": "HONO (ppb)",
}

MAIN_SEED = 34
SEED_RANGE = list(range(15, 35))  # repeated-split robustness analysis

# ------------------------------------------------------------------
# Figure style (Times New Roman, embedded Type-42 fonts, 600 dpi)
# ------------------------------------------------------------------
def set_plot_style(font_size: float = 12) -> None:
    """Apply the manuscript figure style."""
    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 2,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, stem: str) -> None:
    """Save the figure as a 600 dpi PNG."""
    path = FIG_DIR / f"{stem}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", transparent=True)
    print(f"[saved] {path}")


def safe_name(s: str) -> str:
    """Filesystem-safe version of a variable name."""
    return (s.replace(" ", "_").replace("(", "").replace(")", "")
             .replace("%", "pct").replace("/", "_").replace("$", ""))


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    """Root-mean-square error (version-independent of scikit-learn)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "n": len(y_true),
    }


# ------------------------------------------------------------------
# Step 1 - load and preprocess
# ------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load the measured dataset, or a synthetic surrogate if it is absent."""
    if DATA_FILE.exists():
        raw = pd.read_excel(DATA_FILE)
        raw.columns = raw.columns.astype(str).str.strip()
        print(f"[data] measured dataset loaded: {raw.shape[0]} rows")
        return raw

    from make_synthetic_data import make_synthetic_dataset  # local import
    print("[data] data/Data.xlsx not found - using the synthetic surrogate "
          "dataset. Figure values will not match the manuscript.")
    return make_synthetic_dataset(seed=MAIN_SEED)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """
    NOx-based 1-minute alignment, Date x Group event labelling and
    missing-value filling.

    Continuous variables are linearly interpolated within each Date x Group
    event block, with a group-median and then global-median fallback.
    Operational variables are forward/backward filled within the event block,
    with a modal fallback.
    """
    dt = raw.dropna(subset=["NOx (ppb)"]).copy().reset_index(drop=True)

    dt["Event_Group"] = (dt["Date"].astype("Int64").astype(str)
                         + "_" + dt["Group"].astype(str))

    for c in FILL_COLS:
        dt[c] = pd.to_numeric(dt[c], errors="coerce")

    for c in CONT_COLS:
        dt[c] = dt.groupby("Event_Group", group_keys=False)[c].apply(
            lambda s: s.interpolate(method="linear", limit_direction="both"))
        dt[c] = dt.groupby("Group", group_keys=False)[c].apply(
            lambda s: s.fillna(s.median()))
        dt[c] = dt[c].fillna(dt[c].median())

    for c in OP_COLS:
        dt[c] = dt.groupby("Event_Group", group_keys=False)[c].apply(
            lambda s: s.ffill().bfill())
        dt[c] = (dt[c].fillna(dt[c].mode(dropna=True).iloc[0])
                      .round().astype(int))

    print(f"[data] preprocessing complete: {dt.shape[0]} rows retained")
    return dt


def build_model_frame(dt_model: pd.DataFrame) -> pd.DataFrame:
    """Add the binary Occupancy flag and drop rows with missing model inputs."""
    ml_data = dt_model.copy()
    ml_data["Occupancy"] = (
        pd.to_numeric(ml_data["Occupants"], errors="coerce").fillna(0) > 0
    ).astype(int)

    for c in PRIMARY_FEATURES + [TARGET]:
        ml_data[c] = pd.to_numeric(ml_data[c], errors="coerce")

    return ml_data.dropna(subset=PRIMARY_FEATURES + [TARGET]).reset_index(drop=True)


def split_data(ml_data: pd.DataFrame, seed: int = MAIN_SEED):
    """70 / 15 / 15 training / testing / validation split."""
    X = ml_data[PRIMARY_FEATURES].copy()
    y = ml_data[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, shuffle=True)

    return X, y, X_train, X_test, X_val, y_train, y_test, y_val


def get_data(seed: int = MAIN_SEED):
    """Convenience wrapper: raw -> preprocessed -> model frame -> split."""
    ml_data = build_model_frame(preprocess(load_raw()))
    return (ml_data,) + split_data(ml_data, seed=seed)


# ------------------------------------------------------------------
# Step 2 - candidate models and hyper-parameter search space
# ------------------------------------------------------------------
def _make_bagging(seed: int):
    """BaggingRegressor constructor compatible with old and new scikit-learn."""
    try:
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=seed),
            random_state=seed)
    except TypeError:  # scikit-learn < 1.2
        return BaggingRegressor(
            base_estimator=DecisionTreeRegressor(random_state=seed),
            random_state=seed)


def candidate_models(seed: int = MAIN_SEED) -> dict:
    return {
        "CatBoost": CatBoostRegressor(random_seed=seed, loss_function="RMSE",
                                      verbose=False),
        "XGBoost": XGBRegressor(random_state=seed,
                                objective="reg:squarederror",
                                eval_metric="rmse", n_jobs=-1),
        "Random Forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "Bagging Regressor": _make_bagging(seed),
    }


PARAM_GRIDS = {
    "CatBoost": {"iterations": [300, 600, 900], "depth": [3, 4, 6],
                 "learning_rate": [0.03, 0.05, 0.1], "l2_leaf_reg": [1, 3, 5]},
    "XGBoost": {"n_estimators": [200, 400, 600], "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]},
    "Random Forest": {"n_estimators": [200, 400, 600],
                      "max_depth": [None, 8, 12],
                      "min_samples_leaf": [1, 2, 4],
                      "max_features": ["sqrt", 0.8, 1.0]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300],
                          "max_depth": [2, 3, 4],
                          "learning_rate": [0.03, 0.05, 0.1],
                          "subsample": [0.8, 1.0]},
    "Bagging Regressor": {"n_estimators": [50, 100, 200],
                          "max_samples": [0.7, 0.9, 1.0],
                          "max_features": [0.7, 0.9, 1.0]},
}


def tune_models(X_train, y_train, X_test, y_test, X_val, y_val,
                seed: int = MAIN_SEED, n_iter: int = 10):
    """
    Randomised hyper-parameter search with 10-fold CV on the training set.

    Returns
    -------
    best_models : dict of fitted estimators
    model_results : long-format DataFrame of split-wise metrics
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    best_models, rows = {}, []

    for name, model in candidate_models(seed).items():
        print(f"[tuning] {name} ...")
        grid = PARAM_GRIDS[name]
        search = RandomizedSearchCV(
            model, grid,
            n_iter=min(n_iter, int(np.prod([len(v) for v in grid.values()]))),
            scoring="r2", cv=cv, n_jobs=-1, random_state=seed, refit=True)
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        for split, Xs, ys in (("Train", X_train, y_train),
                              ("Test", X_test, y_test),
                              ("Validation", X_val, y_val)):
            rows.append({"Model": name, "Split": split,
                         "Best_CV_R2": search.best_score_,
                         "Best_Params": json.dumps(search.best_params_),
                         **reg_metrics(ys, search.best_estimator_.predict(Xs))})

    return best_models, pd.DataFrame(rows)


def wide_results(model_results: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long metric table into one row per model."""
    rw = model_results.pivot(index="Model", columns="Split",
                             values=["R2", "RMSE", "MAE"])
    rw.columns = [f"{split}_{metric}" for metric, split in rw.columns]
    return rw.reset_index().set_index("Model")


def fit_catboost(X_train, y_train, best_models=None, seed: int = MAIN_SEED):
    """Return the tuned CatBoost model, or a default one if tuning is skipped."""
    if best_models is not None and "CatBoost" in best_models:
        return best_models["CatBoost"]
    model = CatBoostRegressor(random_seed=seed, loss_function="RMSE",
                              verbose=False)
    model.fit(X_train, y_train)
    return model
