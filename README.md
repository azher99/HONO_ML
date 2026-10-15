# Explainable Machine Learning Analysis of Indoor HONO Formation

### Impacts of Combustion, Occupancy and Ventilation

Figure reproduction code accompanying:

> **Explainable Machine Learning Analysis of Indoor HONO Formation: Impacts of
> Combustion, Occupancy and Ventilation**
>
> Qingxin Deng, Muhammad Azher Hassan\*, Bowen He, Zhu Ran, Yan Ren,
> Manolis Romanias, Adrien Gandolfo, Peng Zheng, Bin Yuan,
> Jiangping Liu\*, Sasho Gligorovski\*
>
> Under review at *Environmental Science & Technology* (ES&T).
>
> \* Corresponding authors.

**Code author and contact:** Muhammad Azher Hassan — azher@tju.edu.cn

This repository contains the analysis and plotting code required to reproduce
each figure in the manuscript and the supporting information. It is organised
as one self-contained folder per figure so that a specific result can be traced
to the code that produced it.

---

## Repository layout

```
HONO-ML-indoor/
├── src/
│   ├── common.py                 # shared configuration, preprocessing, split, models
│
├── figure02_model_performance/   # Fig. 2  — measured vs. predicted + 10-fold CV
├── figure03_shap_summary/        # Fig. 3  — SHAP beeswarm + mean |SHAP|
├── figure04_two_way_pdp/         # Fig. 4  — two-way partial dependence surfaces
├── figure05_shap_dependence/     # Fig. 5  — SHAP dependence with fitted trends
│
├── figureS3_correlation_matrix/  # Fig. S3 — Pearson correlation matrix
├── figureS4_shap_rank_stability/ # Fig. S4 — SHAP ranking stability (20 splits)
├── figureS5_pdp_ice/             # Fig. S5 — one-way PDP and ICE curves
│
├── data/                         # place Data.xlsx here (not distributed)
├── outputs/figures/              # generated PNG figures
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

| Script | Figure | Location in paper |
|---|---|---|
| `figure02_model_performance/run_figure02.py` | Figure 2 | Main text |
| `figure03_shap_summary/run_figure03.py` | Figure 3 | Main text |
| `figure04_two_way_pdp/run_figure04.py` | Figure 4 | Main text |
| `figure05_shap_dependence/run_figure05.py` | Figure 5 | Main text |
| `figureS3_correlation_matrix/run_figureS3.py` | Figure S3 | Supporting information |
| `figureS4_shap_rank_stability/run_figureS4.py` | Figure S4 | Supporting information |
| `figureS5_pdp_ice/run_figureS5.py` | Figure S5 | Supporting information |

---

## Data availability

The measured raw dataset is **not** distributed with this repository. Scripts
expect an Excel workbook at `data/Data.xlsx` with the column structure
described in [`data/README.md`](data/README.md). Data can be obtained from the
corresponding author on reasonable request

---

## Installation

Python 3.9 or newer is recommended.

```bash

pip install -r requirements.txt
```

---

## Usage

Run any figure script from the repository root:

```bash
python figure02_model_performance/run_figure02.py
python figure03_shap_summary/run_figure03.py
python figure04_two_way_pdp/run_figure04.py
python figure05_shap_dependence/run_figure05.py

python figureS3_correlation_matrix/run_figureS3.py
python figureS4_shap_rank_stability/run_figureS4.py
python figureS5_pdp_ice/run_figureS5.py
```

Each script writes a 600 dpi PNG to `outputs/figures/`, and prints the key
numerical results (model metrics, SHAP rankings) to the console.

Scripts are independent; each performs its own data loading, splitting and
model fitting, so they can be run in any order.

---

## Analysis workflow

1. **Preprocessing** (`src/common.py`). Records lacking an NOx measurement are
   removed, a `Date × Group` event identifier is constructed, continuous
   variables are linearly interpolated within each event block (with group- and
   global-median fallbacks), and operational variables are forward/backward
   filled within the event block (with a modal fallback).
2. **Feature screening.** CO₂ and CH₄ are excluded from the predictor set
   because of high pairwise Pearson correlation (|r| > 0.75) with retained
   variables (Figure S3). Occupant count is recoded as a binary occupancy flag.
3. **Partitioning.** A single 70 / 15 / 15 training / testing / validation
   split with `random_state = 34`.
4. **Model selection.** Five top tree-ensemble regressors (CatBoost, XGBoost,
   Random Forest, Gradient Boosting, Bagging) are tuned by randomised search
   with 10-fold cross-validation on the training set (Figure 2).
5. **Interpretation.** CatBoost native TreeSHAP values on the validation subset
   (Figures 3 and 5), partial dependence and ICE curves (Figures 4 and S5), and
   a repeated-split check on the stability of the SHAP ranking (Figure S4).

### Reproducibility

Minor numerical differences may still arise across library
versions and hardware, particularly in the randomised search.
Versions used during development are pinned in `requirements.txt`.

### Interpretation note

SHAP values and partial dependence describe the behaviour of the fitted model
rather than the underlying physical system. Associations reported here are
consistent with established indoor HONO formation pathways, but alternative
explanations, including unmeasured covariates and correlations among
predictors, cannot be ruled out on the basis of these analyses alone.

---

## Citation

If this code is useful in your work, please cite the associated manuscript.
Citation metadata is provided in [`CITATION.cff`](CITATION.cff) and will be
updated with the final DOI on publication.

## Contact

Questions about the code should be directed to Muhammad Azher Hassan
(azher@tju.edu.cn). Questions about the measurements or the study design
should be directed to the corresponding authors of the manuscript.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
