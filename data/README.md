# Data

The measured raw dataset is not distributed in this repository. And are available 
from the corresponding author on reasonable request.

## Expected columns

The first row must contain these column headers, with these exact units.
One row per 1-minute averaging interval.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Date` | integer | – | Experiment/day identifier |
| `HONO (ppb)` | float | ppb | Nitrous acid (target variable) |
| `NOx (ppb)` | float | ppb | Nitrogen oxides |
| `O3 (ppb)` | float | ppb | Ozone |
| `RH (%)` | float | % | Relative humidity |
| `NH3 (ppb)` | float | ppb | Ammonia |
| `N2O (ppm)` | float | ppm | Nitrous oxide |
| `CO2 (ppm)` | float | ppm | Carbon dioxide |
| `CH4 (ppm)` | float | ppm | Methane |
| `TVOCs (ppb)` | float | ppb | Total volatile organic compounds |
| `Combustion` | 0/1 | – | Combustion source active |
| `Occupants` | integer | – | Number of occupants present |
| `Ventilation` | 0/1 | – | Mechanical ventilation active |
| `Group` | text | – | Experiment group label |

Notes:

* `Occupants` is recoded internally as a binary `Occupancy` flag
  (0 = unoccupied, 1 = one or more occupants present).
* `CO2 (ppm)` and `CH4 (ppm)` are used for correlation screening only
  (Figure S3) and are excluded from the predictor set of the main model.
* Rows without an `NOx (ppb)` value are removed during preprocessing.
* Remaining gaps are filled as described in `src/common.py` and in the README.

## Availability

Data supporting the findings of this study are available from the
corresponding author on reasonable request.