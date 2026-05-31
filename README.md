# DengAI: weekly dengue case forecasting

Course project (Data Exploration): predict weekly dengue cases in **San Juan, Puerto Rico** and **Iquitos, Peru** using weather, vegetation indices, and lagged features. Evaluation metric: **MAE** on case counts.

**Write-up:** [Predicting the Spread of Dengue Fever (DengAI)](https://medium.com/@sebastian.jozef.lukasik/predicting-the-spread-of-dengue-fever-dengai-fe5d0f7c8fef)

**Authors:** Sebastian Józef Łukasik & *[add teammate name]*

---

## What we did

- EDA for two cities (different seasonality and outbreak patterns)
- Lag search (0–15 weeks) and feature engineering (`cases_moving_avg`, cyclic week features)
- Compared **XGBoost**, **Random Forest**, **LightGBM**, a small **MLP**, and **Amazon Chronos-T5** (zero-shot, cases only)
- Tuning with **RandomizedSearchCV**, **Optuna**, and **GridSearch** (XGBoost)
- Forecast horizons +1 / +4 / +8 / +12 weeks
- **SHAP** for the tuned XGBoost models

**Note:** On the 80/20 hold-out, Random Forest was slightly better for Iquitos (~5.01 MAE vs ~5.13 for XGBoost). Plots, horizons, and SHAP in the notebook use **tuned XGBoost for both cities** so the pipeline stays consistent.

### Hold-out MAE (after Optuna, approximate)

| City | XGBoost | Random Forest | LightGBM |
|------|---------|---------------|----------|
| San Juan | **8.84** | 9.40 | 10.04 |
| Iquitos | 5.13 | **5.01** | 5.10 |

---

## Repository layout

```
data/                 # CSV files (train features + labels)
figures/              # Plots for report / Medium
notebooks/
  dengue_forecasting.ipynb
requirements.txt
```

---

## Setup

Python 3.10+ recommended.

```bash
git clone https://github.com/YOUR_USERNAME/dengai-dengue-forecast.git
cd dengai-dengue-forecast
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Heavy optional parts: `tensorflow` (MLP), `torch` + `chronos-forecasting` (Chronos). The notebook runs without them if you skip those sections.

---

## Run

1. Open `notebooks/dengue_forecasting.ipynb`.
2. Run from the top. Data paths assume CSV files in `data/` (notebook may use `../data/` depending on kernel cwd — adjust one path if needed).

On macOS, the notebook sets `N_JOBS = 1` for XGBoost/LightGBM to avoid joblib worker crashes in Jupyter.

---

## Figures

| File | Content |
|------|---------|
| `04_importance_sj.png` | Top-10 features, XGBoost, San Juan |
| `05_importance_iq.png` | Top-10 features, XGBoost, Iquitos |
| `beeshap_*.png` | SHAP summary |
| `waterfall_*.png` | SHAP waterfall (epidemic peaks) |
| `chronos_*.png` | Chronos vs actual (zero-shot) |

---

## References

- [DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
- Chen et al. – Chronos (time series foundation models)
- Lundberg & Lee – SHAP
- Chen & Guestrin – XGBoost

---

*University course project — not production epidemiological forecasting.*
