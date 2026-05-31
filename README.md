# DengAI: weekly dengue case forecasting

I built this project to predict weekly dengue case counts in **San Juan (Puerto Rico)** and **Iquitos (Peru)** from weather, satellite vegetation indices, and epidemic history. It was completed for a **Data Exploration** course using the [DengAI](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) dataset (DrivenData / U.S. CDC).

**Full narrative:** [Predicting the Spread of Dengue Fever (DengAI) — Medium](https://medium.com/@sebastian.jozef.lukasik/predicting-the-spread-of-dengue-fever-dengai-fe5d0f7c8fef)

**Author:** [Sebastian Łukasik](https://github.com/SebaLukasik)

---

## Summary (for a quick read)

I implemented an end-to-end **time-series ML pipeline**: EDA → lag-based feature engineering → model comparison → hyperparameter tuning (RandomizedSearch, Optuna, GridSearch) → multi-week forecast horizons → **SHAP** explainability. **Gradient boosting (XGBoost)** beat a neural network and a zero-shot foundation model (Chronos-T5) when weather and lags were included. San Juan is harder (higher MAE, stronger outliers); Iquitos is more stable. The strongest predictors in the final models are **recent case momentum** (`cases_moving_avg`) and **lagged** weather, not same-week conditions alone.

---

## Problem

Dengue is mosquito-borne and climate-sensitive. Health agencies care about **weekly case forecasts** for hospital load and control campaigns. The competition provides separate weekly series for two cities with different ecology:

| City | Weeks | Period (approx.) | Typical pattern |
|------|-------|------------------|-----------------|
| **San Juan** | 936 | 1990–2008 | Autumn peak, large 1994 outbreak |
| **Iquitos** | 520 | 2000–2010 | Mid-year dip, surge around year boundary |

**Target:** `total_cases` per week. **Primary metric:** MAE (mean absolute error on raw case counts).

---

## Approach

### 1. Exploratory analysis

- Right-skewed case distributions (long epidemic tails).
- Different seasonal curves (week-of-year effects).
- Missing values mainly in NDVI (San Juan) and station sensors (Iquitos).
- Univariate **lag correlation** (0–15 weeks) to motivate which weather signals to lag (e.g. San Juan station temperature ~10–11 weeks; Iquitos humidity often lag 0).

### 2. Feature engineering

Custom features per week, including:

- `cases_moving_avg` — 4-week rolling mean of past cases (shifted by 1 week).
- `precip_sum_4w`, `temp_amplitude`, `heat_humidity_idx`, `temp_change_1w`.
- `week_sin` / `week_cos` for seasonality.
- Top **15** weather/NDVI variables each shifted by its **best lag** from correlation search (e.g. `station_avg_temp_c_lag10`).

### 3. Modeling protocol (avoid leakage)

- **Chronological 80% train / 20% test** — no random shuffle.
- Train on `log1p(total_cases)`, evaluate with `expm1` on predictions.
- Tuning: `TimeSeriesSplit(n_splits=3)` inside the training window.
- **SARIMA** excluded after poor results on this feature set.

### 4. Models compared

| Model | Role |
|-------|------|
| **XGBoost** | Main model; tuning + final plots + SHAP |
| **Random Forest** | Strong baseline; best hold-out MAE on Iquitos |
| **LightGBM** | Gradient boosting alternative |
| **MLP (Keras)** | Neural baseline on scaled tabular features |
| **Amazon Chronos-T5** | Zero-shot forecast on case history only (no weather) |

**Design choice:** After Optuna, Random Forest is ~0.12 MAE better than XGBoost on Iquitos hold-out. I still use **tuned XGBoost for both cities** in forecast plots, horizon tests, and SHAP so one pipeline is documented end-to-end. See the Medium article for rationale.

---

## Results

### Phase 1 — RandomizedSearchCV (screening, test MAE)

| San Juan | MAE | Iquitos | MAE |
|----------|-----|---------|-----|
| XGBoost | **8.45** | Random Forest | **5.05** |
| Random Forest | 9.04 | XGBoost | 5.19 |
| LightGBM | 10.23 | LightGBM | 5.22 |

### Phase 2 — Optuna (50 trials per algorithm, test MAE)

| City | Best algorithm | MAE |
|------|----------------|-----|
| San Juan | XGBoost | **8.84** |
| Iquitos | Random Forest | **5.01** (XGBoost: 5.13) |

### Phase 3 — XGBoost tuning methods (test MAE)

| San Juan | MAE | Iquitos | MAE |
|----------|-----|---------|-----|
| Optuna | **8.72** | Optuna / RandomSearch | **5.09** |
| GridSearch | 8.84 | GridSearch | 5.15 |
| RandomSearch | 8.84 | — | — |

### Final XGBoost (hold-out, used in plots & SHAP)

| City | MAE (approx.) | Notes |
|------|---------------|--------|
| San Juan | ~8.8 | Harder city; 1994 spike |
| Iquitos | ~5.1–5.2 | Lower variance |

### Alternatives (sanity check)

| Model | San Juan MAE | Iquitos MAE |
|-------|--------------|-------------|
| MLP | 21.50 | 7.23 |
| Chronos-T5 (no weather) | 27.82 | 8.60 |

Chronos and MLP miss large outbreak peaks when they cannot use lagged weather and epidemic momentum.

### Forecast horizons (XGBoost + Optuna params, shifted target)

| Horizon | San Juan | Iquitos |
|---------|----------|---------|
| +1 week | 11.01 | 5.38 |
| +4 weeks | 13.27 | 6.07 |
| +8 weeks | 16.59 | 7.01 |
| +12 weeks | 16.93 | 7.31 |

Error grows with horizon; +8/+12 weeks on San Juan approach ~2× the one-step model error — long horizons would need a dedicated multi-step or probabilistic setup.

### Weather ablation (best tuned model per city)

- **Iquitos:** removing weather columns **worsens** MAE (~+0.2) → weather adds signal.
- **San Juan:** removing weather **improves** MAE slightly (~−0.6) → possible redundancy with `cases_moving_avg` and lagged temperature features.

### Explainability (SHAP, XGBoost)

Global drivers align with epidemiology:

1. **`cases_moving_avg`** — autoregressive epidemic momentum.
2. **Lagged weather / NDVI** — not current-week values only.
3. **Seasonality** (`weekofyear`, cyclic features).

Local waterfall plots on epidemic peaks show combined effects (season + humidity/temperature + high recent cases).

---

## What this project demonstrates

- Time-series aware validation (no shuffle, `TimeSeriesSplit`).
- Feature engineering tied to domain lags (mosquito life cycle).
- Practical model selection beyond a single algorithm (boosting vs RF vs DL vs foundation TS model).
- Hyperparameter search at scale (RandomizedSearch + Optuna + GridSearch).
- Communicating trade-offs (best hold-out model vs consistent reporting model).
- Interpretability with SHAP, not only accuracy tables.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Optuna, matplotlib, seaborn, Plotly, SHAP; optional TensorFlow (MLP), PyTorch + chronos-forecasting.

---

## Repository layout

```
data/                          # Training CSVs (see data/README.md)
figures/                       # Exported plots (report / Medium)
notebooks/
  dengue_forecasting.ipynb     # Full analysis pipeline
requirements.txt
```

Key figures:

| File | Description |
|------|-------------|
| `04_importance_sj.png`, `05_importance_iq.png` | XGBoost top-10 feature importance |
| `beeshap_*.png` | SHAP summary (beeswarm-style) |
| `waterfall_*.png` | SHAP waterfall at epidemic peaks |
| `chronos_*.png` | Chronos zero-shot vs actual |

---

## Setup & run

**Requirements:** Python 3.10+

```bash
git clone https://github.com/SebaLukasik/dengai-dengue-forecast.git
cd dengai-dengue-forecast
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
jupyter notebook notebooks/dengue_forecasting.ipynb
```

- Data are loaded from `data/` via paths in the notebook (`../data` when the kernel cwd is `notebooks/`).
- Figures save to `figures/` when you run the importance-plot cell.
- **Optional sections:** MLP needs TensorFlow; Chronos needs `torch` + `chronos-forecasting` (downloads weights on first run).
- On **macOS**, the notebook uses `N_JOBS = 1` for parallel tree fitting in Jupyter to avoid worker crashes.

---

## Limitations & possible extensions

- Single chronological 80/20 split (no rolling-origin backtest).
- No formal statistical test on MAE differences between models.
- Chronos evaluated only on univariate case history.
- Production use would need external validation, uncertainty intervals, and public-health review.

**Next steps I would try next:** rolling-window CV, per-horizon models, stronger feature selection for San Juan weather redundancy, exported sklearn pipeline (pickle/ONNX).

---

## References

- [DengAI – Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
- Chen, T. & Guestrin, C. — XGBoost (KDD 2016)
- Akiba, T. et al. — Optuna (KDD 2019)
- Lundberg, S. & Lee, S.-I. — SHAP (NeurIPS 2017)
- Chronos — pretrained time series models (Amazon / Hugging Face)

---

*Academic course project. Not validated for operational outbreak forecasting.*
