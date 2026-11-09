# Running the Pipeline

A step-by-step guide to setting up the environment and executing each module of the train-delay Trustworthy AI pipeline — from a fresh clone to full explainability results.

> [!NOTE]
> All commands assume you are in a **WSL (Ubuntu) bash shell** with the project root as your working directory:
> ```bash
> cd /home/shiva/train_delay_experiment
> ```

---

## 1. Create the Virtual Environment

Python 3.14 is used. Create and activate a venv, then install all pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> [!IMPORTANT]
> `shap` is **not** in `requirements.txt` (it was added later for Module 5). Install it once after the base install:
> ```bash
> pip install shap
> ```

After this step your environment is fully ready for all modules.

---

## 2. Download and Prepare the Data

These two scripts only need to be run **once**. They fetch the raw dataset from Hugging Face and produce the chronological train / calibration / test parquet files that all modules read.

```bash
python src/download_data.py   # ~7.3 GB download — RIDE Gold Lite from HuggingFace
python src/prepare_data.py    # builds data/prepared/{train,calibration,test}.parquet
```

**Why:** The pipeline uses a fixed forward-in-time benchmark (training snapshots from 2023–2024, test snapshots from 2025). `prepare_data.py` performs the chronological split and writes interpretable, physically-unitised feature columns so that students work with real delay seconds and minutes rather than raw normalised values.

---

## 3. Running Modules 1–4 (Full Pipeline)

Run these **in order** — each module depends on artifacts saved by the one before it.

### Module 1 — Prediction Accuracy

```bash
python src/module1_accuracy.py
```

**What it does:** Trains an XGBoost point-prediction model on the 2023–2024 training split and evaluates Mean Absolute Error (MAE) on the 2025 test split. Compares against the naive *translation baseline* (predict that delay stays exactly as it is now).

**Why we run it:** A low average error can mask regime-specific failures. Module 1 prints MAE breakdowns by prediction horizon (how far ahead we're predicting) and by delay-change bin, so we can see exactly where the model succeeds and where it doesn't — the foundation for everything that follows.

**Artifacts saved:**
- `data/models/xgb_point.json` — the trained point model (used by all later modules)
- `data/predictions/test_point_predictions.parquet`

---

### Module 2 — Prediction Uncertainty (Conformal Intervals)

```bash
python src/module2_uncertainty.py
```

**What it does:** Wraps the Module 1 model with two contrasting conformal prediction methods from MAPIE:
1. **Split Conformal** (`SplitConformalRegressor`) — near-constant-width intervals with a marginal coverage guarantee.
2. **Conformalized Quantile Regression / CQR** (`ConformalizedQuantileRegressor`) — adaptive-width intervals that widen in uncertain conditions and narrow in predictable ones.

Both are calibrated on the chronologically-last slice of training data and evaluated on the 2025 test period. Prints coverage and interval width at 80% and 90% nominal levels, and saves a reliability diagram.

**Why we run it:** Point predictions alone are not enough for infrastructure decisions. We need to know *how uncertain* each prediction is. CQR's adaptive intervals are what Module 5 explains with SHAP — so this module must run first to produce the interval predictions.

**Artifacts saved:**
- `data/models/xgb_cqr_lo_90.json` / `xgb_cqr_hi_90.json` *(quantile models — also cached by Module 5 if missing)*
- `data/predictions/test_interval_predictions.parquet`
- `reports/figures/reliability_diagram.png`, `width_adaptivity.png`

---

### Module 3 — Temporal Drift & Conditional Reliability

```bash
python src/module3_drift.py
```

**What it does:** Slices the Module 2 interval predictions by operating condition to test whether the marginal coverage guarantee holds *conditionally*:
- Prediction horizon (0–5 min → 45+ min ahead)
- Operating regime (on-time / slightly / heavily delayed trains)
- Local network congestion
- Calendar month of the 2025 test period (temporal drift)

**Why we run it:** A model can appear well-calibrated on average but systematically fail under specific conditions (long horizons, heavy delays, high congestion). Module 3 makes these conditional failures visible — the "drift audit" that motivates the scenario analysis in Module 4.

**Artifacts saved:**
- `reports/figures/drift_*.png` (one per breakdown slice)

---

### Module 4 — Engineering Decision Scenarios

```bash
python src/module4_scenarios.py
```

**What it does:** Mines the 2025 test predictions for real instances of four decision-critical situations:
- **Scenario A** — Accurate point estimate but top-decile uncertainty (wide interval)
- **Scenario B** — Narrow interval but the true value falls outside it
- **Scenario C** — Fine on average, but interval breaks down during disruptions (high delay + high congestion)
- **Scenario D** — Same train at same snapshot: comparing near-horizon vs far-horizon predictions

Prints a scenario table and saves it to CSV.

**Why we run it:** This converts abstract coverage statistics into concrete, human-readable decision moments. Scenario A and C rows are the specific test instances that Module 5 explains at the local (single-prediction) level.

**Artifacts saved:**
- `reports/decision_scenarios.csv`

---

## 4. Running Module 5 — Uncertainty-Aware Explainability (SHAP)

Module 5 uses exact **TreeSHAP** to decompose both the point prediction and the 90% CQR interval *width* feature-by-feature. It answers: *what drives expected delay?* vs *what drives prediction uncertainty?* — and how do those drivers change during disruptions?

### Happy Path — Models Already in Cache

If you have already run Module 2 (or a previous Module 5 run), the quantile models are cached at:
```
data/models/xgb_cqr_lo_90.json
data/models/xgb_cqr_hi_90.json
```

Just run:
```bash
python src/module5_explainability.py
```

Module 5 detects the cached files and skips straight to TreeSHAP. Expected runtime: **~2–3 minutes** (2,000 test snapshots × 3 models).

---

### Cold Start — Models Not in Cache

If the quantile models are missing (e.g. fresh clone, or `data/models/` was cleared), Module 5 will **fit them from scratch** on the full training set before computing SHAP. This is the slow path.

> [!WARNING]
> Fitting both quantile models on the full training set uses ~4 GB RAM and takes **~8–10 minutes** on a typical laptop. The models are saved immediately after fitting, so subsequent runs are fast.

```bash
python src/module5_explainability.py
```

You will see:
```
Cached quantile models not found. Fitting 90% quantile models once ...
Saved quantile models to data/models/xgb_cqr_lo_90.json and data/models/xgb_cqr_hi_90.json
Computing exact TreeSHAP values on 2,000 test snapshots (1000 normal, 1000 disrupted) ...
```

**To avoid the cold start in the future**, you can run Module 2 first which fits and caches the same quantile models as part of its own pipeline:
```bash
python src/module2_uncertainty.py   # caches quantile models
python src/module5_explainability.py  # now uses cache
```

---

### Module 5 Artifacts

| File | Description |
|---|---|
| `reports/shap_feature_importance.csv` | Full feature table: point |SHAP|, width |SHAP|, normal vs disrupted breakdown, amplification ratio |
| `reports/figures/shap_point_vs_width.png` | Side-by-side bar chart: accuracy drivers vs uncertainty drivers, normal vs disrupted regime |
| `reports/figures/shap_scenario_explanations.png` | Local waterfall-style bars for Scenario A (accurate but uncertain) and Scenario C (disruption interval breakdown) |

---

## Quick Reference — Full Run Order

```bash
# 1. Environment setup (once)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install shap

# 2. Data (once)
python src/download_data.py
python src/prepare_data.py

# 3. Pipeline (in order)
python src/module1_accuracy.py
python src/module2_uncertainty.py
python src/module3_drift.py
python src/module4_scenarios.py
python src/module5_explainability.py
```

