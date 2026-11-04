"""Module 2 - Prediction uncertainty.

Builds prediction intervals around the Module 1 XGBoost model using two
deliberately contrasting conformal methods from MAPIE:

  1. Split conformal prediction (SplitConformalRegressor): wraps the point
     model with near-constant-width intervals that guarantee marginal coverage.
  2. Conformalized quantile regression (ConformalizedQuantileRegressor): wraps
     XGBoost quantile models, producing adaptive-width intervals.

Both are calibrated on the chronologically-last slice of the training period
and evaluated on the 2025 test period. Prints coverage and width at 80% and
90% nominal levels, and saves a regression reliability diagram (nominal vs
empirical coverage) plus a width-adaptivity plot.

Usage:
    python src/module2_uncertainty.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor
from xgboost import XGBRegressor

from common import (
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    coverage,
    ensure_dirs,
    feature_columns,
    load_split,
    mae,
    mean_width,
    median_width,
)

CONFIDENCE_LEVELS = [0.8, 0.9]

QUANTILE_XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=7,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def fit_quantile_model(X, y, quantile: float) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile, **QUANTILE_XGB_PARAMS
    )
    model.fit(X, y)
    return model


def main() -> None:
    ensure_dirs()
    train = load_split("train")
    cal = load_split("calibration")
    test = load_split("test")
    feats = feature_columns(train)

    # float64 targets: MAPIE's conformity-score consistency check fails on
    # float32 inputs due to rounding at the last ulp.
    X_cal, y_cal = cal[feats], cal["y_delta_sec"].to_numpy(dtype=np.float64)
    X_test, y_test = test[feats], test["y_delta_sec"].to_numpy(dtype=np.float64)

    point_model = XGBRegressor()
    point_model.load_model(MODELS_DIR / "xgb_point.json")

    out = test[["snapshot_ts", "month", "train_id", "horizon_step", "obs_horizon_min",
                "last_known_delay_sec", "congestion_total_trains",
                "congestion_mean_link_delay", "y_delta_sec", "y_abs_delay_sec"]].copy()

    # ------------------------------------------------------------------
    # Method 1: split conformal around the point model
    # ------------------------------------------------------------------
    print("Conformalizing split conformal regressor (MAPIE) ...")
    scr = SplitConformalRegressor(
        estimator=point_model, confidence_level=CONFIDENCE_LEVELS, prefit=True
    )
    scr.conformalize(X_cal, y_cal)
    y_pred, intervals = scr.predict_interval(X_test)
    out["y_pred"] = y_pred
    for k, level in enumerate(CONFIDENCE_LEVELS):
        pct = int(level * 100)
        out[f"split_lower_{pct}"] = intervals[:, 0, k]
        out[f"split_upper_{pct}"] = intervals[:, 1, k]

    # ------------------------------------------------------------------
    # Method 2: conformalized quantile regression (adaptive widths)
    # ------------------------------------------------------------------
    X_train, y_train = train[feats], train["y_delta_sec"].to_numpy(dtype=np.float64)
    median_model = None
    for level in CONFIDENCE_LEVELS:

        lo_model = fit_quantile_model(X_train, y_train, lo_q)
        hi_model = fit_quantile_model(X_train, y_train, hi_q)
        if median_model is None:
            median_model = fit_quantile_model(X_train, y_train, 0.5)
            median_model.save_model(MODELS_DIR / "xgb_cqr_median.json")
        lo_model.save_model(MODELS_DIR / f"xgb_cqr_lo_{pct}.json")
        hi_model.save_model(MODELS_DIR / f"xgb_cqr_hi_{pct}.json")

        pct = int(level * 100)
        lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
        t0 = time.time()
        print(f"Training XGBoost quantile models for {pct}% intervals "
              f"(quantiles {lo_q:.3f}/{hi_q:.3f}/0.5) ...")
        lo_model = fit_quantile_model(X_train, y_train, lo_q)
        hi_model = fit_quantile_model(X_train, y_train, hi_q)
        if median_model is None:
            median_model = fit_quantile_model(X_train, y_train, 0.5)
        print(f"  done in {time.time() - t0:.0f}s")

        cqr = ConformalizedQuantileRegressor(
            estimator=[lo_model, hi_model, median_model],
            confidence_level=level,
            prefit=True,
        )
        cqr.conformalize(X_cal, y_cal)
        cqr_pred, cqr_intervals = cqr.predict_interval(X_test)
        out[f"cqr_lower_{pct}"] = cqr_intervals[:, 0, 0]
        out[f"cqr_upper_{pct}"] = cqr_intervals[:, 1, 0]
        if "cqr_median" not in out:
            out["cqr_median"] = cqr_pred

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    print("\n--- Module 2: prediction uncertainty (test period: 2025) ---")
    print(f"Point model MAE: {mae(y_test, out['y_pred']):.2f} s\n")
    header = f"{'method':<18}{'nominal':>9}{'coverage':>10}{'mean width':>12}{'median width':>14}"
    print(header)
    print("-" * len(header))
    for method in ("split", "cqr"):
        for level in CONFIDENCE_LEVELS:
            pct = int(level * 100)
            lo, hi = out[f"{method}_lower_{pct}"], out[f"{method}_upper_{pct}"]
            print(
                f"{method:<18}{level:>9.2f}{coverage(y_test, lo, hi):>10.3f}"
                f"{mean_width(lo, hi):>10.1f} s{median_width(lo, hi):>12.1f} s"
            )

    # ------------------------------------------------------------------
    # Reliability diagram: nominal vs empirical coverage
    # ------------------------------------------------------------------
    # For split conformal the whole curve is cheap: quantiles of calibration
    # conformity scores (absolute residuals) evaluated against test residuals.
    cal_scores = np.abs(y_cal - point_model.predict(X_cal))
    test_scores = np.abs(y_test - out["y_pred"].to_numpy())
    nominal_grid = np.arange(0.50, 0.96, 0.05)
    n_cal = len(cal_scores)
    empirical = []
    for level in nominal_grid:
        q = np.quantile(cal_scores, min((n_cal + 1) * level / n_cal, 1.0), method="higher")
        empirical.append(float(np.mean(test_scores <= q)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.45, 1.0], [0.45, 1.0], "k--", lw=1, label="perfect calibration")
    ax.plot(nominal_grid, empirical, "o-", label="split conformal (XGBoost)")
    cqr_cov = [
        coverage(y_test, out[f"cqr_lower_{int(l * 100)}"], out[f"cqr_upper_{int(l * 100)}"])
        for l in CONFIDENCE_LEVELS
    ]
    ax.plot(CONFIDENCE_LEVELS, cqr_cov, "s", ms=9, label="conformalized quantile regression")
    ax.set_xlabel("Nominal coverage (confidence level)")
    ax.set_ylabel("Empirical coverage on 2025 test data")
    ax.set_title("Reliability diagram: stated vs observed coverage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Width adaptivity: interval width vs actual absolute error (90% level)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    err_bins = np.quantile(test_scores, np.linspace(0, 1, 11))
    bin_idx = np.clip(np.digitize(test_scores, err_bins[1:-1]), 0, 9)
    for ax, method, name in (
        (axes[0], "split", "Split conformal (constant width)"),
        (axes[1], "cqr", "CQR (adaptive width)"),
    ):
        widths = (out[f"{method}_upper_90"] - out[f"{method}_lower_90"]).to_numpy()
        mean_w = [widths[bin_idx == b].mean() for b in range(10)]
        centers = [(err_bins[b] + err_bins[b + 1]) / 2 for b in range(10)]
        ax.plot(centers, mean_w, "o-")
        ax.set_title(name)
        ax.set_xlabel("Actual absolute error of point prediction (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean 90% interval width (s)")
    fig.suptitle("Do intervals widen when predictions are actually harder?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "width_adaptivity.png", dpi=150)
    plt.close(fig)

    out.to_parquet(PREDICTIONS_DIR / "test_interval_predictions.parquet", index=False)
    print(f"\nSaved interval predictions to {PREDICTIONS_DIR / 'test_interval_predictions.parquet'}")
    print(f"Saved figures: {FIGURES_DIR / 'reliability_diagram.png'}, "
          f"{FIGURES_DIR / 'width_adaptivity.png'}")


if __name__ == "__main__":
    main()
