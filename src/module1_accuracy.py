"""Module 1 - Prediction accuracy.

Trains the XGBoost point-prediction model on the prepared training split and
evaluates Mean Absolute Error on the chronologically-later test split (2025),
against the naive Translation baseline (delay stays at its last known value,
i.e. predicted delay change = 0).

Prints overall MAE plus breakdowns by prediction horizon and by delay-change
bin - the tables that show how a low average error can hide regime-specific
failures.

Usage:
    python src/module1_accuracy.py
"""

import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common import (
    DELTA_BIN_EDGES,
    DELTA_BIN_LABELS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    ensure_dirs,
    feature_columns,
    horizon_bin,
    load_split,
    mae,
    print_table,
)

XGB_PARAMS = dict(
    # Absolute-error objective: the evaluation metric is MAE and delay changes
    # are heavy-tailed, so optimizing squared error would chase outliers.
    objective="reg:absoluteerror",
    n_estimators=400,
    learning_rate=0.08,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def main() -> None:
    ensure_dirs()
    train = load_split("train")
    test = load_split("test")
    feats = feature_columns(train)
    print(f"Training rows: {len(train):,}  |  test rows: {len(test):,}  |  features: {len(feats)}")

    model = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    model.fit(train[feats], train["y_delta_sec"])
    print(f"XGBoost trained in {time.time() - t0:.0f}s")
    model.save_model(MODELS_DIR / "xgb_point.json")

    y_true = test["y_delta_sec"].to_numpy()
    y_pred = model.predict(test[feats])

    print("\n--- Module 1: prediction accuracy (test period: 2025) ---")
    print(f"MAE  XGBoost point model : {mae(y_true, y_pred):8.2f} s")
    print(f"MAE  Translation baseline: {mae(y_true, np.zeros_like(y_true)):8.2f} s"
          "   (predict 'delay stays the same')")

    df = pd.DataFrame(
        {
            "y_delta_sec": y_true,
            "y_pred": y_pred,
            "horizon_bin": horizon_bin(test["obs_horizon_min"]),
            "delta_bin": pd.cut(y_true, bins=DELTA_BIN_EDGES, labels=DELTA_BIN_LABELS),
        }
    )

    by_horizon = df.groupby("horizon_bin", observed=True).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mae_xgb_sec": mae(g["y_delta_sec"], g["y_pred"]),
                "mae_baseline_sec": mae(g["y_delta_sec"], np.zeros(len(g))),
            }
        ),
        include_groups=False,
    ).reset_index()
    print_table(by_horizon, "MAE by prediction horizon (minutes ahead)", "{:.2f}")

    by_delta = df.groupby("delta_bin", observed=True).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mae_xgb_sec": mae(g["y_delta_sec"], g["y_pred"]),
                "mae_baseline_sec": mae(g["y_delta_sec"], np.zeros(len(g))),
            }
        ),
        include_groups=False,
    ).reset_index()
    print_table(by_delta, "MAE by delay-change bin (how much the delay actually changed)", "{:.2f}")

    out = test[["snapshot_ts", "month", "train_id", "horizon_step", "obs_horizon_min",
                "last_known_delay_sec", "y_delta_sec", "y_abs_delay_sec"]].copy()
    out["y_pred"] = y_pred
    out["congestion_total_trains"] = test["congestion_total_trains"]
    out["congestion_mean_link_delay"] = test["congestion_mean_link_delay"]
    out.to_parquet(PREDICTIONS_DIR / "test_point_predictions.parquet", index=False)
    print(f"\nSaved point predictions to {PREDICTIONS_DIR / 'test_point_predictions.parquet'}")


if __name__ == "__main__":
    main()
