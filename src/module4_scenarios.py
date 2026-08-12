"""Module 4 - Engineering decision scenarios.

Mines the 2025 test predictions for real instances of the four decision
situations described in the proposal:

  A. Accurate but highly uncertain: small point error, top-decile interval width.
  B. Narrow but wrong: bottom-decile interval width, true value outside interval.
  C. Fine on average, unreliable during disruption: coverage of heavily delayed
     high-congestion trains vs the overall aggregate.
  D. Short vs long horizon: the same train at the same snapshot, first vs last
     predicted event.

Prints a scenario table with everything a student needs to reason about each
case and saves it to CSV.

Usage:
    python src/module4_scenarios.py
"""

import numpy as np
import pandas as pd

from common import PREDICTIONS_DIR, REPORTS_DIR, coverage, ensure_dirs, mae, print_table

PCT = 90  # use the 90% intervals

DISPLAY_COLS = [
    "scenario", "snapshot_ts", "train_id", "obs_horizon_min", "last_known_delay_sec",
    "y_pred", "lower", "upper", "width", "y_delta_sec", "abs_error", "inside_interval",
]


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["obs_horizon_min"] = out["obs_horizon_min"].round(1)
    for c in ("y_pred", "lower", "upper", "width", "y_delta_sec", "abs_error",
              "last_known_delay_sec"):
        out[c] = out[c].round(0)
    return out[DISPLAY_COLS]


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PREDICTIONS_DIR / "test_interval_predictions.parquet")
    df["lower"] = df[f"cqr_lower_{PCT}"]
    df["upper"] = df[f"cqr_upper_{PCT}"]
    df["width"] = df["upper"] - df["lower"]
    df["abs_error"] = (df["y_delta_sec"] - df["y_pred"]).abs()
    df["inside_interval"] = (df["y_delta_sec"] >= df["lower"]) & (df["y_delta_sec"] <= df["upper"])

    wide_cut = df["width"].quantile(0.9)
    narrow_cut = df["width"].quantile(0.1)
    rng = np.random.default_rng(0)

    print(f"--- Module 4: decision scenarios mined from the 2025 test set "
          f"(CQR {PCT}% intervals) ---")

    # Scenario A: accurate point prediction, very uncertain interval.
    a_pool = df[(df["abs_error"] < 30) & (df["width"] >= wide_cut)]
    a = a_pool.iloc[rng.choice(len(a_pool), 3, replace=False)].assign(scenario="A: accurate but uncertain")
    print(f"\nScenario A pool: {len(a_pool):,} rows "
          f"(point error < 30s but width in top decile, >= {wide_cut:.0f}s)")

    # Scenario B: narrow interval that misses the truth.
    b_pool = df[(df["width"] <= narrow_cut) & ~df["inside_interval"]]
    b = b_pool.iloc[rng.choice(len(b_pool), 3, replace=False)].assign(scenario="B: narrow but wrong")
    print(f"Scenario B pool: {len(b_pool):,} rows "
          f"(width in bottom decile, <= {narrow_cut:.0f}s, true value outside interval)")

    # Scenario C: good aggregate reliability, poor during disruptions.
    disrupted = df[(df["last_known_delay_sec"] > 300)
                   & (df["congestion_mean_link_delay"] > df["congestion_mean_link_delay"].quantile(0.9))]
    cov_all = coverage(df["y_delta_sec"], df["lower"], df["upper"])
    cov_dis = coverage(disrupted["y_delta_sec"], disrupted["lower"], disrupted["upper"])
    print(f"\nScenario C: overall coverage = {cov_all:.3f} vs disrupted-operations coverage "
          f"= {cov_dis:.3f}  ({len(disrupted):,} disrupted rows: last known delay > 5min "
          f"and top-decile local congestion delay)")
    print(f"           overall MAE = {mae(df['y_delta_sec'], df['y_pred']):.1f}s vs "
          f"disrupted MAE = {mae(disrupted['y_delta_sec'], disrupted['y_pred']):.1f}s")
    c_pool = disrupted[~disrupted["inside_interval"]]
    c = c_pool.iloc[rng.choice(len(c_pool), 3, replace=False)].assign(scenario="C: fails in disruption")

    # Scenario D: same train and snapshot, shortest vs longest horizon.
    keys = ["snapshot_ts", "train_id"]
    grouped = df.sort_values("horizon_step").groupby(keys, observed=True)
    firsts = grouped.head(1).set_index(keys)
    lasts = grouped.tail(1).set_index(keys)
    both = firsts.join(lasts, lsuffix="_short", rsuffix="_long", how="inner")
    both = both[both["horizon_step_long"] - both["horizon_step_short"] >= 10]
    growth = both["width_long"] / both["width_short"].clip(lower=1)
    d_keys = growth.sort_values(ascending=False).index[:3]
    d_rows = []
    for key in d_keys:
        for suffix, label in (("_short", "D: short horizon"), ("_long", "D: long horizon")):
            row = both.loc[key, [c + suffix for c in DISPLAY_COLS[3:]]].copy()
            row.index = [c.removesuffix(suffix) for c in row.index]
            row["snapshot_ts"], row["train_id"] = key
            row["scenario"] = label
            d_rows.append(row)
    d = pd.DataFrame(d_rows)
    print(f"Scenario D: {len(both):,} train instances with both a short- and long-horizon "
          "prediction; showing the 3 with the largest interval growth")

    scenarios = pd.concat([fmt(a), fmt(b), fmt(c), fmt(d)], ignore_index=True)
    print_table(scenarios, "Decision scenarios (delay change in seconds)", "{:.0f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "decision_scenarios.csv"
    scenarios.to_csv(out_path, index=False)
    print(f"\nSaved scenario table to {out_path}")


if __name__ == "__main__":
    main()
