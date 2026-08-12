"""Prepare partially-prepared train / calibration / test datasets from RIDE Gold Lite.

Reads the dense tabular arrays (452 normalized features per train instance,
15 future delay-delta targets), selects an interpretable subset of features,
inverts the RIDE normalization so all quantities are in physical units
(seconds, counts, meters), and expands each instance into one row per
(instance, future event) pair.

Chronological strictness:
  - RIDE Gold Lite already separates training snapshots (2023-2024) from test
    snapshots (2025).
  - Within the training period, the calibration set for conformal prediction
    is taken from the LAST 10% of training snapshots in temporal order, so
    calibration never sees data older than the training data it follows.

Output (data/prepared/):
  train.parquet, calibration.parquet, test.parquet, column_manifest.csv

Usage:
    python src/prepare_data.py [--n-train 120000] [--n-cal 30000] [--n-test 200000]
"""

import argparse

import numpy as np
import pandas as pd

from common import (
    CORE_DIR,
    PREPARED_DIR,
    RANDOM_SEED,
    TABULAR_DIR,
    denormalize,
    ensure_dirs,
    load_metadata_array,
    load_normalization,
    load_scheme,
)

N_FUTURE = 15

LINK_COLS = [
    f"link{i}_{k}"
    for i in range(10)
    for k in ("distance", "nb_of_trains", "avg_delay", "is_placeholder")
]
WEATHER_COLS = [
    "weather_temperature_2m",
    "weather_rain",
    "weather_snowfall",
    "weather_relative_humidity_2m",
    "weather_wind_speed_10m",
]
CYCLICAL_COLS = [
    f"snapshot_{unit}_{fn}_{h}"
    for unit in ("hour", "year")
    for fn in ("sin", "cos")
    for h in (1, 2, 4)
]
RELATION_COLS = [
    f"train_relation_type_{t}"
    for t in ("CHARTER", "EURST", "EXTRA", "IC", "ICE", "INT", "L", "P", "S", "TGV", "THAL", "nan")
]
DOW_COLS = [
    f"snapshot_day_of_week_{d}"
    for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
]
PAST_DELAY_COLS = [f"past_delay_sec_{j}" for j in range(1, 16)]
PAST_PLANNED_COLS = [f"past_planned_delta_{j}" for j in range(1, 16)]

SHARED_COLS = (
    ["operator_is_sncb_nmbs"] + LINK_COLS + WEATHER_COLS + CYCLICAL_COLS
    + RELATION_COLS + DOW_COLS + PAST_DELAY_COLS + PAST_PLANNED_COLS
)


def extract_shared_features(x: np.ndarray, col_index: dict, normalization: dict) -> pd.DataFrame:
    """Denormalized shared (per-instance) features as float32."""
    data = {}
    for col in SHARED_COLS:
        data[col] = denormalize(x[:, col_index[col]], col, normalization)
    df = pd.DataFrame(data)

    # Congestion summaries over the local network links (placeholder links excluded).
    nb = df[[f"link{i}_nb_of_trains" for i in range(10)]].to_numpy()
    dl = df[[f"link{i}_avg_delay" for i in range(10)]].to_numpy()
    real = df[[f"link{i}_is_placeholder" for i in range(10)]].to_numpy() < 0.5
    n_real = np.maximum(real.sum(axis=1), 1)
    df["congestion_total_trains"] = (nb * real).sum(axis=1).astype(np.float32)
    df["congestion_mean_link_delay"] = ((dl * real).sum(axis=1) / n_real).astype(np.float32)
    return df


def expand_instances(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    md: pd.DataFrame,
    col_index: dict,
    normalization: dict,
    eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand instance-level arrays into one row per (instance, future event j)."""
    shared = extract_shared_features(x, col_index, normalization)
    shared["snapshot_ts"] = md["snapshot_ts"].to_numpy()
    shared["train_id"] = md["train_id"].astype(str).to_numpy()
    # Best available last-known delay proxy; overridden by the exact eval-table
    # value for the test split below.
    shared["last_known_delay_sec"] = shared["past_delay_sec_1"]

    blocks = []
    for j in range(1, N_FUTURE + 1):
        valid = y_mask[:, j - 1] > 0.5
        target = denormalize(y[:, j - 1], f"future_delay_delta_{j}", normalization)
        if eval_df is not None:
            valid &= eval_df[f"future_delay_{j}"].notna().to_numpy()

        block = shared.loc[valid].copy()
        block["horizon_step"] = np.int8(j)
        block["planned_horizon_sec"] = denormalize(
            x[valid, col_index[f"future_planned_delta_{j}"]],
            f"future_planned_delta_{j}",
            normalization,
        )
        for et, name in (("A", "arrival"), ("D", "departure"), ("P", "passage")):
            block[f"future_is_{name}"] = x[valid, col_index[f"future_event_type_{j}_{et}"]].astype(
                np.float32
            )
        block["y_delta_sec"] = target[valid]

        if eval_df is not None:
            ev = eval_df.loc[valid]
            block["last_known_delay_sec"] = ev["last_known_delay"].to_numpy(dtype=np.float32)
            block["y_abs_delay_sec"] = ev[f"future_delay_{j}"].to_numpy(dtype=np.float32)
            obs_horizon = (
                ev[f"future_obs_ts_{j}"] - ev["ts"]
            ).dt.total_seconds().to_numpy() / 60.0
            block["obs_horizon_min"] = obs_horizon.astype(np.float32)
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    out["month"] = out["snapshot_ts"].dt.strftime("%Y-%m")
    return out


def write_manifest() -> None:
    rows = [
        ("operator_is_sncb_nmbs", "feature", "1 if the train is operated by SNCB/NMBS", "binary"),
        ("link[0-9]_distance", "feature", "Distance of nearby rail segment from the train", "meters"),
        ("link[0-9]_nb_of_trains", "feature", "Number of trains currently on that segment (congestion)", "count"),
        ("link[0-9]_avg_delay", "feature", "Average delay of trains on that segment", "seconds"),
        ("link[0-9]_is_placeholder", "feature", "1 if the segment slot is unused (fewer than 10 nearby segments)", "binary"),
        ("congestion_total_trains", "feature", "Total trains over all real nearby segments (network congestion)", "count"),
        ("congestion_mean_link_delay", "feature", "Mean of segment average delays (local disruption level)", "seconds"),
        ("weather_*", "feature", "Weather at the train location at snapshot time", "C, mm, %, km/h"),
        ("snapshot_hour_sin/cos_*", "feature", "Cyclical encoding of time of day", "unitless"),
        ("snapshot_year_sin/cos_*", "feature", "Cyclical encoding of day of year (season)", "unitless"),
        ("train_relation_type_*", "feature", "One-hot train service category (IC, L, S, ...)", "binary"),
        ("snapshot_day_of_week_*", "feature", "One-hot day of week", "binary"),
        ("past_delay_sec_[1-15]", "feature", "Delay at the 15 most recent observed events (1 = most recent)", "seconds"),
        ("past_planned_delta_[1-15]", "feature", "Scheduled time of past events relative to the snapshot (negative = in the past)", "seconds"),
        ("horizon_step", "feature", "Index of the future scheduled event being predicted (1-15)", "index"),
        ("planned_horizon_sec", "feature", "Scheduled time between snapshot and the future event", "seconds"),
        ("future_is_arrival/departure/passage", "feature", "Type of the future scheduled event", "binary"),
        ("y_delta_sec", "target", "Change in delay at the future event relative to last known delay", "seconds"),
        ("snapshot_ts", "metadata", "Timestamp of the prediction snapshot", "datetime"),
        ("month", "metadata", "Snapshot month (YYYY-MM), for drift analysis", "string"),
        ("train_id", "metadata", "Train identifier", "string"),
        ("last_known_delay_sec", "metadata", "Delay of the train at prediction time", "seconds"),
        ("y_abs_delay_sec", "metadata", "Observed absolute delay at the future event (test only)", "seconds"),
        ("obs_horizon_min", "metadata", "Observed time between snapshot and future event (test only)", "minutes"),
    ]
    pd.DataFrame(rows, columns=["column", "role", "description", "units"]).to_csv(
        PREPARED_DIR / "column_manifest.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=120_000, help="training instances to sample")
    parser.add_argument("--n-cal", type=int, default=30_000, help="calibration instances to sample")
    parser.add_argument("--n-test", type=int, default=200_000, help="test instances to sample")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_dirs()
    rng = np.random.default_rng(args.seed)
    scheme = load_scheme()
    normalization = load_normalization()
    col_index = {c: i for i, c in enumerate(scheme["x_columns"])}

    # ------------------------------------------------------------------
    # Training period (2023-2024): chronological train / calibration split
    # ------------------------------------------------------------------
    print("Loading training metadata ...")
    md_train = load_metadata_array(TABULAR_DIR / "train" / "md.npy")
    snapshots = np.sort(md_train["snapshot_ts"].unique())
    cutoff = snapshots[int(len(snapshots) * 0.9)]
    print(f"  {len(md_train):,} instances over {len(snapshots):,} snapshots")
    print(f"  calibration cutoff (last 10% of snapshots): {cutoff}")

    is_cal = (md_train["snapshot_ts"] >= cutoff).to_numpy()
    train_pool = np.flatnonzero(~is_cal)
    cal_pool = np.flatnonzero(is_cal)
    train_idx = np.sort(rng.choice(train_pool, size=min(args.n_train, len(train_pool)), replace=False))
    cal_idx = np.sort(rng.choice(cal_pool, size=min(args.n_cal, len(cal_pool)), replace=False))

    x_train_mm = np.load(TABULAR_DIR / "train" / "x.npy", mmap_mode="r")
    y_train_mm = np.load(TABULAR_DIR / "train" / "y.npy", mmap_mode="r")
    m_train_mm = np.load(TABULAR_DIR / "train" / "y_mask.npy", mmap_mode="r")

    for name, idx in (("train", train_idx), ("calibration", cal_idx)):
        print(f"Building {name} split ({len(idx):,} instances) ...")
        df = expand_instances(
            np.asarray(x_train_mm[idx], dtype=np.float32),
            np.asarray(y_train_mm[idx], dtype=np.float32),
            np.asarray(m_train_mm[idx], dtype=np.float32),
            md_train.iloc[idx].reset_index(drop=True),
            col_index,
            normalization,
        )
        df.to_parquet(PREPARED_DIR / f"{name}.parquet", index=False)
        print(f"  {len(df):,} (instance, horizon) rows -> {name}.parquet")

    del md_train

    # ------------------------------------------------------------------
    # Test period (2025)
    # ------------------------------------------------------------------
    print("Loading test metadata and evaluation table ...")
    md_test = load_metadata_array(TABULAR_DIR / "test" / "md.npy")
    eval_df = pd.read_parquet(CORE_DIR / "test_eval_table.parquet")
    assert len(md_test) == len(eval_df), "test arrays and eval table row counts differ"
    same_train = (md_test["train_id"].astype(str).to_numpy() == eval_df["train_id"].astype(str).to_numpy())
    assert same_train.all(), "test md.npy and test_eval_table.parquet are not row-aligned"

    test_idx = np.sort(rng.choice(len(md_test), size=min(args.n_test, len(md_test)), replace=False))
    x_test_mm = np.load(TABULAR_DIR / "test" / "x.npy", mmap_mode="r")
    y_test_mm = np.load(TABULAR_DIR / "test" / "y.npy", mmap_mode="r")
    m_test_mm = np.load(TABULAR_DIR / "test" / "y_mask.npy", mmap_mode="r")

    print(f"Building test split ({len(test_idx):,} instances) ...")
    df_test = expand_instances(
        np.asarray(x_test_mm[test_idx], dtype=np.float32),
        np.asarray(y_test_mm[test_idx], dtype=np.float32),
        np.asarray(m_test_mm[test_idx], dtype=np.float32),
        md_test.iloc[test_idx].reset_index(drop=True),
        col_index,
        normalization,
        eval_df=eval_df.iloc[test_idx].reset_index(drop=True),
    )
    df_test.to_parquet(PREPARED_DIR / "test.parquet", index=False)
    print(f"  {len(df_test):,} (instance, horizon) rows -> test.parquet")

    # Sanity check: denormalized target vs eval-table delay difference.
    approx = df_test["y_abs_delay_sec"] - df_test["last_known_delay_sec"]
    diff = (df_test["y_delta_sec"] - approx).abs()
    print(
        "Target sanity check (y_delta vs eval-table delay difference): "
        f"median abs diff = {diff.median():.1f}s, 90th pct = {diff.quantile(0.9):.1f}s"
    )

    write_manifest()
    print(f"\nDone. Prepared files in {PREPARED_DIR}")


if __name__ == "__main__":
    main()
