"""Shared paths, data loading, denormalization, and reliability metrics.

All downstream module scripts import from here so that units, bins, and
metric definitions stay consistent across the pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TABULAR_DIR = RAW_DIR / "tabular"
CORE_DIR = RAW_DIR / "core"

# Prediction-horizon bins in minutes, matching the RIDE evaluation protocol.
HORIZON_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, np.inf]
HORIZON_BIN_LABELS = [
    "0-5", "5-10", "10-15", "15-20", "20-25",
    "25-30", "30-35", "35-40", "40-45", "45+",
]

# Delay-change bins in seconds (change relative to last known delay).
DELTA_BIN_EDGES = [-np.inf, -120, -30, 30, 120, 300, np.inf]
DELTA_BIN_LABELS = ["<-2min", "-2..-0.5min", "-0.5..0.5min", "0.5..2min", "2..5min", ">5min"]

# Columns of the prepared parquets that must never be fed to a model.
NON_FEATURE_COLS = [
    "snapshot_ts",
    "train_id",
    "month",
    "last_known_delay_sec",
    "y_delta_sec",
    "y_abs_delay_sec",
    "obs_horizon_min",
]

RANDOM_SEED = 42


def feature_columns(df: "pd.DataFrame") -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def load_split(name: str) -> "pd.DataFrame":
    path = PREPARED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run src/prepare_data.py first")
    return pd.read_parquet(path)


def ensure_dirs() -> None:
    for d in (PREPARED_DIR, MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_scheme() -> dict:
    with open(TABULAR_DIR / "scheme.yaml") as f:
        return yaml.safe_load(f)


def load_normalization() -> dict:
    with open(TABULAR_DIR / "normalization.yaml") as f:
        return yaml.safe_load(f)


def denormalize(values: np.ndarray, column: str, normalization: dict) -> np.ndarray:
    """Invert the RIDE normalization for one column, returning physical units.

    Normalized columns are either min-max scaled or z-scored, the latter
    optionally after a signed square-root transform. Columns absent from the
    normalization spec (binary indicators, cyclical encodings) pass through.
    """
    spec = normalization.get(column)
    if spec is None:
        return values.astype(np.float32)
    values = values.astype(np.float64)
    if spec["norm"] == "minmax":
        out = values * (spec["max"] - spec["min"]) + spec["min"]
    else:  # zscore
        out = values * spec["std"] + spec["mean"]
        if spec.get("sqrt", False):
            out = np.sign(out) * out**2
    return out.astype(np.float32)


def load_metadata_array(path: Path) -> pd.DataFrame:
    """Load an md.npy file into a DataFrame with snapshot_ts / train_id / service_date."""
    try:
        arr = np.load(path, mmap_mode="r")
    except ValueError:
        arr = np.load(path, allow_pickle=True)
    if arr.dtype.names:  # structured array
        df = pd.DataFrame({name: arr[name] for name in arr.dtype.names})
    else:
        df = pd.DataFrame(np.asarray(arr), columns=["snapshot_ts", "train_id", "service_date"])
    df.columns = ["snapshot_ts", "train_id", "service_date"][: len(df.columns)]
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    return df


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values falling inside [lower, upper]."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def median_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.median(np.asarray(upper) - np.asarray(lower)))


def horizon_bin(horizon_minutes: pd.Series) -> pd.Categorical:
    return pd.cut(
        horizon_minutes,
        bins=HORIZON_BIN_EDGES,
        labels=HORIZON_BIN_LABELS,
        right=False,
        include_lowest=True,
    )


def group_reliability_table(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "y_delta_sec",
    pred_col: str = "y_pred",
    lower_col: str = "lower",
    upper_col: str = "upper",
) -> pd.DataFrame:
    """Per-group MAE / coverage / mean width / count summary."""
    rows = []
    for group, g in df.groupby(group_col, observed=True):
        rows.append(
            {
                group_col: group,
                "n": len(g),
                "mae_sec": mae(g[y_col], g[pred_col]),
                "coverage": coverage(g[y_col], g[lower_col], g[upper_col]),
                "mean_width_sec": mean_width(g[lower_col], g[upper_col]),
            }
        )
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str, float_fmt: str = "{:.3f}") -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: float_fmt.format(v)))
