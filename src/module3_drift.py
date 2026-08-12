"""Module 3 - Temporal drift and conditional reliability.

Slices the Module 2 interval predictions by operating condition to test
whether the marginal coverage guarantee holds *conditionally*:

  - prediction horizon (0-5 ... 45+ minutes ahead),
  - operating regime (on-time vs slightly vs heavily delayed trains),
  - local congestion (active trains on nearby segments),
  - calendar month of the 2025 test period (temporal drift since the
    2023-2024 training data).

Prints one conditional-reliability table per breakdown and saves the
corresponding plots.

Usage:
    python src/module3_drift.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    FIGURES_DIR,
    PREDICTIONS_DIR,
    ensure_dirs,
    group_reliability_table,
    horizon_bin,
    print_table,
)

NOMINAL = 0.9  # analyze the 90% intervals


def add_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["horizon_bin"] = horizon_bin(df["obs_horizon_min"])
    df["delay_regime"] = pd.cut(
        df["last_known_delay_sec"],
        bins=[-np.inf, 60, 300, np.inf],
        labels=["on time (<1min)", "slightly delayed (1-5min)", "heavily delayed (>5min)"],
    )
    congestion_cut = df["congestion_total_trains"].quantile(0.75)
    df["congestion_regime"] = np.where(
        df["congestion_total_trains"] > congestion_cut,
        f"high (>{congestion_cut:.0f} trains)",
        f"low (<={congestion_cut:.0f} trains)",
    )
    return df


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PREDICTIONS_DIR / "test_interval_predictions.parquet")
    df = add_condition_columns(df)
    pct = int(NOMINAL * 100)

    print(f"--- Module 3: conditional reliability of the {pct}% intervals (2025 test period) ---")
    print(f"Test rows: {len(df):,}")

    tables = {}
    for method, label in (("split", "Split conformal"), ("cqr", "CQR")):
        lo, up = f"{method}_lower_{pct}", f"{method}_upper_{pct}"
        for cond, title in (
            ("horizon_bin", "prediction horizon (min ahead)"),
            ("delay_regime", "delay regime at prediction time"),
            ("congestion_regime", "local congestion"),
            ("month", "test month (drift since 2023-2024 training data)"),
        ):
            t = group_reliability_table(df, cond, lower_col=lo, upper_col=up)
            tables[(method, cond)] = t
            print_table(t, f"{label}: reliability by {title}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, marker, label in (("split", "o", "split conformal"), ("cqr", "s", "CQR")):
        th = tables[(method, "horizon_bin")]
        axes[0].plot(th["horizon_bin"].astype(str), th["coverage"], marker + "-", label=label)
        axes[1].plot(th["horizon_bin"].astype(str), th["mean_width_sec"], marker + "-", label=label)
        tm = tables[(method, "month")].sort_values("month")
        axes[2].plot(tm["month"], tm["coverage"], marker + "-", label=label)

    axes[0].axhline(NOMINAL, color="k", ls="--", lw=1, label=f"nominal {pct}%")
    axes[0].set_title("Coverage by prediction horizon")
    axes[0].set_xlabel("Horizon (min ahead)")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_title("Interval width by prediction horizon")
    axes[1].set_xlabel("Horizon (min ahead)")
    axes[1].set_ylabel("Mean interval width (s)")
    axes[2].axhline(NOMINAL, color="k", ls="--", lw=1)
    axes[2].set_title("Coverage by test month (temporal drift)")
    axes[2].set_xlabel("Month of 2025")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conditional_reliability.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond, title in (
        (axes[0], "delay_regime", "Delay regime at prediction time"),
        (axes[1], "congestion_regime", "Local congestion"),
    ):
        labels = None
        for method, label in (("split", "split conformal"), ("cqr", "CQR")):
            t = tables[(method, cond)]
            labels = t[cond].astype(str)
            ax.bar(
                np.arange(len(t)) + (0 if method == "split" else 0.35),
                t["coverage"],
                width=0.35,
                label=label,
            )
        ax.axhline(NOMINAL, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(len(labels)) + 0.175)
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(f"Empirical coverage of {pct}% intervals")
    fig.suptitle("Does the average coverage hide failures in specific operating conditions?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regime_coverage.png", dpi=150)
    plt.close(fig)

    all_tables = pd.concat(
        [
            t.assign(method=m, condition=c).rename(columns={c: "group"})
            for (m, c), t in tables.items()
        ],
        ignore_index=True,
    )
    all_tables.to_csv(PREDICTIONS_DIR / "conditional_reliability_tables.csv", index=False)
    print(f"\nSaved tables to {PREDICTIONS_DIR / 'conditional_reliability_tables.csv'}")
    print(f"Saved figures: {FIGURES_DIR / 'conditional_reliability.png'}, "
          f"{FIGURES_DIR / 'regime_coverage.png'}")


if __name__ == "__main__":
    main()
