"""Module 5 - Uncertainty-aware explainability (TreeSHAP for Point vs CQR Width).

Decomposes both the point prediction (xgb_point) and the 90% CQR prediction
interval width (xgb_cqr_hi_90 - xgb_cqr_lo_90) using exact TreeSHAP.

Demonstrates three core Trustworthy-AI concepts for the ASEE module & portfolio:
  1. Accuracy drivers vs Uncertainty drivers: Features that drive expected delay
     change are not always the same features that drive interval width (uncertainty).
  2. Regime shift attribution: How feature attributions shift between normal
     operations and disrupted/high-congestion regimes (Scenario C).
  3. Scenario-level explanations: Local SHAP breakdowns for Module 4's
     'Scenario A (accurate but uncertain)' and 'Scenario C (disruption failure)'.

Usage:
    python src/module5_explainability.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from common import (
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    ensure_dirs,
    feature_columns,
    load_split,
    print_table,
)
from module2_uncertainty import QUANTILE_XGB_PARAMS, fit_quantile_model

PCT = 90
LO_Q, HI_Q = 0.05, 0.95
SAMPLE_SIZE = 2000  # Stratified test sample for fast, exact TreeSHAP computation


def load_or_fit_quantile_models(feats: list[str]) -> tuple[XGBRegressor, XGBRegressor]:
    """Load saved 90% CQR quantile models, or fit and cache them if missing."""
    lo_path = MODELS_DIR / f"xgb_cqr_lo_{PCT}.json"
    hi_path = MODELS_DIR / f"xgb_cqr_hi_{PCT}.json"

    if lo_path.exists() and hi_path.exists():
        print(f"Loading cached {PCT}% quantile models from {MODELS_DIR} ...")
        lo_model, hi_model = XGBRegressor(), XGBRegressor()
        lo_model.load_model(lo_path)
        hi_model.load_model(hi_path)
        return lo_model, hi_model

    print(f"Cached quantile models not found. Fitting {PCT}% quantile models once ...")
    train = load_split("train")
    X_train = train[feats]
    y_train = train["y_delta_sec"].to_numpy(dtype=np.float64)

    lo_model = fit_quantile_model(X_train, y_train, LO_Q)
    hi_model = fit_quantile_model(X_train, y_train, HI_Q)
    lo_model.save_model(lo_path)
    hi_model.save_model(hi_path)
    print(f"Saved quantile models to {lo_path} and {hi_path}")
    return lo_model, hi_model


def main() -> None:
    ensure_dirs()
    test = load_split("test")
    preds = pd.read_parquet(PREDICTIONS_DIR / "test_interval_predictions.parquet")
    feats = feature_columns(test)

    # Load point model and 90% CQR lower/upper quantile models
    point_model = XGBRegressor()
    point_model.load_model(MODELS_DIR / "xgb_point.json")
    lo_model, hi_model = load_or_fit_quantile_models(feats)

    # Identify disrupted regime rows (matching Module 4 Scenario C criteria)
    cong_q90 = test["congestion_mean_link_delay"].quantile(0.9)
    is_disrupted = (test["last_known_delay_sec"] > 300) & (
        test["congestion_mean_link_delay"] > cong_q90
    )

    rng = np.random.default_rng(RANDOM_SEED)
    normal_idx = np.flatnonzero(~is_disrupted.to_numpy())
    disrupted_idx = np.flatnonzero(is_disrupted.to_numpy())

    n_half = SAMPLE_SIZE // 2
    idx_norm_sample = rng.choice(normal_idx, size=min(n_half, len(normal_idx)), replace=False)
    idx_dis_sample = rng.choice(disrupted_idx, size=min(n_half, len(disrupted_idx)), replace=False)
    sample_idx = np.concatenate([idx_norm_sample, idx_dis_sample])

    X_sample = test.iloc[sample_idx][feats]
    regime_labels = np.array(
        ["normal"] * len(idx_norm_sample) + ["disrupted"] * len(idx_dis_sample)
    )

    print(f"Computing exact TreeSHAP values on {len(X_sample):,} test snapshots "
          f"({len(idx_norm_sample)} normal, {len(idx_dis_sample)} disrupted) ...")

    explainer_point = shap.TreeExplainer(point_model)
    explainer_lo = shap.TreeExplainer(lo_model)
    explainer_hi = shap.TreeExplainer(hi_model)

    shap_point = explainer_point.shap_values(X_sample)
    shap_lo = explainer_lo.shap_values(X_sample)
    shap_hi = explainer_hi.shap_values(X_sample)

    # By linearity of SHAP: attribution for CQR interval width (hi - lo)
    shap_width = shap_hi - shap_lo

    # ------------------------------------------------------------------
    # Global Importance Table: Point Prediction vs Interval Width (Uncertainty)
    # ------------------------------------------------------------------
    mean_abs_point = np.abs(shap_point).mean(axis=0)
    mean_abs_width = np.abs(shap_width).mean(axis=0)
    mean_width_norm = np.abs(shap_width[regime_labels == "normal"]).mean(axis=0)
    mean_width_dis = np.abs(shap_width[regime_labels == "disrupted"]).mean(axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "feature": feats,
                "point_mae_shap_sec": mean_abs_point,
                "cqr_width_shap_sec": mean_abs_width,
                "width_shap_normal_sec": mean_width_norm,
                "width_shap_disrupted_sec": mean_width_dis,
                "disruption_amplification_ratio": mean_width_dis
                / np.clip(mean_width_norm, 1e-3, None),
            }
        )
        .sort_values("cqr_width_shap_sec", ascending=False)
        .reset_index(drop=True)
    )

    print_table(
        importance_df.head(12),
        "Top 12 Features: Point Prediction Impact vs 90% CQR Uncertainty Width Impact",
    )
    importance_df.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    # ------------------------------------------------------------------
    # Figure 1: Point Prediction Drivers vs Uncertainty Width Drivers (Top 10)
    # ------------------------------------------------------------------
    top_k = 10
    top_df = importance_df.head(top_k).iloc[::-1]  # reverse for horizontal bar plot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    y_pos = np.arange(top_k)

    axes[0].barh(y_pos - 0.18, top_df["point_mae_shap_sec"], height=0.35, label="Point Prediction (|SHAP|)")
    axes[0].barh(y_pos + 0.18, top_df["cqr_width_shap_sec"], height=0.35, label="90% CQR Interval Width (|SHAP|)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_df["feature"], fontsize=9)
    axes[0].set_xlabel("Mean |SHAP| contribution (seconds)")
    axes[0].set_title("What drives predicted delay vs predictive uncertainty?")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y_pos - 0.18, top_df["width_shap_normal_sec"], height=0.35, label="Normal Regime")
    axes[1].barh(y_pos + 0.18, top_df["width_shap_disrupted_sec"], height=0.35, label="Disrupted Regime (Delay>5m & High Cong.)")
    axes[1].set_xlabel("Mean |SHAP| on 90% Interval Width (seconds)")
    axes[1].set_title("How uncertainty drivers shift during network disruptions")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Trustworthy AI Attribution: Explaining Point Predictions vs Conformal Interval Width", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_point_vs_width.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Local Explanation of Module 4 Decision Scenarios (A vs C)
    # ------------------------------------------------------------------
    preds["width"] = preds[f"cqr_upper_{PCT}"] - preds[f"cqr_lower_{PCT}"]
    preds["abs_error"] = (preds["y_delta_sec"] - preds["y_pred"]).abs()
    wide_cut = preds["width"].quantile(0.9)

    # Pick 1 representative row from Scenario A (accurate but uncertain) & Scenario C (disrupted miss)
    scen_a_idx = preds.index[(preds["abs_error"] < 30) & (preds["width"] >= wide_cut)][0]
    scen_c_idx = preds.index[
        is_disrupted & ((preds["y_delta_sec"] < preds[f"cqr_lower_{PCT}"]) | (preds["y_delta_sec"] > preds[f"cqr_upper_{PCT}"]))
    ][0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, row_idx, title in (
        (axes[0], scen_a_idx, "Scenario A: Accurate Point Error (<30s) but Top-Decile Uncertainty"),
        (axes[1], scen_c_idx, "Scenario C: Disrupted Operations Interval Breakdown"),
    ):
        x_row = test.loc[[row_idx], feats]
        w_shap = (explainer_hi.shap_values(x_row) - explainer_lo.shap_values(x_row))[0]
        top_feat_idx = np.argsort(np.abs(w_shap))[-8:]
        labels = [f"{feats[i]} = {x_row.iloc[0, i]:.1f}" for i in top_feat_idx]
        vals = w_shap[top_feat_idx]
        colors = ["#d95f02" if v > 0 else "#1b9e77" for v in vals]

        ax.barh(np.arange(len(top_feat_idx)), vals, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(top_feat_idx)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Contribution to 90% CQR Interval Width (s)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Local Interval-Width Attribution for Engineering Decision Scenarios", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_scenario_explanations.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved SHAP table: {REPORTS_DIR / 'shap_feature_importance.csv'}")
    print(f"Saved figures: {FIGURES_DIR / 'shap_point_vs_width.png'}, "
          f"{FIGURES_DIR / 'shap_scenario_explanations.png'}")


if __name__ == "__main__":
    main()