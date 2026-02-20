"""
Thesis Visualisation & Calibration Analysis for demo_all (best Cox PH model)
=============================================================================
Produces:
  1. combined_km_all_folds.png  — all 5 KM curves on one figure (2×3 grid)
  2. calibration_plot.png       — observed vs predicted risk (decile calibration)
  3. logrank_table.csv          — log-rank p-value per fold

Run after completing the ablation study:
    python thesis_plots.py
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ── Paths — adjust to match your environment ──────────────────────────────────
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from cox_survival_analysis import (
    SurvivalDataLoader, CoxSurvivalAnalyzer,
    CONFIG, HAND_FEATURE_COLS, DEMO_FEATURE_COLS
)
import cox_survival_analysis as _cox_mod

OUT_DIR = CONFIG["output_dir"] / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# RE-RUN demo_all to get fold results in memory
# (skip if you already have results stored)
# ─────────────────────────────────────────────────────────────────────────────

def get_demo_all_results():
    """Fit demo_all (Sex+Age+SmokingStatus) across 5 folds and return results."""
    import copy

    with open(CONFIG["kfold_splits_path"], "rb") as f:
        kfold_splits = pickle.load(f)

    cfg = copy.deepcopy(CONFIG)
    cfg["penalizer"] = 0.5
    cfg["l1_ratio"]  = 0.0

    loader = SurvivalDataLoader(cfg)
    df = loader.prepare_full_dataset(
        use_cnn=False, use_hand=False, use_demo=True
    )

    analyzer = CoxSurvivalAnalyzer(cfg)
    results = []
    for fold_num in range(1, 6):
        split   = kfold_splits[f"fold_{fold_num}"]
        tr = df[df["Patient"].isin(split["train"])].reset_index(drop=True)
        va = df[df["Patient"].isin(split["val"])].reset_index(drop=True)
        te = df[df["Patient"].isin(split.get("test", split["val"]))].reset_index(drop=True)
        res = analyzer.fit_fold(tr, va, te, fold_num)
        if res:
            results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMBINED KM PLOT — all 5 folds + summary panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_km(results, out_path: Path):
    """
    2×3 grid: folds 1-5 + summary panel.
    Each subplot shows val-set KM curves for high vs low risk.
    Summary panel shows mean ± std C-index and log-rank p per fold.
    """
    CRIMSON   = "#C0392B"
    STEELBLUE = "#2471A3"
    CI_RED    = "#F1948A"
    CI_BLUE   = "#AED6F1"

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    fold_stats = []
    kmf = KaplanMeierFitter()

    for idx, res in enumerate(results):
        fold_num = res["fold"]
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        val_df   = res["val_df"]
        val_risk = res["val_risk"]
        median   = val_risk.median()

        hi_mask = val_risk.values >= median
        lo_mask = ~hi_mask

        # Log-rank
        lr_p = np.nan
        if hi_mask.sum() > 0 and lo_mask.sum() > 0:
            lr = logrank_test(
                val_df.loc[hi_mask, "time"], val_df.loc[lo_mask, "time"],
                event_observed_A=val_df.loc[hi_mask, "event"],
                event_observed_B=val_df.loc[lo_mask, "event"],
            )
            lr_p = lr.p_value

        fold_stats.append({
            "fold": fold_num,
            "val_ci": res["val_ci"],
            "test_ci": res["test_ci"],
            "logrank_p": lr_p,
            "n_hi": int(hi_mask.sum()),
            "n_lo": int(lo_mask.sum()),
        })

        # Plot each group
        for mask, label, lc, fc in [
            (hi_mask, "High risk", CRIMSON,   CI_RED),
            (lo_mask, "Low risk",  STEELBLUE, CI_BLUE),
        ]:
            t = val_df.loc[mask, "time"].values
            e = val_df.loc[mask, "event"].values
            if len(t) == 0:
                continue
            kmf.fit(t, e, label=label)
            kmf.plot_survival_function(
                ax=ax, ci_show=True,
                color=lc,
                linewidth=2,
                legend=False,
            )
            # Manually adjust CI color
            if ax.collections:
                ax.collections[-1].set_facecolor(fc)
                ax.collections[-1].set_alpha(0.25)

        # Annotations
        p_str  = f"p = {lr_p:.3f}" if not np.isnan(lr_p) else "p = n/a"
        ci_str = f"C = {res['val_ci']:.3f}"
        sig    = "✱" if (not np.isnan(lr_p) and lr_p < 0.05) else ""
        ax.set_title(
            f"Fold {fold_num}  {sig}",
            fontsize=12, fontweight="bold", color="#1A1A2E", pad=6
        )
        ax.text(
            0.97, 0.97, f"{ci_str}\n{p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#333333",
            bbox=dict(fc="white", ec="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.3")
        )
        ax.set_xlabel("Time (weeks)", fontsize=9)
        ax.set_ylabel("Progression-free prob.", fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # ── Summary panel (bottom-right) ──────────────────────────────────────────
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.axis("off")

    stats_df = pd.DataFrame(fold_stats)
    mean_val  = stats_df["val_ci"].mean()
    std_val   = stats_df["val_ci"].std()
    mean_test = stats_df["test_ci"].mean()
    std_test  = stats_df["test_ci"].std()
    n_sig     = (stats_df["logrank_p"] < 0.05).sum()

    summary_text = (
        f"Model:  demo_all\n"
        f"Features:  Sex · Age · SmokingStatus\n"
        f"Regularisation:  Ridge  (λ = 0.5)\n\n"
        f"Val C-index:    {mean_val:.3f} ± {std_val:.3f}\n"
        f"Test C-index:  {mean_test:.3f} ± {std_test:.3f}\n\n"
        f"Log-rank p < 0.05:  {n_sig} / 5 folds\n\n"
        f"✱  Significant fold"
    )
    ax_s.text(
        0.05, 0.95, summary_text,
        transform=ax_s.transAxes, ha="left", va="top",
        fontsize=9.5, family="monospace",
        color="#1A1A2E",
        bbox=dict(fc="#F0F4FF", ec="#2E75B6", linewidth=1.5,
                  boxstyle="round,pad=0.6", alpha=0.95)
    )

    # Shared legend
    legend_els = [
        Line2D([0],[0], color=CRIMSON,   lw=2.5, label="High risk"),
        Line2D([0],[0], color=STEELBLUE, lw=2.5, label="Low risk"),
    ]
    fig.legend(
        handles=legend_els, loc="lower center",
        ncol=2, fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.38, -0.01),
        edgecolor="#CCCCCC",
    )

    fig.suptitle(
        "Kaplan-Meier Risk Stratification — demo_all  (Sex + Age + SmokingStatus)\n"
        "Cox Proportional Hazards · 5-Fold Cross-Validation · IPF Cohort (N=84)",
        fontsize=13, fontweight="bold", color="#1A1A2E", y=1.01
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Combined KM plot → {out_path}")
    return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CALIBRATION PLOT — observed vs predicted risk
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(results, out_path: Path, n_deciles: int = 5):
    """
    Pool val-set patients across all folds, bin by predicted risk decile,
    and compare mean predicted risk to observed event rate per bin.

    With only ~84 patients / 17 per fold, use quintiles (5 bins) not deciles.
    Also show a secondary panel: predicted risk distribution by event status.
    """
    # Pool val-set data across folds
    all_risk, all_event, all_time = [], [], []
    for res in results:
        all_risk.extend(res["val_risk"].values.tolist())
        all_event.extend(res["val_df"]["event"].values.tolist())
        all_time.extend(res["val_df"]["time"].values.tolist())

    pool = pd.DataFrame({
        "risk":  all_risk,
        "event": all_event,
        "time":  all_time,
    })

    # Normalise risk to [0,1] for interpretability
    pool["risk_norm"] = (pool["risk"] - pool["risk"].min()) / \
                        (pool["risk"].max() - pool["risk"].min() + 1e-9)

    # Bin into quintiles
    pool["bin"] = pd.qcut(pool["risk_norm"], q=n_deciles, labels=False,
                          duplicates="drop")

    bin_stats = pool.groupby("bin").agg(
        mean_pred  = ("risk_norm", "mean"),
        obs_rate   = ("event",     "mean"),
        n          = ("event",     "count"),
        mean_time  = ("time",      "mean"),
    ).reset_index()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # Panel A — calibration scatter
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4, label="Perfect calibration")

    scatter = ax.scatter(
        bin_stats["mean_pred"], bin_stats["obs_rate"],
        s=bin_stats["n"] * 18,   # size proportional to N in bin
        c=bin_stats["mean_time"],
        cmap="RdYlGn_r",
        edgecolors="#333333", linewidths=0.8,
        zorder=5, alpha=0.9,
    )
    cb = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Mean time in bin (weeks)", fontsize=8)

    # Annotate each point with bin N
    for _, row in bin_stats.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["mean_pred"], row["obs_rate"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color="#333333"
        )

    # Pearson r between predicted and observed
    if len(bin_stats) >= 3:
        r_val, p_val = pearsonr(bin_stats["mean_pred"], bin_stats["obs_rate"])
        ax.text(0.05, 0.93, f"Pearson r = {r_val:.3f}\np = {p_val:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="#AAAAAA", alpha=0.85, boxstyle="round,pad=0.3"))

    ax.set_xlabel("Mean predicted risk (normalised, quintile)", fontsize=10)
    ax.set_ylabel("Observed event rate (quintile)", fontsize=10)
    ax.set_title("A — Calibration: Predicted vs Observed", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)
    note = (f"Bins = {n_deciles} quintiles · N per bin ∝ marker size\n"
            "Pooled across all 5 validation folds")
    ax.text(0.05, 0.04, note, transform=ax.transAxes,
            fontsize=7.5, color="#666666", style="italic")

    # Panel B — risk score distribution by event status
    ax2 = axes[1]
    ev_risk  = pool.loc[pool["event"] == 1, "risk_norm"].values
    cen_risk = pool.loc[pool["event"] == 0, "risk_norm"].values

    bins_b = np.linspace(0, 1, 16)
    ax2.hist(cen_risk, bins=bins_b, alpha=0.65, color="#2471A3",
             label=f"Censored (n={len(cen_risk)})", edgecolor="white", linewidth=0.5)
    ax2.hist(ev_risk,  bins=bins_b, alpha=0.65, color="#C0392B",
             label=f"Event (n={len(ev_risk)})",    edgecolor="white", linewidth=0.5)

    # Median lines
    ax2.axvline(np.median(ev_risk),  color="#C0392B",  ls="--", lw=1.5,
                label=f"Median event: {np.median(ev_risk):.3f}")
    ax2.axvline(np.median(cen_risk), color="#2471A3", ls="--", lw=1.5,
                label=f"Median censored: {np.median(cen_risk):.3f}")

    ax2.set_xlabel("Predicted risk score (normalised)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("B — Risk Score Distribution by Event Status", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.spines[["top","right"]].set_visible(False)

    # KS-test note
    from scipy.stats import ks_2samp
    if len(ev_risk) > 0 and len(cen_risk) > 0:
        ks_stat, ks_p = ks_2samp(ev_risk, cen_risk)
        ax2.text(0.97, 0.97,
                 f"KS statistic = {ks_stat:.3f}\nKS p = {ks_p:.3f}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
                 bbox=dict(fc="white", ec="#AAAAAA", alpha=0.85, boxstyle="round,pad=0.3"))

    fig.suptitle(
        "Calibration Analysis — demo_all (Sex + Age + SmokingStatus)\n"
        "Pooled validation set · 5-fold CV · N=84 IPF patients",
        fontsize=12, fontweight="bold", color="#1A1A2E", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Calibration plot → {out_path}")
    return pool, bin_stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOG-RANK TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_logrank_table(stats_df: pd.DataFrame, out_path: Path):
    stats_df = stats_df.copy()
    stats_df["significant"] = stats_df["logrank_p"].apply(
        lambda p: "Yes ✱" if p < 0.05 else "No"
    )
    stats_df["val_ci"]    = stats_df["val_ci"].round(4)
    stats_df["test_ci"]   = stats_df["test_ci"].round(4)
    stats_df["logrank_p"] = stats_df["logrank_p"].round(4)
    stats_df.rename(columns={
        "fold":       "Fold",
        "val_ci":     "Val C-index",
        "test_ci":    "Test C-index",
        "logrank_p":  "Log-rank p",
        "n_hi":       "N high-risk",
        "n_lo":       "N low-risk",
        "significant":"p < 0.05",
    }, inplace=True)

    # Add mean/std summary rows
    num_cols = ["Val C-index", "Test C-index", "Log-rank p"]
    mean_row = {"Fold": "Mean"}
    std_row  = {"Fold": "Std"}
    for c in num_cols:
        mean_row[c] = round(stats_df[c].mean(), 4)
        std_row[c]  = round(stats_df[c].std(),  4)

    stats_df = pd.concat(
        [stats_df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )
    stats_df.to_csv(out_path, index=False)
    print(f"  ✓ Log-rank table → {out_path}")
    print(stats_df.to_string(index=False))
    return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("THESIS FIGURES — demo_all  (Sex + Age + SmokingStatus)")
    print(f"{'='*70}\n")

    print("Re-fitting demo_all across 5 folds...")
    results = get_demo_all_results()
    print(f"  ✓ {len(results)} folds fitted\n")

    print("1. Combined KM plot...")
    stats_df = plot_combined_km(
        results,
        OUT_DIR / "combined_km_demo_all.png"
    )

    print("\n2. Calibration plot...")
    pool, bin_stats = plot_calibration(
        results,
        OUT_DIR / "calibration_demo_all.png",
        n_deciles=5,
    )

    print("\n3. Log-rank table...")
    save_logrank_table(stats_df, OUT_DIR / "logrank_table_demo_all.csv")

    print(f"\n{'='*70}")
    print(f"✓ All figures saved → {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()