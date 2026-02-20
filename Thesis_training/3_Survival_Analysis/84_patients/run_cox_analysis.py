"""
Full Ablation Study for Cox PH Survival Analysis — IPF Progression
===================================================================
Mirrors the Excel table (ipf_ablation_full.xlsx) exactly.

Blocks
------
  BLOCK 1 — Demographic features (Sex, Age, SmokingStatus and combos)
  BLOCK 2 — Handcrafted radiomics (individual + subsets + all-9)
  BLOCK 3 — CNN deep features (mean/max pool 4-stat, PCA variants)
  BLOCK 4 — Cross-block combinations (best choices from Blocks 1-3)

Preferred choices (justified by diagnostics):
  - Demographic : Sex only  (C=0.536, stable; Age/Smoking unstable or inverted)
  - Handcrafted : All 9 with Ridge  (best test C-index 0.519 despite EPV concern)
                  OR Kurtosis + ApproxVol + Sex  (EPV-safe parsimonious model)
  - CNN         : None directly in Cox  (EPV constraint kills all PCA variants)
                  Proposed: use LightGBM P(progression) as single scalar
  - Best combo  : hand_all_demo  (all 9 handcrafted + Sex, penalizer=0.5, Ridge)
"""

from pathlib import Path
import copy
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from cox_survival_analysis import (
    SurvivalDataLoader, CoxSurvivalAnalyzer,
    CONFIG, HAND_FEATURE_COLS, HAND_FEATURE_COLS_ALL,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(use_cnn=False, use_hand=True, use_demo=True,
         hand_cols=None, penalizer=None, description="",
         pooling_type=None, cnn_stats=None, use_pca=False, n_pca=3):
    """Build a single ablation config entry."""
    cfg = {
        "use_cnn":          use_cnn,
        "use_hand":         use_hand,
        "use_demo":         use_demo,
        "hand_feature_cols": hand_cols,
        "penalizer":        penalizer,   # None → auto (0.5 no-CNN, 10.0 CNN)
        "description":      description,
    }
    if pooling_type is not None:
        cfg["pooling_type"] = pooling_type
    if cnn_stats is not None:
        cfg["cnn_stats_to_use"] = cnn_stats
    if use_pca:
        cfg["use_cnn_pca"] = True
        cfg["n_pca_components"] = n_pca
    return cfg

def _make_cox_config(use_cnn: bool, penalizer_override=None, 
                     pooling_type=None, cnn_stats=None, 
                     use_pca=False, n_pca=3) -> dict:
    cfg = copy.deepcopy(CONFIG)
    cfg["penalizer"] = penalizer_override if penalizer_override is not None \
                       else (10.0 if use_cnn else 0.5)
    cfg["l1_ratio"]  = 0.0
    if pooling_type is not None:
        cfg["pooling_type"] = pooling_type
    if cnn_stats is not None:
        cfg["cnn_stats_to_use"] = cnn_stats
    if use_pca:
        cfg["use_cnn_pca"] = True
        cfg["n_pca_components"] = n_pca
    return cfg


# Individual handcrafted feature column names (for single-feature ablations)
KURTOSIS   = ["Kurtosis_30_60"]
APPROXVOL  = ["ApproxVol_30_60"]
THICKNESS  = ["Avg_Tissue_thickness_30_60"]
LUNG_RATIO = ["Avg_TissueByLung_30_60"]
MEAN_HU    = ["Mean_30_60"]
SKEW       = ["Skew_30_60"]
REDUNDANT  = ["Avg_NumTissuePixel_30_60", "Avg_Tissue_30_60", "Avg_TissueByTotal_30_60"]

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIG REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: CNN-containing configs require CNN feature extraction (~4 min on GPU).
#       Set SKIP_CNN = True to run only handcrafted / demographic blocks.
SKIP_CNN = False

ABLATION_CONFIGS = {}

# ══ BLOCK 1: DEMOGRAPHIC ═════════════════════════════════════════════════════

ABLATION_CONFIGS["demo_sex_only"] = _cfg(
    use_hand=False, use_demo=True,
    description="[DEMO] Sex only — best single demographic (C=0.536)",
)
ABLATION_CONFIGS["demo_age_only"] = _cfg(
    use_hand=False, use_demo=True,
    # Override DEMO_FEATURE_COLS at runtime via monkey-patch — see run_ablation()
    description="[DEMO] Age only — unstable sign across folds",
)
ABLATION_CONFIGS["demo_smoking_only"] = _cfg(
    use_hand=False, use_demo=True,
    description="[DEMO] SmokingStatus only — inverted, C≈0.493",
)
ABLATION_CONFIGS["demo_sex_age"] = _cfg(
    use_hand=False, use_demo=True,
    description="[DEMO] Sex + Age — does Age add to Sex?",
)
ABLATION_CONFIGS["demo_all"] = _cfg(
    use_hand=False, use_demo=True,
    description="[DEMO] Sex + Age + SmokingStatus — full demographic block",
)

# ══ BLOCK 2: HANDCRAFTED ══════════════════════════════════════════════════════

ABLATION_CONFIGS["hand_kurtosis_only"] = _cfg(
    use_demo=False, hand_cols=KURTOSIS,
    description="[HAND] Kurtosis only — best univariate (Diag 2: C=0.583)",
)
ABLATION_CONFIGS["hand_approxvol_only"] = _cfg(
    use_demo=False, hand_cols=APPROXVOL,
    description="[HAND] ApproxVol only — 2nd best univariate (C=0.571)",
)
ABLATION_CONFIGS["hand_thickness_only"] = _cfg(
    use_demo=False, hand_cols=THICKNESS,
    description="[HAND] Thickness only — moderate (C=0.527)",
)
ABLATION_CONFIGS["hand_lungfrac_only"] = _cfg(
    use_demo=False, hand_cols=LUNG_RATIO,
    description="[HAND] TissueByLung only — borderline (C=0.518)",
)
ABLATION_CONFIGS["hand_mean_only"] = _cfg(
    use_demo=False, hand_cols=MEAN_HU,
    description="[HAND] Mean HU only — weakest single (C=0.514)",
)
ABLATION_CONFIGS["hand_skew_only"] = _cfg(
    use_demo=False, hand_cols=SKEW,
    description="[HAND] Skew only — sign flips across folds",
)
ABLATION_CONFIGS["hand_redundant_cluster"] = _cfg(
    use_demo=False, hand_cols=REDUNDANT,
    description="[HAND] NumTissuePixel+Tissue+TissueByTotal — collinear cluster (r>0.85)",
)
ABLATION_CONFIGS["hand_reduced_3"] = _cfg(
    use_demo=False, hand_cols=HAND_FEATURE_COLS,
    description="[HAND] Kurtosis+ApproxVol+Thickness — reduced 3-feature set",
)
ABLATION_CONFIGS["hand_top2"] = _cfg(
    use_demo=False, hand_cols=KURTOSIS + APPROXVOL,
    description="[HAND] Kurtosis+ApproxVol — top-2 features",
)
ABLATION_CONFIGS["hand_all_9"] = _cfg(
    use_demo=False, hand_cols=HAND_FEATURE_COLS_ALL,
    description="[HAND] All 9 handcrafted — best test C-index in hand-only block",
)

# ══ BLOCK 3: CNN ══════════════════════════════════════════════════════════════

if not SKIP_CNN:
    # Setup A: Mean pooling → 4 statistics (legacy — confirmed noise)
    ABLATION_CONFIGS["cnn_4stat_only"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=False,
        description="[CNN Setup A] Mean-pool → 4 statistics (Mean/Var/L2/Entropy) — legacy aggregation",
    )
    ABLATION_CONFIGS["cnn_4stat_sex"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=True,
        description="[CNN Setup A] 4-stat + Sex",
    )
    
    # Setup B: Max pooling → 4 statistics (test stability)
    ABLATION_CONFIGS["cnn_maxpool_4stat"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=False,
        pooling_type='max',
        description="[CNN Setup B] Max-pool → 4 statistics — test variance stability vs mean-pool",
    )
    ABLATION_CONFIGS["cnn_maxpool_4stat_sex"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=True,
        pooling_type='max',
        description="[CNN Setup B] Max-pool → 4 statistics + Sex",
    )
    ABLATION_CONFIGS["cnn_maxpool_4stat_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        pooling_type='max',
        description="[CNN Setup B] Max-pool → 4 statistics + All 9 handcrafted + Sex",
    )
    
    # Setup C: Mean pooling → PCA (low-rank reduction)
    ABLATION_CONFIGS["cnn_pca3_only"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=False,
        use_pca=True, n_pca=3,
        description="[CNN Setup C] Mean-pool → PCA 3 components — low-rank reduction",
    )
    ABLATION_CONFIGS["cnn_pca3_sex"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=True,
        use_pca=True, n_pca=3,
        description="[CNN Setup C] PCA 3 components + Sex",
    )
    ABLATION_CONFIGS["cnn_pca3_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        use_pca=True, n_pca=3,
        description="[CNN Setup C] PCA 3 components + All 9 handcrafted + Sex",
    )
    ABLATION_CONFIGS["cnn_pca5_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        use_pca=True, n_pca=5,
        description="[CNN Setup C] PCA 5 components + All 9 handcrafted + Sex — more variance retained",
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # PRIORITY 1: CNN-ONLY VARIANTS (Isolate CNN Signal)
    # ═════════════════════════════════════════════════════════════════════════
    ABLATION_CONFIGS["cnn_pca5_only"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=False,
        use_pca=True, n_pca=5,
        description="[CNN-ONLY] PCA 5 components — isolate CNN signal",
    )
    ABLATION_CONFIGS["cnn_maxpool_no_l2norm_only"] = _cfg(
        use_cnn=True, use_hand=False, use_demo=False,
        pooling_type='max',
        cnn_stats=['mean', 'variance', 'entropy'],
        description="[CNN-ONLY] Max-pool → 3 stats (no L2Norm) — isolate CNN signal",
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # PRIORITY 2: PCA COMPONENT GRANULARITY (Find Optimal Dimensionality)
    # ═════════════════════════════════════════════════════════════════════════
    ABLATION_CONFIGS["cnn_pca2_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        use_pca=True, n_pca=2,
        description="[PCA-SWEEP] PCA 2 components + All 9 + Sex — ultra parsimonious",
    )
    ABLATION_CONFIGS["cnn_pca7_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        use_pca=True, n_pca=7,
        description="[PCA-SWEEP] PCA 7 components + All 9 + Sex — more variance",
    )
    ABLATION_CONFIGS["cnn_pca10_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        use_pca=True, n_pca=10,
        description="[PCA-SWEEP] PCA 10 components + All 9 + Sex — high variance retention",
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # PRIORITY 3: NO L2NORM CONFIGS (Test Stability Without L2Norm)
    # ═════════════════════════════════════════════════════════════════════════
    ABLATION_CONFIGS["cnn_3stat_no_l2_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        cnn_stats=['mean', 'variance', 'entropy'],
        description="[NO-L2NORM] Mean-pool → 3 stats (no L2Norm) + All 9 + Sex",
    )
    ABLATION_CONFIGS["cnn_maxpool_3stat_no_l2_hand_all_sex"] = _cfg(
        use_cnn=True, use_hand=True, use_demo=True,
        hand_cols=HAND_FEATURE_COLS_ALL,
        pooling_type='max',
        cnn_stats=['mean', 'variance', 'entropy'],
        description="[NO-L2NORM] Max-pool → 3 stats (no L2Norm) + All 9 + Sex",
    )

# ══ BLOCK 4: CROSS-BLOCK COMBINATIONS ════════════════════════════════════════

ABLATION_CONFIGS["hand_kurtosis_sex"] = _cfg(
    use_demo=True, hand_cols=KURTOSIS,
    description="[COMBO] Kurtosis + Sex — minimal EPV-safe model (2 df)",
)
ABLATION_CONFIGS["hand_top2_sex"] = _cfg(
    use_demo=True, hand_cols=KURTOSIS + APPROXVOL,
    description="[COMBO] Kurtosis + ApproxVol + Sex — 3 df, EPV=10",
)
ABLATION_CONFIGS["hand_reduced_3_sex"] = _cfg(
    use_demo=True, hand_cols=HAND_FEATURE_COLS,
    description="[COMBO] Kurtosis+ApproxVol+Thickness + Sex — reduced 3 + demo",
)
ABLATION_CONFIGS["hand_all_demo"] = _cfg(
    use_demo=True, hand_cols=HAND_FEATURE_COLS_ALL,
    description="[COMBO ★ RECOMMENDED] All 9 handcrafted + Sex — best test C-index",
)
ABLATION_CONFIGS["hand_all_9_no_demo"] = _cfg(
    use_demo=False, hand_cols=HAND_FEATURE_COLS_ALL,
    description="[COMBO] All 9 handcrafted, no demographics — isolate hand signal",
)
ABLATION_CONFIGS["kurtosis_approxvol_thickness_sex"] = _cfg(
    use_demo=True, hand_cols=KURTOSIS + APPROXVOL + THICKNESS,
    description="[COMBO] Top-3 individual + Sex",
)
ABLATION_CONFIGS["demo_all_thickness"] = _cfg(
    use_cnn=False,
    use_hand=True,
    use_demo=True,
    hand_cols=["Avg_Tissue_thickness_30_60"],
    description="[COMBO] Sex+Age+SmokingStatus + Avg_Tissue_Thickness — best demo + best hand",
)
# 1. Your new best — close the loop with all 9 hand features added to demo_all
ABLATION_CONFIGS["demo_all_hand_all"] = _cfg(
    use_hand=True, use_demo=True,
    hand_cols=HAND_FEATURE_COLS_ALL,
    description="[COMBO] Sex+Age+Smoking + All 9 hand — full combo",
)

# 2. Your current best (demo_all_thickness) but swap Sex-only for full demographics
# i.e. confirm it's Age+Smoking driving the gain, not just Sex+Thickness
ABLATION_CONFIGS["demo_sex_thickness"] = _cfg(
    use_hand=True, use_demo=True,
    hand_cols=["Avg_Tissue_thickness_30_60"],
    description="[COMBO] Sex only + Thickness — isolate Sex vs full demo contribution",
)

# 3. The collinear cluster was a surprise winner — does it add to demo_all?
ABLATION_CONFIGS["demo_all_redundant_cluster"] = _cfg(
    use_hand=True, use_demo=True,
    hand_cols=["Avg_NumTissuePixel_30_60", "Avg_Tissue_30_60", "Avg_TissueByTotal_30_60"],
    description="[COMBO] Sex+Age+Smoking + collinear cluster — surprise hand block + best demo",
)

# CNN combos (only if not skipping)
if not SKIP_CNN:
    ABLATION_CONFIGS["cnn_4stat_hand_all_sex"] = _cfg(
        use_cnn=True, use_demo=True, hand_cols=HAND_FEATURE_COLS_ALL,
        description="[COMBO] CNN 4-stat (mean-pool) + All 9 handcrafted + Sex",
    )
    ABLATION_CONFIGS["cnn_4stat_kurtosis_sex"] = _cfg(
        use_cnn=True, use_demo=True, hand_cols=KURTOSIS,
        description="[COMBO] CNN 4-stat (mean-pool) + Kurtosis + Sex — minimal CNN combo",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO-FEATURE OVERRIDE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# The SurvivalDataLoader reads DEMO_FEATURE_COLS at module level.
# For single-demographic ablations we monkey-patch it temporarily.

import cox_survival_analysis as _cox_mod

_DEMO_OVERRIDES = {
    "demo_age_only":      ["Age"],
    "demo_smoking_only":  ["SmokingStatus"],
    "demo_sex_age":       ["Sex", "Age"],
    "demo_all":           ["Sex", "Age", "SmokingStatus"],
    # default (Sex only) — no override needed for demo_sex_only
}


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(configs: dict = None, skip_estimated: bool = False):
    """
    Run the full ablation study across all registered configs.

    Parameters
    ----------
    configs : dict, optional
        Subset of ABLATION_CONFIGS to run. Default = all.
    skip_estimated : bool
        If True, skip configs whose results are already known from prior runs
        (speeds up iteration when adding new configs).
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    print(f"\n{'='*80}")
    print(f"FULL ABLATION STUDY — {len(configs)} configurations")
    print(f"{'='*80}\n")

    with open(CONFIG["kfold_splits_path"], "rb") as f:
        kfold_splits = pickle.load(f)

    all_results   = {}
    summary_rows  = []
    original_demo = _cox_mod.DEMO_FEATURE_COLS[:]

    for config_name, params in configs.items():
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"        {params['description']}")
        print(f"{'='*80}")

        # Apply demo override if needed
        demo_override = _DEMO_OVERRIDES.get(config_name)
        if demo_override:
            _cox_mod.DEMO_FEATURE_COLS = demo_override
            print(f"  [override] DEMO_FEATURE_COLS → {demo_override}")
        else:
            _cox_mod.DEMO_FEATURE_COLS = original_demo

        use_cnn  = params["use_cnn"]
        use_hand = params["use_hand"]
        use_demo = params["use_demo"]
        hand_cols = params.get("hand_feature_cols")
        pen_override = params.get("penalizer")
        
        # New CNN parameters
        pooling_type = params.get("pooling_type")
        cnn_stats = params.get("cnn_stats_to_use")
        use_pca = params.get("use_cnn_pca", False)
        n_pca = params.get("n_pca_components", 3)

        cfg = _make_cox_config(use_cnn, penalizer_override=pen_override,
                                pooling_type=pooling_type, cnn_stats=cnn_stats,
                                use_pca=use_pca, n_pca=n_pca)

        loader = SurvivalDataLoader(cfg)
        df = loader.prepare_full_dataset(
            use_cnn=use_cnn,
            use_hand=use_hand,
            use_demo=use_demo,
            hand_feature_cols=hand_cols,
        )

        analyzer = CoxSurvivalAnalyzer(cfg)
        results  = analyzer.run_cross_validation(
            df, kfold_splits,
            use_cnn=use_cnn,
            use_hand=use_hand,
            use_demo=use_demo,
            experiment_name=config_name,
        )
        all_results[config_name] = results

        if results:
            val_cis  = [r["val_ci"]  for r in results]
            test_cis = [r["test_ci"] for r in results]
            summary_rows.append({
                "Config":            config_name,
                "Description":       params["description"],
                "Mean Val C-index":  f"{np.mean(val_cis):.4f}",
                "Std Val C-index":   f"{np.std(val_cis):.4f}",
                "Mean Test C-index": f"{np.mean(test_cis):.4f}",
                "Std Test C-index":  f"{np.std(test_cis):.4f}",
            })

    # Restore original demo cols
    _cox_mod.DEMO_FEATURE_COLS = original_demo

    # ── Final comparison table ─────────────────────────────────────────────────
    comparison = pd.DataFrame(summary_rows)
    comparison = comparison.sort_values("Mean Test C-index", ascending=False)

    print(f"\n{'='*80}")
    print("ABLATION STUDY — FINAL COMPARISON (sorted by Test C-index)")
    print(f"{'='*80}")
    print(comparison.to_string(index=False))

    out_path = CONFIG["output_dir"] / "ablation_full_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\n✓ Full comparison saved → {out_path}")
    return all_results


def run_recommended_only():
    """Run only the recommended/finalist configs (faster iteration)."""
    recommended = {
        k: v for k, v in ABLATION_CONFIGS.items()
        if k in {
            "demo_sex_only",
            "hand_kurtosis_only",
            "hand_approxvol_only",
            "hand_all_9",
            "hand_kurtosis_sex",
            "hand_top2_sex",
            "hand_all_demo",           # ← recommended best
            "kurtosis_approxvol_thickness_sex",
        }
    }
    return run_ablation_study(configs=recommended)


def run_grid_search_experiment():
    """
    Run grid search for regularization hyperparameters.
    Tests penalizer and l1_ratio combinations to find optimal regularization.
    """
    print(f"\n{'='*80}")
    print("GRID SEARCH EXPERIMENT")
    print(f"{'='*80}\n")
    
    with open(CONFIG["kfold_splits_path"], "rb") as f:
        kfold_splits = pickle.load(f)
    
    # Test on the recommended configuration (all 9 handcrafted + Sex)
    cfg = _make_cox_config(use_cnn=False, penalizer_override=None)
    
    loader = SurvivalDataLoader(cfg)
    df = loader.prepare_full_dataset(
        use_cnn=False,
        use_hand=True,
        use_demo=True,
        hand_feature_cols=HAND_FEATURE_COLS_ALL,
    )
    
    analyzer = CoxSurvivalAnalyzer(cfg)
    
    # Grid search with recommended values
    best_config, grid_df = analyzer.grid_search_regularization(
        df, kfold_splits,
        penalizer_values=[0.01, 0.1, 0.5, 1, 5],
        l1_ratio_values=[0, 0.5, 1],
        use_cnn=False, use_hand=True, use_demo=True
    )
    
    # Save results
    out_path = CONFIG["output_dir"] / "grid_search_results.csv"
    grid_df.to_csv(out_path, index=False)
    print(f"\n✓ Grid search results saved → {out_path}")
    
    return best_config, grid_df


def run_cnn_statistics_ablation():
    """
    Test different CNN statistics combinations (especially without L2Norm).
    """
    print(f"\n{'='*80}")
    print("CNN STATISTICS ABLATION")
    print(f"{'='*80}\n")
    
    with open(CONFIG["kfold_splits_path"], "rb") as f:
        kfold_splits = pickle.load(f)
    
    stats_variants = {
        "all_4_stats": ['mean', 'variance', 'l2norm', 'entropy'],
        "no_l2norm": ['mean', 'variance', 'entropy'],
        "mean_var_only": ['mean', 'variance'],
        "mean_entropy": ['mean', 'entropy'],
    }
    
    results_summary = []
    
    for variant_name, stats_to_use in stats_variants.items():
        print(f"\n{'='*80}")
        print(f"Variant: {variant_name} — stats: {stats_to_use}")
        print(f"{'='*80}")
        
        cfg = _make_cox_config(use_cnn=True, penalizer_override=10.0,
                                cnn_stats=stats_to_use)
        
        loader = SurvivalDataLoader(cfg)
        df = loader.prepare_full_dataset(
            use_cnn=True,
            use_hand=True,
            use_demo=True,
            hand_feature_cols=HAND_FEATURE_COLS_ALL,
        )
        
        analyzer = CoxSurvivalAnalyzer(cfg)
        results = analyzer.run_cross_validation(
            df, kfold_splits,
            use_cnn=True, use_hand=True, use_demo=True,
            experiment_name=f"cnn_stats_{variant_name}"
        )
        
        if results:
            val_cis = [r['val_ci'] for r in results]
            test_cis = [r['test_ci'] for r in results]
            
            results_summary.append({
                'Variant': variant_name,
                'Stats': ', '.join(stats_to_use),
                'N_CNN_features': len(stats_to_use),
                'Mean_Val_CI': f"{np.mean(val_cis):.4f}",
                'Std_Val_CI': f"{np.std(val_cis):.4f}",
                'Mean_Test_CI': f"{np.mean(test_cis):.4f}",
                'Std_Test_CI': f"{np.std(test_cis):.4f}",
            })
    
    # Summary
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values('Mean_Test_CI', ascending=False)
    
    print(f"\n{'='*80}")
    print("CNN STATISTICS ABLATION — SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    out_path = CONFIG["output_dir"] / "cnn_statistics_ablation.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n✓ CNN statistics ablation saved → {out_path}")
    
    return summary_df


def run_cnn_setup_comparison():
    """
    Compare Setup A (mean-pool + 4 stats), Setup B (max-pool + 4 stats), 
    and Setup C (mean-pool + PCA).
    
    Tests with full feature set: All 9 handcrafted + Sex
    """
    print(f"\n{'='*80}")
    print("CNN SETUP COMPARISON: A (mean+stats) vs B (max+stats) vs C (PCA)")
    print(f"{'='*80}\n")
    
    cnn_setups = {
        "cnn_4stat_hand_all_sex": "Setup A (mean-pool + 4 stats)",
        "cnn_maxpool_4stat_hand_all_sex": "Setup B (max-pool + 4 stats)",
        "cnn_pca3_hand_all_sex": "Setup C (mean-pool + PCA-3)",
        "cnn_pca5_hand_all_sex": "Setup C (mean-pool + PCA-5)",
    }
    
    return run_ablation_study(configs={k: ABLATION_CONFIGS[k] for k in cnn_setups.keys()})


def run_cnn_only_comparison():
    """
    Compare CNN-only configurations to isolate CNN signal.
    Tests if CNN adds any predictive power without handcrafted features.
    """
    print(f"\n{'='*80}")
    print("CNN-ONLY COMPARISON: Isolating CNN Signal")
    print(f"{'='*80}\n")
    
    cnn_only_configs = {
        "cnn_4stat_only": "4 stats (mean-pool)",
        "cnn_maxpool_4stat": "4 stats (max-pool)",
        "cnn_pca3_only": "PCA 3 components",
        "cnn_pca5_only": "PCA 5 components",
        "cnn_maxpool_no_l2norm_only": "3 stats no L2Norm (max-pool)",
    }
    
    return run_ablation_study(configs={k: ABLATION_CONFIGS[k] for k in cnn_only_configs.keys()})


def run_pca_sweep():
    """
    Sweep PCA components to find optimal dimensionality.
    Tests 2, 3, 5, 7, 10 components with full feature set.
    """
    print(f"\n{'='*80}")
    print("PCA COMPONENT SWEEP: Finding Optimal Dimensionality")
    print(f"{'='*80}\n")
    
    pca_configs = {
        "cnn_pca2_hand_all_sex": "PCA 2 components",
        "cnn_pca3_hand_all_sex": "PCA 3 components",
        "cnn_pca5_hand_all_sex": "PCA 5 components",
        "cnn_pca7_hand_all_sex": "PCA 7 components",
        "cnn_pca10_hand_all_sex": "PCA 10 components",
    }
    
    return run_ablation_study(configs={k: ABLATION_CONFIGS[k] for k in pca_configs.keys()})


def run_no_l2norm_comparison():
    """
    Compare configurations with and without L2Norm.
    Tests if dropping L2Norm improves stability.
    """
    print(f"\n{'='*80}")
    print("L2NORM COMPARISON: Test Stability Without L2Norm")
    print(f"{'='*80}\n")
    
    l2norm_configs = {
        "cnn_4stat_hand_all_sex": "With L2Norm (mean-pool, 4 stats)",
        "cnn_3stat_no_l2_hand_all_sex": "Without L2Norm (mean-pool, 3 stats)",
        "cnn_maxpool_4stat_hand_all_sex": "With L2Norm (max-pool, 4 stats)",
        "cnn_maxpool_3stat_no_l2_hand_all_sex": "Without L2Norm (max-pool, 3 stats)",
    }
    
    return run_ablation_study(configs={k: ABLATION_CONFIGS[k] for k in l2norm_configs.keys()})


def run_penalizer_tuning_4stat():
    """
    Fine-tune penalizer for mean-pool 4-stat configurations.
    Tests dense grid of penalizer values to find optimal L2 regularization
    that reduces variance while maintaining C-index.
    
    Focus: Stability over performance.
    """
    print(f"\n{'='*80}")
    print("PENALIZER TUNING: Mean-Pool 4-Stat (Stability Focus)")
    print(f"{'='*80}\n")
    
    with open(CONFIG["kfold_splits_path"], "rb") as f:
        kfold_splits = pickle.load(f)
    
    # Test configurations: CNN-only, CNN+Sex, CNN+Hand+Sex
    test_configs = [
        ("cnn_4stat_only", False, False),
        ("cnn_4stat_sex", False, True),
        ("cnn_4stat_hand_all_sex", True, True),
    ]
    
    # Dense grid of penalizer values (focus on higher values for stability)
    penalizer_values = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30]
    
    all_results = {}
    
    for config_name, use_hand, use_demo in test_configs:
        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"{'='*80}")
        
        config_results = []
        hand_cols = HAND_FEATURE_COLS_ALL if use_hand else None
        
        for pen in penalizer_values:
            print(f"\n{'-'*80}")
            print(f"Testing penalizer = {pen}")
            print(f"{'-'*80}")
            
            cfg = _make_cox_config(use_cnn=True, penalizer_override=pen)
            
            loader = SurvivalDataLoader(cfg)
            df = loader.prepare_full_dataset(
                use_cnn=True,
                use_hand=use_hand,
                use_demo=use_demo,
                hand_feature_cols=hand_cols,
            )
            
            analyzer = CoxSurvivalAnalyzer(cfg)
            results = analyzer.run_cross_validation(
                df, kfold_splits,
                use_cnn=True, use_hand=use_hand, use_demo=use_demo,
                experiment_name=f"{config_name}_pen{pen}"
            )
            
            if results:
                val_cis = [r['val_ci'] for r in results]
                test_cis = [r['test_ci'] for r in results]
                
                config_results.append({
                    'config': config_name,
                    'penalizer': pen,
                    'mean_val_ci': np.mean(val_cis),
                    'std_val_ci': np.std(val_cis),
                    'mean_test_ci': np.mean(test_cis),
                    'std_test_ci': np.std(test_cis),
                    'total_variance': np.std(val_cis) + np.std(test_cis),
                })
        
        all_results[config_name] = pd.DataFrame(config_results)
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print("PENALIZER TUNING RESULTS")
    print(f"{'='*80}\n")
    
    for config_name, results_df in all_results.items():
        print(f"\n{config_name}:")
        print(results_df.to_string(index=False))
        
        # Find best penalizer (min variance while maintaining reasonable C-index)
        # Strategy: minimize variance, require val_ci > 0.55
        valid = results_df[results_df['mean_val_ci'] > 0.55]
        if len(valid) > 0:
            best_idx = valid['total_variance'].idxmin()
            best = results_df.loc[best_idx]
            print(f"\n  ✓ Recommended penalizer: {best['penalizer']}")
            print(f"    Val C-index: {best['mean_val_ci']:.4f} ± {best['std_val_ci']:.4f}")
            print(f"    Test C-index: {best['mean_test_ci']:.4f} ± {best['std_test_ci']:.4f}")
        else:
            print(f"\n  ⚠ All penalizers resulted in Val C-index < 0.55")
    
    # Save combined results
    combined_df = pd.concat([df.assign(config=name) for name, df in all_results.items()])
    out_path = CONFIG["output_dir"] / "penalizer_tuning_4stat.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"\n✓ Penalizer tuning results saved → {out_path}")
    
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full Cox PH Ablation Study — IPF Survival"
    )
    parser.add_argument(
        "--mode", 
        choices=["full", "recommended", "single", "gridsearch", 
                "cnn-stats", "cnn-setup-compare", "cnn-only", "pca-sweep", 
                "no-l2norm", "tune-penalizer"], 
        default="recommended",
        help="""
        full=all configs, recommended=finalist configs, single=one config by name,
        gridsearch=find optimal penalizer/l1_ratio, 
        cnn-stats=test CNN statistics variants,
        cnn-setup-compare=compare Setup A/B/C (mean vs max vs PCA),
        cnn-only=isolate CNN signal (no handcrafted features),
        pca-sweep=find optimal PCA components (2,3,5,7,10),
        no-l2norm=compare with/without L2Norm statistic,
        tune-penalizer=fine-tune penalizer for mean-pool 4-stat (stability focus)
        """,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config name for --mode single (e.g. hand_all_demo)",
    )
    parser.add_argument(
        "--skip-cnn", action="store_true",
        help="Skip CNN-containing configs (faster, no GPU needed)",
    )
    args = parser.parse_args()

    if args.skip_cnn:
        # Remove CNN configs from registry
        for k in list(ABLATION_CONFIGS.keys()):
            if ABLATION_CONFIGS[k]["use_cnn"]:
                del ABLATION_CONFIGS[k]
        print(f"  [--skip-cnn] Removed CNN configs. Remaining: {len(ABLATION_CONFIGS)}")

    if args.mode == "full":
        run_ablation_study()
    elif args.mode == "recommended":
        run_recommended_only()
    elif args.mode == "single":
        if args.config not in ABLATION_CONFIGS:
            print(f"Unknown config '{args.config}'. Available:")
            for k in ABLATION_CONFIGS:
                print(f"  {k}")
        else:
            run_ablation_study(configs={args.config: ABLATION_CONFIGS[args.config]})
    elif args.mode == "gridsearch":
        run_grid_search_experiment()
    elif args.mode == "cnn-stats":
        run_cnn_statistics_ablation()
    elif args.mode == "cnn-setup-compare":
        run_cnn_setup_comparison()
    elif args.mode == "cnn-only":
        run_cnn_only_comparison()
    elif args.mode == "pca-sweep":
        run_pca_sweep()
    elif args.mode == "no-l2norm":
        run_no_l2norm_comparison()
    elif args.mode == "tune-penalizer":
        run_penalizer_tuning_4stat()