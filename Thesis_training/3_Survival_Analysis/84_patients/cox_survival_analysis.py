"""
Cox Proportional Hazards Survival Analysis for IPF Progression Prediction
Updated based on diagnostic findings:
  - Diag 2: Only Kurtosis, ApproxVol, Thickness, Sex have meaningful signal
  - Diag 5: Avg_NumTissuePixel ↔ Avg_Tissue ↔ Avg_TissueByTotal are redundant (r>0.85)
  - Diag 7: Val C-index peaks at k=1-2 features; >3 features = overfitting
  - Diag 3: Best penalizer = 0.5 with pure Ridge (l1_ratio=0.0)
  - CNN 4-stat aggregation adds noise, not signal → dropped from default config

Fixes applied:
  - _plot_km / _plot_risk_dist: reset_index() on val_df and val_risk to fix
    index-mismatch crash when lifelines receives an empty Series
  - fit_fold: allowed_hand now derived from the columns actually present in
    train_df, not hardcoded to HAND_FEATURE_COLS, so ablation configs that
    pass HAND_FEATURE_COLS_ALL work correctly
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy as scipy_entropy, spearmanr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

sys.path.append(str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS  (evidence-based from diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

HAND_FEATURE_COLS_ALL = [
    'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60',
    'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
]

# REDUCED set — diagnostics-driven selection:
#   Kurtosis   C=0.583  (best univariate)
#   ApproxVol  C=0.571
#   Thickness  C=0.527
#   Dropped: Avg_NumTissuePixel/Avg_Tissue/Avg_TissueByTotal (r>0.85 cluster)
#            Mean_30_60 (C=0.514), Avg_TissueByLung (C=0.518), Skew (sign flips)
HAND_FEATURE_COLS = [
    'Kurtosis_30_60',
    'ApproxVol_30_60',
    'Avg_Tissue_thickness_30_60',
]

# Sex retained (C=0.536, p=0.025 in fold 4).
# SmokingStatus dropped (only inverted feature, C=0.493).
# Age dropped (C=0.519, sign unstable across folds).
DEMO_FEATURE_COLS = ['Sex']


CONFIG = {
    "survival_data_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\ground_truth_survival.csv"),
    "ct_scan_path":        Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path":   Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\Folds\kfold_splits_survival_analysis.pkl"),
    "train_csv_path":      Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "output_dir":          Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\results_flipped_config2"),

    'backbone':    'resnet50',
    'pooling_type':'mean',  # 'mean' or 'max' — pooling across slices
    'image_size':  (224, 224),

    'penalizer': 0.5,
    'l1_ratio':  0.0,   # pure Ridge — L1 zeroes everything with weak features
    'cnn_stats_to_use': ['mean', 'variance', 'l2norm', 'entropy'],  # options: mean, variance, l2norm, entropy
    'use_cnn_pca': False,  # If True, use PCA instead of statistics
    'n_pca_components': 3,  # Number of PCA components if use_cnn_pca=True
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

class SurvivalDataLoader:

    def __init__(self, config):
        self.config = config
        self.cnn_extractor = None
        self.cnn_features_cache = None

    def load_survival_data(self):
        df = pd.read_csv(self.config['survival_data_path'])
        if 'PatientID' in df.columns:
            df.rename(columns={'PatientID': 'Patient'}, inplace=True)
        print(f"\n{'='*70}\nLOADED SURVIVAL DATA\n{'='*70}")
        print(f"Total patients : {len(df)}")
        print(f"Events         : {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
        print(f"Censored       : {(df['event']==0).sum()} ({(df['event']==0).mean()*100:.1f}%)")
        print(f"Time range     : {df['time'].min():.0f}–{df['time'].max():.0f} weeks  (mean {df['time'].mean():.1f})")
        return df

    def load_handcrafted_features(self, patient_ids, feature_cols=None):
        if feature_cols is None:
            feature_cols = HAND_FEATURE_COLS
        df = pd.read_csv(self.config['patient_features_path'])
        df = df[df['Patient'].isin(patient_ids)]
        available = [c for c in feature_cols if c in df.columns]
        missing   = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"  ⚠  Missing columns (skipped): {missing}")
        df = df[['Patient'] + available]
        print(f"\nLoaded {len(available)} handcrafted features: {available}")
        return df

    def load_demographics(self, patient_ids):
        df = pd.read_csv(self.config['train_csv_path'])
        df = df[df['Patient'].isin(patient_ids)]
        keep = ['Patient'] + [c for c in DEMO_FEATURE_COLS if c in df.columns]
        df = df[keep].drop_duplicates(subset=['Patient'])
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"Loaded demographics: {[c for c in DEMO_FEATURE_COLS if c in df.columns]}")
        return df

    def extract_cnn_features(self, patient_ids):
        if self.cnn_features_cache is not None:
            cached = self.cnn_features_cache[self.cnn_features_cache['Patient'].isin(patient_ids)]
            print(f"Using cached CNN features: {len(cached)} patients")
            return cached

        print(f"\n{'='*70}\nEXTRACTING CNN FEATURES\n{'='*70}")
        import glob, os
        patient_data = {}
        for pid in patient_ids:
            folder = os.path.join(self.config['ct_scan_path'], pid)
            if not os.path.exists(folder):
                continue
            npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            if npy_files:
                patient_data[pid] = {'slices': npy_files, 'n_slices': len(npy_files),
                                     'gt_has_progressed': 0}

        print(f"Found {len(patient_data)} patients with CT scans")
        if self.cnn_extractor is None:
            from utilities import CNNFeatureExtractor
            self.cnn_extractor = CNNFeatureExtractor(model_name=self.config['backbone'])

        slice_df = self.cnn_extractor.extract_features_patient_grouping(
            patient_data=patient_data, patients_per_batch=4, save_path=None)
        cnn_cols = [c for c in slice_df.columns if c.startswith('cnn_feature_')]
        
        # Apply pooling across slices
        pooling_type = self.config.get('pooling_type', 'mean')
        records, valid = [], []
        for pid in patient_ids:
            slices = slice_df[slice_df['patient_id'] == pid]
            if len(slices) > 0:
                if pooling_type == 'max':
                    pooled = slices[cnn_cols].max(axis=0).values  # Max pooling
                else:
                    pooled = slices[cnn_cols].mean(axis=0).values  # Mean pooling (default)
                records.append(pooled)
                valid.append(pid)
        
        df = pd.DataFrame(np.array(records), columns=cnn_cols)
        df.insert(0, 'Patient', valid)
        self.cnn_features_cache = df
        print(f"✓ Extracted CNN features: {len(valid)} patients, {len(cnn_cols)} dims")
        print(f"  Pooling strategy: {pooling_type}")
        return df

    def prepare_full_dataset(self, use_cnn=False, use_hand=True, use_demo=True,
                              hand_feature_cols=None):
        print(f"\n{'='*70}\nPREPARING DATASET\n{'='*70}")
        print(f"Features: CNN={use_cnn}, Handcrafted={use_hand}, Demographics={use_demo}")

        survival_df = self.load_survival_data()
        patient_ids = survival_df['Patient'].tolist()
        df = survival_df.copy()

        if use_cnn:
            cnn_df = self.extract_cnn_features(patient_ids)
            df = df.merge(cnn_df, on='Patient', how='left')
            print("✓ Added CNN features")

        if use_hand:
            hand_df = self.load_handcrafted_features(patient_ids,
                                                      feature_cols=hand_feature_cols)
            df = df.merge(hand_df, on='Patient', how='left')
            print("✓ Added handcrafted features")

        if use_demo:
            demo_df = self.load_demographics(patient_ids)
            df = df.merge(demo_df, on='Patient', how='left')
            print("✓ Added demographics")

        n_before = len(df)
        df = df.dropna()
        print(f"\nPatients: {n_before} → {len(df)} after dropna  (dropped {n_before-len(df)})")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# COX ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class CoxSurvivalAnalyzer:

    def __init__(self, config):
        self.config = config

    # ── Demographics preprocessing ────────────────────────────────────────────
    
    def preprocess_demographics(self, train_df, val_df, test_df, demo_cols):
        new_cols = []

        if 'Age' in demo_cols and 'Age' in train_df.columns:
            scaler = StandardScaler()
            train_df['Age_normalized'] = scaler.fit_transform(train_df[['Age']].values).flatten()
            val_df['Age_normalized']   = scaler.transform(val_df[['Age']].values).flatten()
            test_df['Age_normalized']  = scaler.transform(test_df[['Age']].values).flatten()
            new_cols.append('Age_normalized')

        if 'Sex' in demo_cols and 'Sex' in train_df.columns:
            for split in [train_df, val_df, test_df]:
                split['Sex_encoded'] = split['Sex'].map({0: -1, 1: 1})
            new_cols.append('Sex_encoded')

        if 'SmokingStatus' in demo_cols and 'SmokingStatus' in train_df.columns:
            categories = sorted(train_df['SmokingStatus'].dropna().unique())
            for cat in categories:
                col_name = f'Smoking_{cat}'
                for split in [train_df, val_df, test_df]:
                    split[col_name] = (split['SmokingStatus'] == cat).astype(float)
                new_cols.append(col_name)
            # Centre each dummy on train mean so Ridge penalises deviations from mean
            for col_name in [f'Smoking_{c}' for c in categories]:
                train_mean = train_df[col_name].mean()
                for split in [train_df, val_df, test_df]:
                    split[col_name] = split[col_name] - train_mean

        if not new_cols:
            print(f'  ⚠  preprocess_demographics: no columns found — demo_cols={demo_cols}')

        return train_df, val_df, test_df, new_cols

    # ── CNN → statistics ──────────────────────────────────────────────────────
    @staticmethod
    def compute_cnn_statistics(cnn_df, stats_to_use=None):
        """
        Compute statistical aggregations from CNN features.
        
        Args:
            cnn_df: DataFrame with CNN features
            stats_to_use: list of stats to compute. Options: 'mean', 'variance', 'l2norm', 'entropy'
                         If None, uses all 4.
        """
        if stats_to_use is None:
            stats_to_use = ['mean', 'variance', 'l2norm', 'entropy']
        
        stats_to_use = [s.lower() for s in stats_to_use]
        arr = cnn_df.values
        rows = []
        
        for v in arr:
            row_stats = {}
            
            if 'mean' in stats_to_use:
                row_stats['CNN_Mean'] = np.mean(v)
            
            if 'variance' in stats_to_use:
                row_stats['CNN_Variance'] = np.var(v)
            
            if 'l2norm' in stats_to_use:
                row_stats['CNN_L2Norm'] = np.linalg.norm(v)
            
            if 'entropy' in stats_to_use:
                shifted = v - v.min() + 1e-10
                probs = shifted / shifted.sum()
                row_stats['CNN_Entropy'] = scipy_entropy(probs)
            
            rows.append(row_stats)
        
        return pd.DataFrame(rows, index=cnn_df.index)

    # ── Single fold ───────────────────────────────────────────────────────────
    def fit_fold(self, train_df, val_df, test_df, fold_num):
        print(f"\n{'='*70}\nFOLD {fold_num}\n{'='*70}")
        print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

        all_cols = [c for c in train_df.columns if c not in ('Patient', 'event', 'time')]
        cnn_cols  = [c for c in all_cols if c.startswith('cnn_feature_')]
        demo_cols = [c for c in all_cols if c in DEMO_FEATURE_COLS]

        # ── FIX: derive allowed hand cols from what's actually in the DataFrame,
        #   not from the global HAND_FEATURE_COLS constant.  This allows ablation
        #   configs that pass HAND_FEATURE_COLS_ALL to work correctly.
        all_hand_universe = set(HAND_FEATURE_COLS_ALL)
        hand_cols = [c for c in all_cols
                     if c in all_hand_universe and c not in cnn_cols and c not in demo_cols]

        print(f"\nFeatures going in: CNN={len(cnn_cols)}  Hand={len(hand_cols)}  Demo={len(demo_cols)}")

        # 1. Demographics
        if demo_cols:
            train_df, val_df, test_df, new_demo_cols = self.preprocess_demographics(
                train_df, val_df, test_df, demo_cols)
        else:
            new_demo_cols = []

        # 2. Handcrafted → StandardScaler (fit on train only)
        if hand_cols:
            scaler_hand = StandardScaler()
            tr_h = pd.DataFrame(scaler_hand.fit_transform(train_df[hand_cols]),
                                 columns=hand_cols, index=train_df.index)
            va_h = pd.DataFrame(scaler_hand.transform(val_df[hand_cols]),
                                 columns=hand_cols, index=val_df.index)
            te_h = pd.DataFrame(scaler_hand.transform(test_df[hand_cols]),
                                 columns=hand_cols, index=test_df.index)
        else:
            tr_h = va_h = te_h = pd.DataFrame()
            scaler_hand = None

        # 3. CNN → statistics OR PCA → StandardScaler
        if cnn_cols:
            use_pca = self.config.get('use_cnn_pca', False)
            
            if use_pca:
                # PCA dimensionality reduction
                n_components = self.config.get('n_pca_components', 3)
                print(f"  Using PCA: {len(cnn_cols)} → {n_components} components")
                
                pca = PCA(n_components=n_components)
                tr_pca = pca.fit_transform(train_df[cnn_cols])
                va_pca = pca.transform(val_df[cnn_cols])
                te_pca = pca.transform(test_df[cnn_cols])
                
                pca_cols = [f'CNN_PC{i+1}' for i in range(n_components)]
                tr_s = pd.DataFrame(tr_pca, columns=pca_cols, index=train_df.index)
                va_s = pd.DataFrame(va_pca, columns=pca_cols, index=val_df.index)
                te_s = pd.DataFrame(te_pca, columns=pca_cols, index=test_df.index)
                
                print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
                scaler_cnn = pca  # Store PCA object for reference
                
            else:
                # Statistical aggregation
                stats_to_use = self.config.get('cnn_stats_to_use', ['mean', 'variance', 'l2norm', 'entropy'])
                tr_s = self.compute_cnn_statistics(train_df[cnn_cols], stats_to_use=stats_to_use)
                va_s = self.compute_cnn_statistics(val_df[cnn_cols], stats_to_use=stats_to_use)
                te_s = self.compute_cnn_statistics(test_df[cnn_cols], stats_to_use=stats_to_use)
                print(f"  Using CNN statistics: {stats_to_use} → {len(tr_s.columns)} features")
                scaler_cnn = None
            
            # CRITICAL: Standardize CNN features (PCA or statistics)
            scaler_cnn_std = StandardScaler()
            tr_c = pd.DataFrame(scaler_cnn_std.fit_transform(tr_s),
                                 columns=tr_s.columns, index=tr_s.index)
            va_c = pd.DataFrame(scaler_cnn_std.transform(va_s),
                                 columns=va_s.columns, index=va_s.index)
            te_c = pd.DataFrame(scaler_cnn_std.transform(te_s),
                                 columns=te_s.columns, index=te_s.index)
            cnn_stat_cols = tr_c.columns.tolist()
        else:
            tr_c = va_c = te_c = pd.DataFrame()
            cnn_stat_cols = []
            scaler_cnn = None
            scaler_cnn_std = None

        # 4. Combine features
        tr_demo = train_df[new_demo_cols] if new_demo_cols else pd.DataFrame(index=train_df.index)
        va_demo = val_df[new_demo_cols]   if new_demo_cols else pd.DataFrame(index=val_df.index)
        te_demo = test_df[new_demo_cols]  if new_demo_cols else pd.DataFrame(index=test_df.index)

        parts_tr = [p for p in [tr_c, tr_h, tr_demo] if not p.empty]
        parts_va = [p for p in [va_c, va_h, va_demo] if not p.empty]
        parts_te = [p for p in [te_c, te_h, te_demo] if not p.empty]

        if not parts_tr:
            print(f"  ❌ Fold {fold_num}: no features after preprocessing — "
                "check columns exist in the merged DataFrame.")
            return None

        train_X = pd.concat(parts_tr, axis=1)
        val_X   = pd.concat(parts_va, axis=1) if parts_va else pd.DataFrame(index=val_df.index)
        test_X  = pd.concat(parts_te, axis=1) if parts_te else pd.DataFrame(index=test_df.index)

        total_features = len(train_X.columns)
        n_events = int(train_df['event'].sum())
        epv = n_events / total_features if total_features > 0 else float('inf')

        print(f"\nFinal feature count: {total_features}  "
              f"(CNN-stats={len(cnn_stat_cols)}, hand={len(hand_cols)}, demo={len(new_demo_cols)})")
        if epv < 10:
            print(f"  ⚠  Events-per-variable = {epv:.1f} (< 10)")
        else:
            print(f"  ✓  Events-per-variable = {epv:.1f}")

        # 5. Build Cox DataFrames (reset index so time/event assignment is safe)
        def to_cox(X, src):
            X = X.reset_index(drop=True).copy()
            X['time']  = src['time'].values
            X['event'] = src['event'].values
            return X

        tr_cox = to_cox(train_X, train_df)
        va_cox = to_cox(val_X,   val_df)
        te_cox = to_cox(test_X,  test_df)

        # 6. Fit Cox model
        print(f"\nFitting CoxPH  penalizer={self.config['penalizer']}  "
              f"l1_ratio={self.config['l1_ratio']}")
        cph = CoxPHFitter(penalizer=self.config['penalizer'],
                          l1_ratio=self.config['l1_ratio'])
        try:
            cph.fit(tr_cox, duration_col='time', event_col='event', show_progress=False)
        except Exception as e:
            print(f"❌ Fit failed: {e}")
            return None

        nonzero = (cph.params_.abs() > 1e-6).sum()
        print(f"Non-zero coefficients: {nonzero}/{len(cph.params_)}")

        # 7. Evaluate
        def ci_score(cox_df, model):
            X = cox_df.drop(['time', 'event'], axis=1)
            return concordance_index(cox_df['time'],
                                     -model.predict_log_partial_hazard(X),
                                     cox_df['event'])

        train_ci = cph.concordance_index_
        val_ci   = ci_score(va_cox, cph)
        test_ci  = ci_score(te_cox, cph)

        print(f"\nTrain C-index : {train_ci:.4f}")
        print(f"Val   C-index : {val_ci:.4f}")
        print(f"Test  C-index : {test_ci:.4f}")
        print(f"Overfit gap   : {train_ci - val_ci:.4f}  (target < 0.10)")

        summary = cph.summary[['coef', 'exp(coef)', 'p']].sort_values('coef')
        print(f"\nCoefficients (sorted):\n{summary.to_string()}")

        # Store risk scores with clean 0-based index (critical for plotting)
        tr_feat = tr_cox.drop(['time', 'event'], axis=1)
        va_feat = va_cox.drop(['time', 'event'], axis=1)
        te_feat = te_cox.drop(['time', 'event'], axis=1)

        train_risk = cph.predict_partial_hazard(tr_feat).reset_index(drop=True)
        val_risk = cph.predict_partial_hazard(va_feat).reset_index(drop=True)
        test_risk = cph.predict_partial_hazard(te_feat).reset_index(drop=True)
        
        corr, _ = spearmanr(val_risk, val_df['time'])
        print(f"Val risk-time Spearman: {corr:.4f}")

        return {
            'fold':      fold_num,
            'model':     cph,
            'scaler_hand': scaler_hand,
            'scaler_cnn':  scaler_cnn,
            'scaler_cnn_std': scaler_cnn_std,  # StandardScaler applied to CNN features
            'train_ci':  train_ci,
            'val_ci':    val_ci,
            'test_ci':   test_ci,
            # DataFrames already reset_index'd via to_cox()
            'train_df':  train_df.reset_index(drop=True),
            'val_df':    val_df.reset_index(drop=True),
            'test_df':   test_df.reset_index(drop=True),
            # Risk Series: force 0-based index to match val_df
            'train_risk': train_risk,
            'val_risk':   val_risk,
            'test_risk':  test_risk,
            'feature_cols': train_X.columns.tolist(),
        }

    # ── Plotting ──────────────────────────────────────────────────────────────
    def _plot_km(self, fold_results, out_dir):
        fold_num = fold_results['fold']
        # Both already have reset 0-based index (guaranteed by fit_fold)
        val_df   = fold_results['val_df']
        val_risk = fold_results['val_risk']
        median   = val_risk.median()

        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(9, 5))
        for mask, label, color in [
            (val_risk >= median, 'High risk', 'crimson'),
            (val_risk <  median, 'Low risk',  'steelblue'),
        ]:
            # Use .values to avoid any residual index issues
            times  = val_df.loc[mask.values, 'time'].values
            events = val_df.loc[mask.values, 'event'].values
            if len(times) == 0:
                print(f"  ⚠  No patients in '{label}' group — skipping KM for fold {fold_num}")
                continue
            kmf.fit(times, events, label=label)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Progression-free probability')
        ax.set_title(f'Fold {fold_num} — Kaplan-Meier by risk group')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'fold_{fold_num}_kaplan_meier.png', dpi=200)
        plt.close(fig)

    def _plot_hr(self, fold_results, out_dir):
        fold_num = fold_results['fold']
        cph = fold_results['model']
        summary = cph.summary[['exp(coef)', 'exp(coef) lower 95%',
                                'exp(coef) upper 95%', 'p']]
        summary = summary.sort_values('exp(coef)', ascending=False)

        fig, ax = plt.subplots(figsize=(9, max(4, len(summary)*0.5 + 1)))
        y  = np.arange(len(summary))
        hr = summary['exp(coef)'].values
        lo = summary['exp(coef) lower 95%'].values
        hi = summary['exp(coef) upper 95%'].values
        ax.errorbar(hr, y, xerr=[hr - lo, hi - hr],
                    fmt='o', color='steelblue', capsize=4)
        ax.axvline(1, color='red', ls='--', alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(summary.index)
        ax.set_xlabel('Hazard Ratio')
        ax.set_title(f'Fold {fold_num} — Hazard Ratios')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(out_dir / f'fold_{fold_num}_hazard_ratios.png', dpi=200)
        plt.close(fig)

    def _plot_risk_dist(self, fold_results, out_dir):
        fold_num = fold_results['fold']
        val_df   = fold_results['val_df']
        val_risk = fold_results['val_risk']

        # Use .values so boolean indexing is index-agnostic
        ev = val_df['event'].values == 1

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(val_risk.values[~ev], bins=15, alpha=0.6,
                color='steelblue', label='Censored')
        ax.hist(val_risk.values[ev],  bins=15, alpha=0.6,
                color='crimson',   label='Event')
        ax.set_xlabel('Risk score')
        ax.set_ylabel('Count')
        ax.set_title(f'Fold {fold_num} — Risk distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'fold_{fold_num}_risk_dist.png', dpi=200)
        plt.close(fig)

    # ── Cross-validation ──────────────────────────────────────────────────────
    def run_cross_validation(self, df, kfold_splits,
                              use_cnn=False, use_hand=True, use_demo=True,
                              experiment_name=None):
        print(f"\n{'='*80}\nCOX SURVIVAL ANALYSIS — CROSS VALIDATION\n{'='*80}")

        tag = experiment_name or (
            f"cnn{int(use_cnn)}_hand{int(use_hand)}_demo{int(use_demo)}"
            f"_pen{self.config['penalizer']}_l1{self.config['l1_ratio']}"
        )
        out_root = self.config['output_dir'] / tag
        out_root.mkdir(parents=True, exist_ok=True)

        all_results = []
        for fold_num in range(1, 6):
            split    = kfold_splits[f'fold_{fold_num}']
            train_df = df[df['Patient'].isin(split['train'])].reset_index(drop=True)
            val_df   = df[df['Patient'].isin(split['val'])].reset_index(drop=True)
            test_df  = df[df['Patient'].isin(
                split.get('test', split['val']))].reset_index(drop=True)

            res = self.fit_fold(train_df, val_df, test_df, fold_num)
            if res is None:
                continue

            fold_dir = out_root / f'fold_{fold_num}'
            fold_dir.mkdir(exist_ok=True)
            self._plot_km(res, fold_dir)
            self._plot_hr(res, fold_dir)
            self._plot_risk_dist(res, fold_dir)
            all_results.append(res)

        self._save_summary(all_results, out_root)
        return all_results

    def grid_search_regularization(self, df, kfold_splits, 
                                     penalizer_values=[0.01, 0.1, 0.5, 1, 5],
                                     l1_ratio_values=[0, 0.5, 1],
                                     use_cnn=False, use_hand=True, use_demo=True):
        """
        Grid search over penalizer and l1_ratio to find optimal regularization.
        
        Returns best config and results DataFrame.
        """
        print(f"\n{'='*80}\nGRID SEARCH: REGULARIZATION HYPERPARAMETERS\n{'='*80}")
        print(f"Penalizer values: {penalizer_values}")
        print(f"L1 ratio values: {l1_ratio_values}")
        
        grid_results = []
        
        for pen in penalizer_values:
            for l1 in l1_ratio_values:
                print(f"\n{'─'*80}")
                print(f"Testing: penalizer={pen}, l1_ratio={l1}")
                print(f"{'─'*80}")
                
                # Update config
                self.config['penalizer'] = pen
                self.config['l1_ratio'] = l1
                
                # Run CV
                results = self.run_cross_validation(
                    df, kfold_splits, 
                    use_cnn=use_cnn, use_hand=use_hand, use_demo=use_demo,
                    experiment_name=f"grid_pen{pen}_l1{l1}"
                )
                
                if results:
                    val_cis = [r['val_ci'] for r in results]
                    test_cis = [r['test_ci'] for r in results]
                    
                    grid_results.append({
                        'penalizer': pen,
                        'l1_ratio': l1,
                        'mean_val_ci': np.mean(val_cis),
                        'std_val_ci': np.std(val_cis),
                        'mean_test_ci': np.mean(test_cis),
                        'std_test_ci': np.std(test_cis),
                    })
        
        # Create results DataFrame
        grid_df = pd.DataFrame(grid_results)
        grid_df = grid_df.sort_values('mean_val_ci', ascending=False)
        
        print(f"\n{'='*80}\nGRID SEARCH RESULTS\n{'='*80}")
        print(grid_df.to_string(index=False))
        
        # Find best config
        best_idx = grid_df['mean_val_ci'].idxmax()
        best_config = grid_df.iloc[best_idx]
        
        print(f"\n{'='*80}\nBEST CONFIGURATION (by validation C-index)\n{'='*80}")
        print(f"Penalizer: {best_config['penalizer']}")
        print(f"L1 ratio:  {best_config['l1_ratio']}")
        print(f"Val C-index:  {best_config['mean_val_ci']:.4f} ± {best_config['std_val_ci']:.4f}")
        print(f"Test C-index: {best_config['mean_test_ci']:.4f} ± {best_config['std_test_ci']:.4f}")
        
        return best_config, grid_df
    
    def _save_summary(self, all_results, out_dir):
        if not all_results:
            print("⚠  No successful folds.")
            return

        rows = [{
            'Fold':          r['fold'],
            'Train C-index': r['train_ci'],
            'Val C-index':   r['val_ci'],
            'Test C-index':  r['test_ci'],
            'Overfit gap':   r['train_ci'] - r['val_ci'],
            'N_features':    len(r['feature_cols']),
        } for r in all_results]

        df = pd.DataFrame(rows)
        mean_row = df.mean(numeric_only=True).to_dict(); mean_row['Fold'] = 'Mean'
        std_row  = df.std(numeric_only=True).to_dict();  std_row['Fold']  = 'Std'
        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        df.to_csv(out_dir / 'cross_validation_summary.csv', index=False)

        print(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY\n{'='*70}")
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print(f"\n✓ Results → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*80}\nCOX PH SURVIVAL ANALYSIS\n{'='*80}")

    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"✓ Loaded {len(kfold_splits)} folds")

    loader = SurvivalDataLoader(CONFIG)
    df = loader.prepare_full_dataset(
        use_cnn=False,
        use_hand=True,
        use_demo=True,
        hand_feature_cols=HAND_FEATURE_COLS,
    )

    analyzer = CoxSurvivalAnalyzer(CONFIG)
    results  = analyzer.run_cross_validation(
        df, kfold_splits,
        use_cnn=False, use_hand=True, use_demo=True,
        experiment_name='hand3_demo1_pen0.5_ridge',
    )

    print(f"\n{'='*80}\n✓ DONE\n{'='*80}")


if __name__ == '__main__':
    main()