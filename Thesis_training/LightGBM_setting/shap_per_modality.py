# IMPROVED: LightGBM with CNN Embeddings + Hand-crafted + Demographics
# Key improvements:
# 1. Better LightGBM regularization for small datasets
# 2. Multiple pooling strategies (max + mean + std)
# 3. Adaptive PCA components based on variance
# 4. Better handling of failed folds
# 5. Diagnostic visualization
# 6. ABLATION STUDY: Test different feature combinations
# 7. SHAP EXPLAINABILITY: Group-level and individual-level SHAP analysis

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import random
import sys
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n{'='*70}\n🔒 RANDOM SEED SET TO: {seed}\n{'='*70}\n")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "npy_root": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),

    'cnn_model': 'resnet50',
    'pretrained': True,
    'batch_size': 32,
    'num_workers': 4,
    'pooling_methods': 'max',

    'pca_variance_threshold': 0.95,
    'pca_max_components': 50,
    'pca_whiten': False,

    'base_seed': 42,
    'n_repeats': 1,

    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 3,
        'max_depth': 2,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 15,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.05,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,
        'class_weight': 'balanced'
    },
    'num_boost_round': 1000,
    'early_stopping_rounds': 100,
    'normalization_type': 'standard',
}

HAND_FEATURE_COLS = [
    'ApproxVol_30_60',
    'Avg_NumTissuePixel_30_60',
    'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60',
    'Avg_TissueByTotal_30_60',
    'Avg_TissueByLung_30_60',
    'Mean_30_60',
    'Skew_30_60',
    'Kurtosis_30_60'
]
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']
FVC_BASELINE_COL = 'baselinefvc'

# Human-readable labels for hand-crafted features
HAND_FEATURE_LABELS = {
    'ApproxVol_30_60':             'Approx. Volume',
    'Avg_NumTissuePixel_30_60':    'Avg. Tissue Pixels',
    'Avg_Tissue_30_60':            'Avg. Tissue HU',
    'Avg_Tissue_thickness_30_60':  'Avg. Tissue Thickness',
    'Avg_TissueByTotal_30_60':     'Tissue/Total Ratio',
    'Avg_TissueByLung_30_60':      'Tissue/Lung Ratio',
    'Mean_30_60':                  'Mean HU',
    'Skew_30_60':                  'HU Skewness',
    'Kurtosis_30_60':              'HU Kurtosis',
}


# =============================================================================
# ABLATION STUDY CONFIGURATIONS
# =============================================================================

ABLATION_CONFIGS = {
    "hand_only": {
        "cnn": None, "pca": False, "hand": True, "demo": False, "fvc_baseline": False,
        "description": "Hand-crafted features only (Clinical Baseline)"
    },
    "hand_demo": {
        "cnn": None, "pca": False, "hand": True, "demo": True, "fvc_baseline": False,
        "description": "Hand-crafted + Demographics (Clinical Baseline)"
    },
    "cnn_mean": {
        "cnn": "mean", "pca": False, "hand": False, "demo": False, "fvc_baseline": False,
        "description": "CNN mean pooling (Best CNN Configuration)"
    },
    "best_cnn_hand_demo": {
        "cnn": "mean", "pca": False, "hand": True, "demo": True, "fvc_baseline": False,
        "description": "CNN + Hand-crafted + Demographics (FULL MODEL)"
    },
    "best_cnn_demo": {
        "cnn": "mean", "pca": False, "hand": False, "demo": True, "fvc_baseline": False,
        "description": "CNN + Demographics (Ablation without hand-crafted)"
    },
    "hand_only_fvc0": {
        "cnn": None, "pca": False, "hand": True, "demo": False, "fvc_baseline": True,
        "description": "❗ Hand-crafted + FVC(0) [CLINICAL BENCHMARK]"
    },
    "best_cnn_hand_fvc0": {
        "cnn": "mean", "pca": False, "hand": True, "demo": False, "fvc_baseline": True,
        "description": "❗ CNN + Hand + FVC(0) [Does imaging add value over FVC?]"
    },
    "best_cnn_hand_demo_fvc0": {
        "cnn": "mean", "pca": False, "hand": True, "demo": True, "fvc_baseline": True,
        "description": "❗ FULLY LOADED [Everything: CNN + Hand + Demo + FVC(0)]"
    },
}


# =============================================================================
# CNN EMBEDDING EXTRACTION
# =============================================================================

def extract_slice_level_features(patient_ids, gt_df, npy_root, cnn_extractor, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"\n✓ Loading cached slice-level features from: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"\n{'='*70}\nEXTRACTING SLICE-LEVEL CNN FEATURES\n{'='*70}")
    patient_data = {}
    for patient_id in patient_ids:
        patient_dir = npy_root / patient_id
        if not patient_dir.exists():
            continue
        npy_files = sorted(list(patient_dir.glob('*.npy')))
        if not npy_files:
            continue
        patient_row = gt_df[gt_df['PatientID'] == patient_id]
        if len(patient_row) == 0:
            continue
        patient_data[patient_id] = {
            'gt_has_progressed': bool(patient_row.iloc[0]['has_progressed']),
            'slices': npy_files
        }

    slice_features_df = cnn_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=cache_path
    )
    return slice_features_df


def apply_pooling_to_slices(slice_features_df, pooling_method='max'):
    print(f"\n{'='*70}\nAPPLYING POOLING TO SLICE-LEVEL FEATURES\n{'='*70}")
    print(f"Pooling method: {pooling_method}")

    cnn_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
    patient_embeddings = []

    for patient_id, patient_slices in slice_features_df.groupby('patient_id'):
        feats = patient_slices[cnn_cols].values
        if pooling_method == 'max':
            pooled = np.max(feats, axis=0)
        elif pooling_method == 'mean':
            pooled = np.mean(feats, axis=0)
        elif pooling_method == 'std':
            pooled = np.std(feats, axis=0)
        elif pooling_method == 'min':
            pooled = np.min(feats, axis=0)
        elif pooling_method == 'max_mean':
            pooled = np.concatenate([np.max(feats, axis=0), np.mean(feats, axis=0)])
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        patient_embeddings.append({'patient_id': patient_id, 'embeddings': pooled, 'num_slices': len(patient_slices)})

    return pd.DataFrame(patient_embeddings)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_demographics(train_csv_path, patient_features_df):
    train_df = pd.read_csv(train_csv_path)
    demo_cols = ['Patient'] + [c for c in DEMO_FEATURE_COLS if c in train_df.columns]
    demographics_df = train_df[demo_cols].drop_duplicates(subset=['Patient'])
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    if 'Sex' in enhanced_df.columns:
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
    if 'SmokingStatus' in enhanced_df.columns:
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(
            {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2})
    return enhanced_df


def load_ground_truth(gt_path):
    gt_df = pd.read_csv(gt_path)
    if 'BaselineFVC' in gt_df.columns:
        gt_df['baselinefvc'] = gt_df['BaselineFVC']
    return gt_df


def create_repeated_kfold_splits(kfold_splits_base, n_repeats, base_seed):
    repeated_splits = {0: kfold_splits_base}
    if n_repeats > 1:
        for repeat_idx in range(1, n_repeats):
            np.random.seed(base_seed + repeat_idx * 1000)
            repeated_splits[repeat_idx] = kfold_splits_base.copy()
    return repeated_splits


# =============================================================================
# FEATURE MATRIX CONSTRUCTION
# =============================================================================

def build_feature_matrix(patient_features_df, embeddings_df, config,
                         hand_feature_cols, demo_feature_cols):
    patient_level_df = patient_features_df.groupby('Patient').first().reset_index()

    if config['cnn'] is not None and embeddings_df is not None:
        combined_df = patient_level_df.merge(
            embeddings_df[['patient_id', 'embeddings']],
            left_on='Patient', right_on='patient_id', how='inner'
        )
    else:
        combined_df = patient_level_df.copy()

    all_features = []
    feature_groups = {'cnn_embeddings': [], 'hand_crafted': [], 'demographics': [], 'fvc_baseline': []}

    if config['cnn'] is not None and 'embeddings' in combined_df.columns:
        embedding_dim = combined_df['embeddings'].iloc[0].shape[0]
        embedding_cols = [f'cnn_emb_{i}' for i in range(embedding_dim)]
        embedding_matrix = np.vstack(combined_df['embeddings'].values)
        for i, col in enumerate(embedding_cols):
            combined_df[col] = embedding_matrix[:, i]
        feature_groups['cnn_embeddings'] = embedding_cols
        all_features.extend(embedding_cols)

    if config['hand']:
        available_hand = [c for c in hand_feature_cols if c in combined_df.columns]
        feature_groups['hand_crafted'] = available_hand
        all_features.extend(available_hand)

    if config['demo']:
        available_demo = [c for c in demo_feature_cols if c in combined_df.columns]
        feature_groups['demographics'] = available_demo
        all_features.extend(available_demo)

    if config.get('fvc_baseline', False) and FVC_BASELINE_COL in combined_df.columns:
        feature_groups['fvc_baseline'] = [FVC_BASELINE_COL]
        all_features.append(FVC_BASELINE_COL)

    return combined_df, all_features, feature_groups


# =============================================================================
# DEMOGRAPHICS PREPROCESSING
# =============================================================================

def preprocess_demographics(train_df, val_df, test_df, demo_feature_cols, normalization_type='standard'):
    demo_encoding_info = {}
    new_demo_cols = []

    if 'Age' in demo_feature_cols and 'Age' in train_df.columns:
        age_scaler = StandardScaler()
        age_scaler.fit(train_df[['Age']].values)
        for df in [train_df, val_df, test_df]:
            df['Age_normalized'] = age_scaler.transform(df[['Age']].values)
        demo_encoding_info['age_scaler'] = age_scaler
        new_demo_cols.append('Age_normalized')

    if 'Sex' in demo_feature_cols and 'Sex' in train_df.columns:
        for df in [train_df, val_df, test_df]:
            df['Sex_encoded'] = df['Sex'].map({0: -1, 1: 1})
        demo_encoding_info['sex_encoding'] = {0: -1, 1: 1}
        new_demo_cols.append('Sex_encoded')

    if 'SmokingStatus' in demo_feature_cols and 'SmokingStatus' in train_df.columns:
        train_dummies = pd.get_dummies(train_df['SmokingStatus'], prefix='Smoking', dtype=float) - 0.5
        smoking_cols = sorted(train_dummies.columns.tolist())
        for col in smoking_cols:
            train_df[col] = train_dummies[col] if col in train_dummies.columns else -0.5
            for df in [val_df, test_df]:
                dummies = pd.get_dummies(df['SmokingStatus'], prefix='Smoking', dtype=float) - 0.5
                df[col] = dummies[col] if col in dummies.columns else -0.5
        demo_encoding_info['smoking_columns'] = smoking_cols
        new_demo_cols.extend(smoking_cols)

    return train_df, val_df, test_df, demo_encoding_info, new_demo_cols


# =============================================================================
# EVALUATION
# =============================================================================

def find_optimal_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }
    return optimal_threshold, metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_roc_with_threshold(y_true, y_pred, optimal_threshold, fold_idx, save_dir, dataset='val'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
             label=f'Threshold = {optimal_threshold:.3f}')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx} ({dataset.upper()})', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold_{fold_idx}_roc_{dataset}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred_binary, fold_idx, save_dir, dataset='test'):
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold_idx} ({dataset.upper()})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"fold_{fold_idx}_confusion_{dataset}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion_matrix(all_y_true, all_y_pred_binary, save_dir):
    cm = confusion_matrix(all_y_true, all_y_pred_binary)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'],
                annot_kws={'size': 16, 'weight': 'bold'})
    plt.xlabel('Predicted', fontsize=13, fontweight='bold')
    plt.ylabel('True', fontsize=13, fontweight='bold')
    plt.title('Aggregated Confusion Matrix - All Folds', fontsize=15, fontweight='bold')
    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    plt.text(0.5, -0.12, f'Accuracy: {acc:.3f}', ha='center', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(save_dir / "aggregate_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_roc(fold_results, save_dir):
    plt.figure(figsize=(10, 8))
    all_aucs = []
    for result in fold_results:
        fpr, tpr, _ = roc_curve(result['test_y_true'], result['test_y_pred'])
        auc_val = result['test_auc']
        all_aucs.append(auc_val)
        plt.plot(fpr, tpr, linewidth=2, alpha=0.7, label=f"Fold {result['fold_idx']} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('FPR', fontsize=13); plt.ylabel('TPR', fontsize=13)
    plt.title(f'ROC Curves (Mean AUC={np.mean(all_aucs):.3f}±{np.std(all_aucs):.3f})', fontsize=14)
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "aggregate_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_fold_diagnostics(fold_results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    folds      = [r['fold_idx'] for r in fold_results]
    val_aucs   = [r['val_auc'] for r in fold_results]
    test_aucs  = [r['test_auc'] for r in fold_results]
    thresholds = [r['optimal_threshold'] for r in fold_results]
    precisions = [r['test_precision'] for r in fold_results]
    recalls    = [r['test_recall'] for r in fold_results]
    f1_scores  = [r['test_f1'] for r in fold_results]

    x = np.arange(len(folds))
    axes[0,0].bar(x-.175, val_aucs,  .35, label='Val',  alpha=.8, color='steelblue')
    axes[0,0].bar(x+.175, test_aucs, .35, label='Test', alpha=.8, color='coral')
    axes[0,0].axhline(y=.5, color='red', linestyle='--', alpha=.5); axes[0,0].legend()
    axes[0,0].set_title('AUC Across Folds'); axes[0,0].grid(True, alpha=.3)

    axes[0,1].bar(folds, thresholds, alpha=.8, color='seagreen')
    axes[0,1].axhline(y=.5, color='red', linestyle='--', alpha=.5)
    axes[0,1].set_title('Optimal Thresholds'); axes[0,1].grid(True, alpha=.3)

    axes[1,0].scatter(recalls, precisions, s=100, alpha=.6, color='purple')
    for i, f in enumerate(folds):
        axes[1,0].annotate(f'F{f}', (recalls[i], precisions[i]), fontsize=9)
    axes[1,0].set_title('Precision-Recall'); axes[1,0].grid(True, alpha=.3)

    colors = ['green' if f>0.5 else 'orange' if f>0.3 else 'red' for f in f1_scores]
    axes[1,1].bar(folds, f1_scores, alpha=.8, color=colors)
    axes[1,1].set_title('F1 Scores per Fold'); axes[1,1].grid(True, alpha=.3)

    plt.tight_layout()
    plt.savefig(save_dir / "fold_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================

def _make_display_name(feat_name: str) -> str:
    """Convert raw feature name to human-readable label for plots."""
    if feat_name in HAND_FEATURE_LABELS:
        return HAND_FEATURE_LABELS[feat_name]
    name_map = {
        'Age_normalized':  'Age (normalised)',
        'Sex_encoded':     'Sex',
        'FVC_normalized':  'FVC Baseline',
        'Smoking_0.0':     'Non-smoker',
        'Smoking_1.0':     'Ex-smoker',
        'Smoking_2.0':     'Current smoker',
        'Smoking_0':       'Non-smoker',
        'Smoking_1':       'Ex-smoker',
        'Smoking_2':       'Current smoker',
    }
    return name_map.get(feat_name, feat_name)


def _classify_feature(feat_name: str, feature_groups: dict) -> str:
    """Return group label for a feature name."""
    if feat_name in feature_groups.get('cnn_embeddings', []):
        return 'CNN Embedding'
    if feat_name in feature_groups.get('hand_crafted', []):
        return 'Hand-crafted'
    if feat_name in feature_groups.get('demographics', []) \
       or feat_name in ('Age_normalized', 'Sex_encoded',
                        'Smoking_0', 'Smoking_1', 'Smoking_2',
                        'Smoking_0.0', 'Smoking_1.0', 'Smoking_2.0'):
        return 'Demographics'
    if feat_name in feature_groups.get('fvc_baseline', []) or feat_name == 'FVC_normalized':
        return 'FVC Baseline'
    return 'Other'


def compute_and_plot_shap(
    fold_shap_data: list,
    feature_groups: dict,
    exp_name: str,
    save_dir: Path,
):
    """
    Compute cross-fold SHAP analysis and generate publication-quality figures.

    Parameters
    ----------
    fold_shap_data : list of dicts
        Each dict must contain:
          - 'model'         : trained LightGBM booster
          - 'X_test'        : np.ndarray (n_patients, n_features) — scaled inputs
          - 'X_test_raw'    : np.ndarray (n_patients, n_features) — pre-scale values
          - 'y_test'        : np.ndarray (n_patients,)
          - 'feature_names' : list[str]  — final column names after preprocessing
    feature_groups : dict
        Maps group name → list of raw feature names.
    exp_name : str
        Used in plot titles and saved filenames.
    save_dir : Path
        Directory where plots are saved.

    Outputs (saved to save_dir/shap/)
    ----------------------------------
    shap_mean_bar.png          — Cross-fold mean |SHAP| bar chart (top-20)
    shap_beeswarm.png          — Pooled beeswarm of top-15 features
    shap_group_importance.png  — Feature-group importance (CNN / Hand / Demo / FVC)
    shap_waterfall_examples.png — Waterfall plots for best-predicted + / - patients
    shap_summary.csv           — Full table of mean |SHAP| per feature per fold
    """
    print(f"\n{'='*70}")
    print(f"SHAP ANALYSIS — {exp_name}")
    print(f"{'='*70}")

    shap_dir = save_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Compute SHAP values for every fold ─────────────────────────────────
    all_shap_values   = []   # list of (n_patients, n_features) arrays
    all_X_test        = []   # scaled inputs for beeswarm display values
    all_X_test_raw    = []   # raw inputs for readable feature values
    all_y_test        = []
    feature_names     = fold_shap_data[0]['feature_names']   # same across folds

    for i, fd in enumerate(fold_shap_data):
        print(f"  Computing SHAP for fold {i+1}/{len(fold_shap_data)} …", end='', flush=True)
        explainer = shap.TreeExplainer(fd['model'])
        sv = explainer.shap_values(fd['X_test'])

        # LightGBM binary: shap_values is a list [neg_class, pos_class] or single array
        # Take the positive-class values (shape: n_patients × n_features)
        if isinstance(sv, list):
            sv = sv[1]

        all_shap_values.append(sv)
        all_X_test.append(fd['X_test'])
        all_X_test_raw.append(fd['X_test_raw'])
        all_y_test.append(fd['y_test'])
        print(" done")

    # Pool across folds
    shap_pool   = np.vstack(all_shap_values)    # (total_patients, n_features)
    X_pool      = np.vstack(all_X_test)
    X_raw_pool  = np.vstack(all_X_test_raw)
    y_pool      = np.concatenate(all_y_test)

    # Per-fold mean |SHAP| for summary table
    fold_mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in all_shap_values])
    mean_abs_shap = fold_mean_abs.mean(axis=0)
    std_abs_shap  = fold_mean_abs.std(axis=0)

    # ── Save CSV summary ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame({
        'feature':          feature_names,
        'display_name':     [_make_display_name(f) for f in feature_names],
        'group':            [_classify_feature(f, feature_groups) for f in feature_names],
        'mean_abs_shap':    mean_abs_shap,
        'std_abs_shap':     std_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)
    summary_df.to_csv(shap_dir / "shap_summary.csv", index=False)
    print(f"  ✓ SHAP summary saved: {len(summary_df)} features")

    # ── Decide whether CNN features should be shown individually ─────────────
    n_cnn = len(feature_groups.get('cnn_embeddings', []))
    n_non_cnn = sum(len(v) for k, v in feature_groups.items() if k != 'cnn_embeddings')
    has_cnn = n_cnn > 0

    # ── 2. Mean |SHAP| bar chart ──────────────────────────────────────────────
    print("  Plotting mean |SHAP| bar chart …")
    _plot_mean_abs_shap_bar(
        mean_abs_shap, std_abs_shap, feature_names, feature_groups,
        exp_name=exp_name, save_path=shap_dir / "shap_mean_bar.png",
        top_n=20
    )

    # ── 3. Beeswarm plot (non-CNN features — clinically interpretable) ────────
    print("  Plotting beeswarm …")
    non_cnn_mask = np.array([f not in feature_groups.get('cnn_embeddings', [])
                              for f in feature_names])
    if non_cnn_mask.sum() > 0:
        _plot_beeswarm(
            shap_pool[:, non_cnn_mask],
            X_raw_pool[:, non_cnn_mask],
            [feature_names[i] for i in range(len(feature_names)) if non_cnn_mask[i]],
            exp_name=exp_name,
            save_path=shap_dir / "shap_beeswarm.png",
            top_n=min(15, non_cnn_mask.sum())
        )
    else:
        # All features are CNN — show top-15 CNN dims
        _plot_beeswarm(
            shap_pool, X_raw_pool, feature_names,
            exp_name=exp_name, save_path=shap_dir / "shap_beeswarm.png", top_n=15
        )

    # ── 4. Group-level importance ─────────────────────────────────────────────
    if has_cnn:
        print("  Plotting group-level importance …")
        _plot_group_importance(
            fold_shap_data, all_shap_values, feature_names, feature_groups,
            exp_name=exp_name, save_path=shap_dir / "shap_group_importance.png"
        )

    # ── 4b. Modality contribution by outcome (violin) ─────────────────────────
    print("  Plotting modality contribution by outcome …")
    _plot_modality_contribution_by_outcome(
        all_shap_values=all_shap_values,
        all_y_test=all_y_test,
        feature_names=feature_names,
        feature_groups=feature_groups,
        exp_name=exp_name,
        save_path=shap_dir / "shap_modality_by_outcome.png",
    )

    # ── 5. Per-patient modality reliance ─────────────────────────────────────
    print("  Plotting per-patient modality reliance …")
    patient_reliance_df = _plot_patient_modality_reliance(
        shap_pool=shap_pool,
        y_pool=y_pool,
        feature_names=feature_names,
        feature_groups=feature_groups,
        exp_name=exp_name,
        save_path=shap_dir / "shap_patient_modality_reliance.png",
    )
    if patient_reliance_df is not None:
        patient_reliance_df.to_csv(shap_dir / "shap_patient_modality_reliance.csv", index=False)

    # ── 6. Waterfall examples ─────────────────────────────────────────────────
    print("  Plotting waterfall examples …")
    _plot_waterfall_examples(
        fold_shap_data[0]['model'],  # use fold 0 model
        fold_shap_data[0]['X_test'],
        fold_shap_data[0]['y_test'],
        feature_names, feature_groups,
        exp_name=exp_name,
        save_path=shap_dir / "shap_waterfall_examples.png"
    )

    print(f"  ✓ All SHAP plots saved to: {shap_dir}")
    return summary_df


def _plot_patient_modality_reliance(
    shap_pool: np.ndarray,
    y_pool: np.ndarray,
    feature_names: list,
    feature_groups: dict,
    exp_name: str,
    save_path: Path,
) -> pd.DataFrame:
    """
    Per-patient modality reliance plot.

    For every patient, compute what fraction of their total |SHAP| mass came from
    each feature group (CNN / Hand-crafted / Demographics / FVC Baseline).
    This reveals heterogeneity: some patients are predicted mainly from imaging,
    others mainly from radiomics or clinical features.

    Layout
    ------
    Top panel  : Stacked bar chart — each bar is one patient, sorted by CNN fraction
                 descending. Height of each colour segment = normalised |SHAP| share.
                 Patients are split left/right by true outcome (blue = Non-prog,
                 red = Progressor) with a gap in the middle.
    Bottom panel: Boxplot per group per outcome — distribution of fractional
                 contribution across patients, split by Progressor / Non-progressor.

    Returns
    -------
    pd.DataFrame with columns: patient_idx, outcome, <group_1>, <group_2>, …
                                (fractional contributions, rows sum to 1.0)
    """
    # ── 1. Map every feature to its group ─────────────────────────────────────
    group_of = np.array([_classify_feature(f, feature_groups) for f in feature_names])

    groups_ordered = ['CNN Embedding', 'Hand-crafted', 'Demographics', 'FVC Baseline']
    palette = {
        'CNN Embedding': '#4C72B0',
        'Hand-crafted':  '#DD8452',
        'Demographics':  '#55A868',
        'FVC Baseline':  '#C44E52',
    }

    # Keep only groups that are actually present in this experiment
    present_groups = [g for g in groups_ordered if (group_of == g).any()]
    if not present_groups:
        return None

    # ── 2. Compute per-patient absolute group sums ─────────────────────────────
    abs_shap = np.abs(shap_pool)           # (n_patients, n_features)
    total    = abs_shap.sum(axis=1)        # (n_patients,)  — avoid div by zero

    valid = total > 0
    abs_shap = abs_shap[valid]
    total    = total[valid]
    y_valid  = y_pool[valid]

    group_sums = {}
    for g in present_groups:
        mask = group_of == g
        group_sums[g] = abs_shap[:, mask].sum(axis=1)   # (n_patients,)

    # ── 3. Normalise → fractions (each patient sums to 1) ─────────────────────
    fractions = {g: group_sums[g] / total for g in present_groups}

    # Build DataFrame for export and boxplot
    reliance_df = pd.DataFrame(fractions)
    reliance_df['outcome'] = y_valid
    reliance_df['patient_idx'] = np.arange(len(reliance_df))
    # Sanity: row sums should be ~1.0 (allow for floating point)
    assert np.allclose(reliance_df[present_groups].sum(axis=1), 1.0, atol=1e-6), \
        "Per-patient fractions do not sum to 1 — check feature group mapping."

    # ── 4. Sort patients: non-progressors first, then progressors;
    #       within each class sort by CNN fraction descending ──────────────────
    non_prog = reliance_df[reliance_df['outcome'] == 0].copy()
    prog     = reliance_df[reliance_df['outcome'] == 1].copy()
    cnn_col  = 'CNN Embedding' if 'CNN Embedding' in present_groups else present_groups[0]
    non_prog = non_prog.sort_values(cnn_col, ascending=False)
    prog     = prog.sort_values(cnn_col, ascending=False)

    # ── 5. Build figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(14, len(reliance_df) * 0.18 + 4), 10))
    gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.45,
                           width_ratios=[len(non_prog), len(prog)])

    ax_bar_np  = fig.add_subplot(gs[0, 0])   # stacked bars — non-progressors
    ax_bar_p   = fig.add_subplot(gs[0, 1])   # stacked bars — progressors
    ax_box     = fig.add_subplot(gs[1, :])   # boxplot (full width)

    outcome_titles = {
        ax_bar_np: f'Non-progressors  (n={len(non_prog)})',
        ax_bar_p:  f'Progressors  (n={len(prog)})',
    }
    outcome_edge = {ax_bar_np: '#4C72B0', ax_bar_p: '#C44E52'}

    for ax, subset in [(ax_bar_np, non_prog), (ax_bar_p, prog)]:
        bottom = np.zeros(len(subset))
        x      = np.arange(len(subset))
        for g in present_groups:
            vals = subset[g].values
            ax.bar(x, vals, bottom=bottom, color=palette[g], alpha=0.88,
                   width=0.92, label=g, edgecolor='none')
            bottom += vals

        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.6, len(subset) - 0.4)
        ax.set_xticks([])
        ax.set_ylabel('Fraction of |SHAP|', fontsize=9)
        ax.set_title(outcome_titles[ax], fontsize=10, fontweight='bold',
                     color=outcome_edge[ax])
        ax.axhline(1, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Light outcome-coloured background
        ax.set_facecolor(outcome_edge[ax] + '0D')   # 5% opacity hex

    # Legend on first bar panel
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[g], alpha=0.88)
               for g in present_groups]
    ax_bar_np.legend(handles, present_groups, loc='upper right', fontsize=8,
                     title='Feature group', title_fontsize=8,
                     framealpha=0.85, ncol=1)

    # ── 6. Boxplot: distribution of fractional contribution by group × outcome ──
    box_data   = []
    box_labels = []
    box_colors = []
    outcome_map = {0: 'Non-prog.', 1: 'Progressor'}

    for g in present_groups:
        for outcome in [0, 1]:
            vals = reliance_df.loc[reliance_df['outcome'] == outcome, g].values
            box_data.append(vals)
            box_labels.append(f"{g}\n({outcome_map[outcome]})")
            box_colors.append(palette[g])

    bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Shade alternating group columns for readability
    n_outcomes = 2
    for gi, g in enumerate(present_groups):
        if gi % 2 == 0:
            x_start = gi * n_outcomes + 0.5
            x_end   = x_start + n_outcomes
            ax_box.axvspan(x_start, x_end, alpha=0.04, color='grey')

    ax_box.set_xticks(range(1, len(box_data) + 1))
    ax_box.set_xticklabels(box_labels, fontsize=8)
    ax_box.set_ylabel('Fractional |SHAP| contribution', fontsize=9)
    ax_box.set_title('Distribution of modality reliance by outcome', fontsize=10)
    ax_box.set_ylim(-0.05, 1.15)
    ax_box.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
    ax_box.grid(True, alpha=0.2, axis='y')
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)

    fig.suptitle(
        f'Per-patient Modality Reliance — {exp_name}\n'
        f'(Each column = 1 patient; height = fraction of total |SHAP| from each group;\n'
        f' sorted by CNN fraction descending within each outcome)',
        fontsize=11, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 7. Print quick summary to console ─────────────────────────────────────
    print(f"\n  Per-patient modality reliance summary ({exp_name}):")
    for g in present_groups:
        for outcome in [0, 1]:
            vals = reliance_df.loc[reliance_df['outcome'] == outcome, g].values
            if len(vals):
                print(f"    {g:20s} {outcome_map[outcome]:12s} "
                      f"median={np.median(vals):.3f}  "
                      f"IQR=[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]")

    return reliance_df


def _plot_mean_abs_shap_bar(mean_abs, std_abs, feature_names, feature_groups,
                            exp_name, save_path, top_n=20):
    """Horizontal bar chart of top-N features by mean |SHAP| across folds."""
    n_cnn = len(feature_groups.get('cnn_embeddings', []))

    # Aggregate CNN features into a single row
    cnn_cols_set = set(feature_groups.get('cnn_embeddings', []))
    cnn_idx      = [i for i, f in enumerate(feature_names) if f in cnn_cols_set]
    non_cnn_idx  = [i for i, f in enumerate(feature_names) if f not in cnn_cols_set]

    rows = []
    # Add individual non-CNN features
    for i in non_cnn_idx:
        rows.append({
            'label': _make_display_name(feature_names[i]),
            'group': _classify_feature(feature_names[i], feature_groups),
            'mean':  mean_abs[i],
            'std':   std_abs[i],
        })
    # Add aggregated CNN entry
    if cnn_idx:
        rows.append({
            'label': f'CNN Embeddings\n({n_cnn} dims, summed)',
            'group': 'CNN Embedding',
            'mean':  mean_abs[cnn_idx].sum(),
            'std':   std_abs[cnn_idx].mean(),  # representative std
        })

    df = pd.DataFrame(rows).sort_values('mean', ascending=False).head(top_n)
    df = df.iloc[::-1]  # reverse for horizontal bar (top at top)

    palette = {
        'CNN Embedding': '#4C72B0',
        'Hand-crafted':  '#DD8452',
        'Demographics':  '#55A868',
        'FVC Baseline':  '#C44E52',
        'Other':         '#8172B2',
    }
    colors = [palette.get(g, '#8172B2') for g in df['group']]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.45)))
    bars = ax.barh(df['label'], df['mean'], xerr=df['std'],
                   color=colors, alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP value| (cross-fold)', fontsize=12)
    ax.set_title(f'Feature Importance — {exp_name}\n(Mean absolute SHAP, ± std across folds)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in palette.values()]
    ax.legend(handles, list(palette.keys()), loc='lower right', fontsize=9,
              title='Feature Group', title_fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_beeswarm(shap_vals, X_raw, feat_names, exp_name, save_path, top_n=15):
    """
    SHAP beeswarm plot for the top-N features by mean |SHAP|.
    Most important feature is displayed at the TOP (standard SHAP convention).
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    # Sort descending by importance; take top_n
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    top_shap  = shap_vals[:, top_idx]
    top_X     = X_raw[:, top_idx]
    top_names = [_make_display_name(feat_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5)))

    # y positions: feature ranked #1 (highest importance) → highest y value → top of plot
    # We assign y = top_n-1 to the most important, y = 0 to the least important.
    # That way set_yticks with these labels puts the best feature at the top naturally.
    y_pos = np.arange(top_n - 1, -1, -1)   # [top_n-1, top_n-2, ..., 0]

    for j, yp in enumerate(y_pos):
        sv   = top_shap[:, j]
        vals = top_X[:, j]
        v_min, v_max = vals.min(), vals.max()
        norm_vals = (vals - v_min) / (v_max - v_min + 1e-9)
        jitter = np.random.normal(0, 0.06, size=len(sv))
        ax.scatter(sv, yp + jitter, c=norm_vals, cmap='RdBu_r',
                   vmin=0, vmax=1, alpha=0.7, s=20, linewidths=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)   # top_names[0] = most important → top
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.set_title(f'SHAP Beeswarm — {exp_name}\n(non-CNN features, top {top_n} by mean |SHAP|)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.6)
    cbar.set_label('Feature value\n(low → high)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_group_importance(fold_shap_data, all_shap_values, feature_names,
                           feature_groups, exp_name, save_path):
    """
    Bar chart of summed mean |SHAP| per feature group across folds.
    Useful when CNN embeddings are present to see their collective contribution.
    Each group's bar is drawn on its own natural scale (no shared y-axis).
    """
    groups_ordered = ['FVC Baseline', 'Hand-crafted', 'Demographics', 'CNN Embedding']
    palette = {
        'FVC Baseline':  '#C44E52',
        'Hand-crafted':  '#DD8452',
        'Demographics':  '#55A868',
        'CNN Embedding': '#4C72B0',
    }

    group_of = {f: _classify_feature(f, feature_groups) for f in feature_names}

    fold_group_imp = {g: [] for g in groups_ordered}
    for sv in all_shap_values:
        abs_mean = np.abs(sv).mean(axis=0)
        for g in groups_ordered:
            idxs = [i for i, f in enumerate(feature_names) if group_of[f] == g]
            fold_group_imp[g].append(abs_mean[idxs].sum() if idxs else 0.0)

    present = [g for g in groups_ordered if max(fold_group_imp[g]) > 0]
    means   = [np.mean(fold_group_imp[g]) for g in present]
    stds    = [np.std(fold_group_imp[g])  for g in present]
    colors  = [palette[g] for g in present]

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.8), 5))
    x    = np.arange(len(present))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=6, edgecolor='white', linewidth=0.5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + ax.get_ylim()[1] * 0.01,
                f'{m:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_ylabel('Summed mean |SHAP| (cross-fold)', fontsize=11)
    ax.set_title(f'Feature Group Importance — {exp_name}\n(± std across folds)',
                 fontsize=12, fontweight='bold')
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')   # no 1e-6 offset
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_modality_contribution_by_outcome(all_shap_values, all_y_test,
                                            feature_names, feature_groups,
                                            exp_name, save_path):
    """
    Violin + strip plot showing the SIGNED sum of SHAP values per feature group,
    split by true outcome (Progressor vs Non-progressor).

    Each modality gets its own subplot with an independent y-axis so that
    small-magnitude groups (Hand-crafted, Demographics) are visible alongside
    the large-magnitude CNN group.

    Fixes vs earlier version:
    - Independent y-axis per subplot (no shared scale → no flat-line artefact)
    - `ticklabel_format(useOffset=False)` on every axis → no scientific offset notation
    - Signed SHAP sum (not absolute) → shows direction of contribution per outcome
    """
    groups_ordered = ['CNN Embedding', 'Hand-crafted', 'Demographics', 'FVC Baseline']
    palette_group  = {
        'CNN Embedding': '#4C72B0',
        'Hand-crafted':  '#DD8452',
        'Demographics':  '#55A868',
        'FVC Baseline':  '#C44E52',
    }
    outcome_colors = {0: '#4C72B0', 1: '#C44E52'}   # blue=non-prog, red=prog
    outcome_labels = {0: 'Non-prog.', 1: 'Progressor'}

    group_of = {f: _classify_feature(f, feature_groups) for f in feature_names}

    # Pool SHAP values and labels across folds
    shap_pool = np.vstack(all_shap_values)     # (N_total, n_features)
    y_pool    = np.concatenate(all_y_test)     # (N_total,)

    # Compute SIGNED group sums per patient
    # (sum of raw SHAP values, not abs — shows direction of contribution)
    group_data = {}
    for g in groups_ordered:
        idxs = [i for i, f in enumerate(feature_names) if group_of[f] == g]
        if idxs:
            group_data[g] = shap_pool[:, idxs].sum(axis=1)  # signed sum

    present = [g for g in groups_ordered if g in group_data]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 6),
                             sharey=False)   # INDEPENDENT y-axes
    if len(present) == 1:
        axes = [axes]

    for ax, g in zip(axes, present):
        vals = group_data[g]
        color = palette_group[g]

        for outcome in [0, 1]:
            mask   = y_pool == outcome
            subset = vals[mask]
            x_pos  = outcome
            # Violin
            if mask.sum() >= 3:
                parts = ax.violinplot([subset], positions=[x_pos],
                                      widths=0.5, showmedians=True,
                                      showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(outcome_colors[outcome])
                    pc.set_alpha(0.35)
                parts['cmedians'].set_color(outcome_colors[outcome])
                parts['cmedians'].set_linewidth(2)
            # Strip (jittered dots)
            jitter = np.random.normal(0, 0.04, size=mask.sum())
            ax.scatter(x_pos + jitter, subset,
                       color=outcome_colors[outcome], alpha=0.7,
                       s=18, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([outcome_labels[0], outcome_labels[1]], fontsize=10)
        ax.set_title(g, fontsize=11, fontweight='bold', color=color)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_ylabel('Modality SHAP sum\n(signed contribution)', fontsize=9)

        # ── KEY FIX: independent scale, no scientific offset notation ─────────
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda val, _: f'{val:.3f}' if abs(val) < 0.1
                              else f'{val:.2f}')
        )
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Modality contribution by outcome — {exp_name}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_waterfall_examples(model, X_test, y_test, feature_names,
                              feature_groups, exp_name, save_path):
    """
    Waterfall plots for 2 individual patients:
    - best predicted progressor  (highest predicted prob, label=1)
    - best predicted non-progressor (lowest predicted prob, label=0)
    Only shows non-CNN features for interpretability.
    """
    preds = model.predict(X_test)
    non_cnn_mask = np.array([f not in feature_groups.get('cnn_embeddings', [])
                              for f in feature_names])
    display_names = [_make_display_name(f)
                     for i, f in enumerate(feature_names) if non_cnn_mask[i]]

    if non_cnn_mask.sum() == 0:
        return   # all CNN, skip

    X_non_cnn = X_test[:, non_cnn_mask]

    # Pick examples
    progressors     = np.where(y_test == 1)[0]
    non_progressors = np.where(y_test == 0)[0]

    if len(progressors) == 0 or len(non_progressors) == 0:
        return

    best_prog_idx     = progressors[np.argmax(preds[progressors])]
    best_nonprog_idx  = non_progressors[np.argmin(preds[non_progressors])]
    examples = [
        (best_prog_idx,    f'Progressor (pred={preds[best_prog_idx]:.2f})'),
        (best_nonprog_idx, f'Non-progressor (pred={preds[best_nonprog_idx]:.2f})'),
    ]

    explainer = shap.TreeExplainer(model)
    base_val  = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (idx, title) in zip(axes, examples):
        sv_full  = explainer.shap_values(X_test[idx:idx+1])[0]
        if isinstance(sv_full, list):
            sv_full = sv_full[1]
        sv_full = sv_full.flatten()
        sv_non_cnn = sv_full[non_cnn_mask]
        x_vals     = X_non_cnn[idx]

        # Sort by |SHAP| descending, keep top 10
        order = np.argsort(np.abs(sv_non_cnn))[::-1][:10]
        sv_plot    = sv_non_cnn[order]
        names_plot = [display_names[i] for i in order]
        colors_plot = ['#C44E52' if s > 0 else '#4C72B0' for s in sv_plot]

        # Horizontal bar (waterfall approximation)
        y_pos = np.arange(len(sv_plot))[::-1]
        ax.barh(y_pos, sv_plot, color=colors_plot, alpha=0.85, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_plot, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP contribution', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Prediction annotation
        cnn_sum = sv_full[~non_cnn_mask].sum()
        if len(feature_groups.get('cnn_embeddings', [])) > 0:
            ax.text(0.97, 0.02, f'CNN total: {cnn_sum:+.3f}',
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize=9, color='#4C72B0',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.4))

    fig.suptitle(f'SHAP Waterfall Examples — {exp_name}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# TRAINING  (modified to return SHAP data)
# =============================================================================

def train_single_fold(
    combined_features_df, gt_df, fold_data, fold_idx, config,
    feature_cols, feature_groups, results_dir, ablation_config, train_ids
):
    """
    Train LightGBM on a single fold.
    Returns metrics dict PLUS shap_data dict needed for SHAP analysis.
    """
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx}")
    print("="*70)

    train_ids_ = fold_data['train']
    val_ids    = fold_data['val']
    test_ids   = fold_data['test']

    train_df = combined_features_df[combined_features_df['Patient'].isin(train_ids_)].copy()
    val_df   = combined_features_df[combined_features_df['Patient'].isin(val_ids)].copy()
    test_df  = combined_features_df[combined_features_df['Patient'].isin(test_ids)].copy()

    for df in [train_df, val_df, test_df]:
        df.drop(columns=[c for c in df.columns if c == 'patient_id'], inplace=True, errors='ignore')

    for split_df, split_name in [(train_df,'Train'),(val_df,'Val'),(test_df,'Test')]:
        split_df.merge(gt_df[['PatientID','has_progressed']],
                       left_on='Patient', right_on='PatientID', how='left')

    train_df = train_df.merge(gt_df[['PatientID','has_progressed']], left_on='Patient', right_on='PatientID', how='left')
    val_df   = val_df.merge(gt_df[['PatientID','has_progressed']],   left_on='Patient', right_on='PatientID', how='left')
    test_df  = test_df.merge(gt_df[['PatientID','has_progressed']],  left_on='Patient', right_on='PatientID', how='left')

    # ── Demographics + FVC preprocessing ──────────────────────────────────────
    demo_features_in_data = [f for f in DEMO_FEATURE_COLS if f in feature_cols]
    fvc_in_data           = FVC_BASELINE_COL in feature_cols

    if demo_features_in_data or fvc_in_data:
        train_df, val_df, test_df, demo_encoding_info, new_demo_cols = preprocess_demographics(
            train_df, val_df, test_df, demo_features_in_data, config['normalization_type']
        )
        updated_feature_cols = [f for f in feature_cols if f not in demo_features_in_data]
        updated_feature_cols.extend(new_demo_cols)

        if fvc_in_data and FVC_BASELINE_COL in train_df.columns:
            fvc_scaler = StandardScaler()
            fvc_scaler.fit(train_df[[FVC_BASELINE_COL]].values)
            for df in [train_df, val_df, test_df]:
                df['FVC_normalized'] = fvc_scaler.transform(df[[FVC_BASELINE_COL]].values)
            updated_feature_cols = [f if f != FVC_BASELINE_COL else 'FVC_normalized'
                                    for f in updated_feature_cols]

        feature_cols = updated_feature_cols

    # ── Fill NaN ──────────────────────────────────────────────────────────────
    for col in feature_cols:
        for df in [train_df, val_df, test_df]:
            if col in df.columns and df[col].isnull().any():
                fill_val = train_df[col].median() if col in HAND_FEATURE_COLS else 0
                df[col].fillna(fill_val, inplace=True)

    # Store raw (pre-scale) test values for SHAP display
    X_test_raw = test_df[feature_cols].values.copy()

    X_train = train_df[feature_cols].values
    y_train = train_df['has_progressed'].values.astype(int)
    X_val   = val_df[feature_cols].values
    y_val   = val_df['has_progressed'].values.astype(int)
    X_test  = test_df[feature_cols].values
    y_test  = test_df['has_progressed'].values.astype(int)

    # ── Optional PCA ──────────────────────────────────────────────────────────
    pca_model = None
    if ablation_config.get('pca', False):
        max_c = min(CONFIG['pca_max_components'], X_train.shape[0], X_train.shape[1])
        pca_full = PCA(n_components=max_c)
        pca_full.fit(X_train)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_c = max(10, min(np.argmax(cumsum >= CONFIG['pca_variance_threshold']) + 1, max_c))
        pca_model = PCA(n_components=n_c, random_state=42)
        pca_model.fit(X_train)
        X_train = pca_model.transform(X_train)
        X_val   = pca_model.transform(X_val)
        X_test  = pca_model.transform(X_test)
        feature_cols = [f'PC{i+1}' for i in range(n_c)]

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Train LightGBM ────────────────────────────────────────────────────────
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)
    model = lgb.train(
        config['lgb_params'], train_data,
        num_boost_round=config['num_boost_round'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(config['early_stopping_rounds'])]
    )
    print(f"✓ Training complete. Best iteration: {model.best_iteration}")

    # ── Threshold + test eval ─────────────────────────────────────────────────
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc   = roc_auc_score(y_val, val_preds)
    optimal_threshold, val_metrics = find_optimal_threshold(y_val, val_preds)

    test_preds        = model.predict(X_test, num_iteration=model.best_iteration)
    test_auc          = roc_auc_score(y_test, test_preds)
    test_preds_binary = (test_preds >= optimal_threshold).astype(int)
    test_accuracy     = accuracy_score(y_test, test_preds_binary)
    test_precision    = precision_score(y_test, test_preds_binary, zero_division=0)
    test_recall       = recall_score(y_test, test_preds_binary, zero_division=0)
    test_f1           = f1_score(y_test, test_preds_binary, zero_division=0)
    cm                = confusion_matrix(y_test, test_preds_binary)
    tn, fp, fn, tp_   = cm.ravel()
    test_specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Val AUC: {val_auc:.4f}  |  Test AUC: {test_auc:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_with_threshold(y_val, val_preds, optimal_threshold, fold_idx, fold_dir, 'val')
    plot_roc_with_threshold(y_test, test_preds, optimal_threshold, fold_idx, fold_dir, 'test')
    plot_confusion_matrix(y_test, test_preds_binary, fold_idx, fold_dir, 'test')

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save_model(str(fold_dir / "lightgbm_model.txt"))
    with open(fold_dir / "preprocessing.pkl", 'wb') as f:
        pickle.dump({
            'scaler': scaler, 'pca_model': pca_model,
            'feature_names': feature_cols, 'feature_groups': feature_groups,
            'ablation_config': ablation_config
        }, f)

    return {
        # --- Metrics ---
        'fold_idx':          fold_idx,
        'val_auc':           val_auc,
        'test_auc':          test_auc,
        'test_accuracy':     test_accuracy,
        'test_precision':    test_precision,
        'test_recall':       test_recall,
        'test_specificity':  test_specificity,
        'test_f1':           test_f1,
        'optimal_threshold': optimal_threshold,
        'test_y_true':       y_test,
        'test_y_pred':       test_preds,
        'test_y_pred_binary':test_preds_binary,
        # --- SHAP data (returned for cross-fold analysis) ---
        'shap_data': {
            'model':         model,
            'X_test':        X_test,       # scaled — what the model sees
            'X_test_raw':    X_test_raw,   # raw   — human-readable values
            'y_test':        y_test,
            'feature_names': feature_cols,
        },
    }


# =============================================================================
# ABLATION STUDY COMPARISON PLOT
# =============================================================================

def plot_ablation_comparison(ablation_results, save_dir):
    experiments = list(ablation_results.keys())
    mean_aucs = [ablation_results[e]['mean_auc'] for e in experiments]
    std_aucs  = [ablation_results[e]['std_auc']  for e in experiments]
    mean_f1s  = [ablation_results[e]['mean_f1']  for e in experiments]
    std_f1s   = [ablation_results[e]['std_f1']   for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_pos  = np.arange(len(experiments))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for ax, vals, stds, ylabel, title in [
        (axes[0], mean_aucs, std_aucs, 'Test AUC', 'AUC Comparison'),
        (axes[1], mean_f1s,  std_f1s,  'Test F1',  'F1 Score Comparison'),
    ]:
        ax.bar(x_pos, vals, yerr=stds, alpha=0.8, capsize=5, color=colors)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiments, rotation=20, ha='right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for i, (m, s) in enumerate(zip(vals, stds)):
            ax.text(i, m + s + 0.015, f'{m:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Ablation comparison saved")


# =============================================================================
# SINGLE ABLATION EXPERIMENT RUNNER
# =============================================================================

def run_single_ablation_experiment(exp_name, exp_config, patient_features_df, gt_df,
                                   repeated_kfold_splits, embeddings_dict,
                                   base_results_dir, config):
    print("\n" + "="*80)
    print(f"ABLATION EXPERIMENT: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print("="*80)

    exp_dir = base_results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    embeddings_df = embeddings_dict.get(exp_config['cnn']) if exp_config['cnn'] is not None else None

    fold_results = []
    fold_shap_data = []   # accumulates per-fold SHAP artefacts
    total_runs = 0

    for repeat_idx in sorted(repeated_kfold_splits.keys()):
        kfold_splits = repeated_kfold_splits[repeat_idx]

        for fold_idx in sorted(kfold_splits.keys()):
            total_runs += 1
            fold_seed = config['base_seed'] + repeat_idx * 1000 + fold_idx * 100
            set_seed(fold_seed)

            fold_data = kfold_splits[fold_idx]

            combined_df, feature_cols, feature_groups = build_feature_matrix(
                patient_features_df, embeddings_df, exp_config,
                HAND_FEATURE_COLS, DEMO_FEATURE_COLS
            )

            unique_fold_id = f"fold{fold_idx}" if config['n_repeats'] == 1 \
                             else f"rep{repeat_idx}_fold{fold_idx}"
            print(f"\n[Run {total_runs}] {unique_fold_id}")

            result = train_single_fold(
                combined_df, gt_df, fold_data, unique_fold_id, config,
                feature_cols, feature_groups, exp_dir, exp_config, fold_data['train']
            )

            fold_shap_data.append(result.pop('shap_data'))   # separate SHAP data
            fold_results.append(result)

    # ── Metrics aggregation ───────────────────────────────────────────────────
    test_aucs        = [r['test_auc']        for r in fold_results]
    test_f1s         = [r['test_f1']         for r in fold_results]
    test_recalls     = [r['test_recall']     for r in fold_results]
    test_precisions  = [r['test_precision']  for r in fold_results]
    test_specificities = [r['test_specificity'] for r in fold_results]

    print(f"\n{'='*70}\nEXPERIMENT {exp_name} RESULTS\n{'='*70}")
    print(f"Test AUC:         {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test F1:          {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Recall:      {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")
    print(f"Test Precision:   {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
    print(f"Test Specificity: {np.mean(test_specificities):.4f} ± {np.std(test_specificities):.4f}")

    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ('test_y_true','test_y_pred','test_y_pred_binary')}
                  for r in fold_results]).to_csv(exp_dir / "fold_results.csv", index=False)

    # ── Standard visualisations ───────────────────────────────────────────────
    plot_fold_diagnostics(fold_results, exp_dir)
    plot_aggregate_roc(fold_results, exp_dir)
    all_y_true  = np.concatenate([r['test_y_true']       for r in fold_results])
    all_y_pred  = np.concatenate([r['test_y_pred_binary'] for r in fold_results])
    plot_aggregate_confusion_matrix(all_y_true, all_y_pred, exp_dir)

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    # Use feature_groups from the last fold (stable across folds)
    shap_summary = compute_and_plot_shap(
        fold_shap_data=fold_shap_data,
        feature_groups=feature_groups,
        exp_name=exp_name,
        save_dir=exp_dir,
    )
    print(f"\nTop-5 features by mean |SHAP| ({exp_name}):")
    print(shap_summary[['display_name','group','mean_abs_shap','std_abs_shap']].head(5).to_string(index=False))

    return {
        'exp_name':         exp_name,
        'config':           exp_config,
        'fold_results':     fold_results,
        'mean_auc':         np.mean(test_aucs),
        'std_auc':          np.std(test_aucs),
        'mean_f1':          np.mean(test_f1s),
        'std_f1':           np.std(test_f1s),
        'mean_recall':      np.mean(test_recalls),
        'std_recall':       np.std(test_recalls),
        'mean_precision':   np.mean(test_precisions),
        'std_precision':    np.std(test_precisions),
        'mean_specificity': np.mean(test_specificities),
        'std_specificity':  np.std(test_specificities),
        'shap_summary':     shap_summary,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(CONFIG['base_seed'])

    base_results_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\lightgbm_fvc_0_less_reg")
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    patient_features_df = pd.read_csv(CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(CONFIG['train_csv_path'], patient_features_df)
    gt_df               = load_ground_truth(CONFIG['gt_path'])

    # Merge FVC baseline into patient features
    if 'baselinefvc' in gt_df.columns:
        fvc_data = gt_df[['PatientID', 'baselinefvc']].copy()
        patient_features_df = patient_features_df.merge(
            fvc_data, left_on='Patient', right_on='PatientID', how='left'
        )
        patient_features_df.drop('PatientID', axis=1, inplace=True, errors='ignore')

    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits_base = pickle.load(f)

    repeated_kfold_splits = create_repeated_kfold_splits(
        kfold_splits_base, CONFIG['n_repeats'], CONFIG['base_seed']
    )

    all_patient_ids     = gt_df['PatientID'].unique().tolist()
    patient_features_df = patient_features_df[patient_features_df['Patient'].isin(all_patient_ids)]

    # ── CNN features ───────────────────────────────────────────────────────────
    slice_features_cache = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\LightGBM_setting\slice_features.csv")
    if not slice_features_cache.exists():
        raise FileNotFoundError(f"Slice features file not found: {slice_features_cache}")

    print(f"✓ Loading pre-extracted slice features from: {slice_features_cache}")
    slice_features_df = pd.read_csv(slice_features_cache)

    embeddings_dict = {'mean': apply_pooling_to_slices(slice_features_df, pooling_method='mean')}

    # ── Run all experiments ────────────────────────────────────────────────────
    ablation_results = {}
    for exp_name, exp_config in ABLATION_CONFIGS.items():
        result = run_single_ablation_experiment(
            exp_name, exp_config, patient_features_df, gt_df,
            repeated_kfold_splits, embeddings_dict, base_results_dir, CONFIG
        )
        ablation_results[exp_name] = result

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*80 + "\nFINAL RESULTS SUMMARY\n" + "="*80)
    summary_data = []
    for exp_name, result in ablation_results.items():
        summary_data.append({
            'Experiment':        exp_name,
            'Description':       result['config']['description'],
            'N_Runs':            len(result['fold_results']),
            'Mean_AUC':          f"{result['mean_auc']:.4f}",
            'Std_AUC':           f"{result['std_auc']:.4f}",
            'Mean_F1':           f"{result['mean_f1']:.4f}",
            'Std_F1':            f"{result['std_f1']:.4f}",
            'Mean_Recall':       f"{result['mean_recall']:.4f}",
            'Std_Recall':        f"{result['std_recall']:.4f}",
            'Mean_Precision':    f"{result['mean_precision']:.4f}",
            'Std_Precision':     f"{result['std_precision']:.4f}",
            'Mean_Specificity':  f"{result['mean_specificity']:.4f}",
            'Std_Specificity':   f"{result['std_specificity']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(base_results_dir / "ablation_summary.csv", index=False)

    # ── Statistical tests ──────────────────────────────────────────────────────
    print(f"\n{'='*80}\nSTATISTICAL SIGNIFICANCE TESTING (Paired t-test)\n{'='*80}")
    exp_names  = list(ablation_results.keys())
    auc_arrays = {e: np.array([r['test_auc'] for r in ablation_results[e]['fold_results']])
                  for e in exp_names}

    pairwise_results = []
    for i, e1 in enumerate(exp_names):
        for j, e2 in enumerate(exp_names):
            if i < j:
                t_stat, p_value = ttest_rel(auc_arrays[e1], auc_arrays[e2])
                mean_diff = auc_arrays[e1].mean() - auc_arrays[e2].mean()
                sig = ('***' if p_value < 0.001 else '**' if p_value < 0.01
                       else '*' if p_value < 0.05 else 'ns')
                pairwise_results.append({
                    'Comparison': f"{e1} vs {e2}",
                    'Mean_Diff':  mean_diff, 't_statistic': t_stat,
                    'p_value': p_value, 'Significance': sig
                })
                print(f"  {e1:25s} vs {e2}")
                print(f"    Δ={mean_diff:+.4f}  t={t_stat:.4f}  p={p_value:.4f} {sig}\n")

    pd.DataFrame(pairwise_results).to_csv(
        base_results_dir / "statistical_tests.csv", index=False)

    plot_ablation_comparison(ablation_results, base_results_dir)

    # ── Cross-experiment SHAP summary ─────────────────────────────────────────
    print(f"\n{'='*80}\nCROSS-EXPERIMENT SHAP SUMMARY\n{'='*80}")
    for exp_name, result in ablation_results.items():
        if 'shap_summary' in result:
            top = result['shap_summary'].head(3)
            print(f"\n  {exp_name} — top-3 features:")
            for _, row in top.iterrows():
                print(f"    {row['display_name']:30s}  |SHAP|={row['mean_abs_shap']:.4f} ± {row['std_abs_shap']:.4f}  [{row['group']}]")

    best_exp = max(ablation_results.items(), key=lambda x: x[1]['mean_auc'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST: {best_exp[0]}  AUC={best_exp[1]['mean_auc']:.4f} ± {best_exp[1]['std_auc']:.4f}")
    print(f"{'='*80}")
    print(f"\n✓ COMPLETE — results → {base_results_dir}")


if __name__ == "__main__":
    main()