"""
THRESHOLD OPTIMIZATION FOR BEST LIGHTGBM MODEL

This script performs comprehensive threshold optimization for the best-performing
LightGBM model (best_cnn_hand_demo_fvc0) by:

1. Loading all fold models and their validation predictions
2. Sweeping threshold from 0.1 to 0.9 in fine steps
3. Computing F1, Precision, Recall, Accuracy for each threshold
4. Plotting F1 vs threshold curve
5. Selecting optimal threshold that maximizes F1 on validation set
6. Recomputing all metrics and confusion matrices with optimal threshold
7. Comparing to original Youden's index threshold

Results saved to: threshold_optimization/ directory
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Path to best model experiment
    "model_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\lightgbm_fvc_0\best_cnn_hand_demo_fvc0"),
    
    # Paths to data sources (needed to reconstruct validation sets)
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "slice_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\LightGBM_setting\slice_features.csv"),
    
    # Threshold sweep settings
    "threshold_min": 0.1,
    "threshold_max": 0.9,
    "threshold_step": 0.01,  # 81 thresholds total
    
    # Output directory
    "output_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\LightGBM_setting\threshold_optimization"),
    
    # Visualization settings
    "figsize": (12, 8),
    "dpi": 150,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_ground_truth():
    """Load ground truth labels."""
    gt_df = pd.read_csv(CONFIG['gt_path'])
    
    # Rename BaselineFVC to baselinefvc for consistency
    if 'BaselineFVC' in gt_df.columns:
        gt_df['baselinefvc'] = gt_df['BaselineFVC']
    
    print(f"✓ Loaded ground truth: {len(gt_df)} patients")
    return gt_df


def load_kfold_splits():
    """Load K-fold split definitions."""
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        repeated_kfold_splits = pickle.load(f)
    
    # The pickle file structure can be:
    # 1. {repeat_idx: {fold_idx: {'train': [...], 'val': [...], 'test': [...]}}}  (repeated k-fold)
    # 2. {fold_idx: {'train': [...], 'val': [...], 'test': [...]}}  (simple k-fold)
    
    # Check the structure
    first_key = list(repeated_kfold_splits.keys())[0]
    first_value = repeated_kfold_splits[first_key]
    
    # If first_value has keys 'train', 'val', 'test', then we have simple k-fold format
    if isinstance(first_value, dict) and 'train' in first_value:
        # Simple k-fold: {fold_idx: {'train': ..., 'val': ..., 'test': ...}}
        print(f"✓ Loaded k-fold splits: {len(repeated_kfold_splits)} folds")
        return repeated_kfold_splits
    
    # Otherwise, assume repeated k-fold format: use first repetition
    # Structure: {repeat_idx: {fold_idx: {'train': ..., 'val': ..., 'test': ...}}}
    if isinstance(first_value, dict):
        # Get the first repetition (repeat_idx = 0 or whatever first key is)
        print(f"✓ Loaded repeated k-fold splits: repetition {first_key}, {len(first_value)} folds")
        return first_value
    
    # Fallback: return as-is and hope for the best
    print("⚠️ Warning: Unknown k-fold structure, returning as-is")
    return repeated_kfold_splits


def load_patient_features_and_embeddings():
    """
    Load patient features, merge demographics, and load CNN embeddings.
    Must match the configuration used in training.
    """
    # Load hand-crafted features
    patient_features_df = pd.read_csv(CONFIG['patient_features_path'])
    print(f"✓ Patient features: {patient_features_df.shape}")
    
    # Merge demographics from train.csv (Age, Sex, SmokingStatus)
    train_df = pd.read_csv(CONFIG['train_csv_path'])
    
    demo_cols = ['Patient', 'Age', 'Sex', 'SmokingStatus']
    demographics_df = train_df[demo_cols].drop_duplicates(subset=['Patient'])
    
    # Merge demographics into patient features
    patient_features_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    print(f"✓ Merged demographics: {patient_features_df.shape}")
    
    # Load pre-extracted slice-level CNN features and apply mean pooling
    if not CONFIG['slice_features_path'].exists():
        raise FileNotFoundError(f"Slice features not found at: {CONFIG['slice_features_path']}")
    
    slice_features_df = pd.read_csv(CONFIG['slice_features_path'])
    print(f"✓ Slice features: {len(slice_features_df)} slices")
    
    # Apply MEAN pooling (same as training)
    embeddings_df = apply_pooling_to_slices(slice_features_df, pooling_method='mean')
    print(f"✓ CNN embeddings: {len(embeddings_df)} patients")
    
    return patient_features_df, embeddings_df


def apply_pooling_to_slices(slice_features_df, pooling_method='mean'):
    """
    Apply pooling to slice-level features to get patient-level embeddings.
    This replicates the logic from lightgbm_cnn_improved.py
    """
    feature_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
    
    if pooling_method == 'mean':
        agg_func = np.mean
    elif pooling_method == 'max':
        agg_func = np.max
    elif pooling_method == 'std':
        agg_func = np.std
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")
    
    patient_embeddings = []
    for patient_id in slice_features_df['patient_id'].unique():
        patient_slices = slice_features_df[slice_features_df['patient_id'] == patient_id]
        features = patient_slices[feature_cols].values
        pooled = agg_func(features, axis=0)
        patient_embeddings.append({
            'patient_id': patient_id,
            'embeddings': pooled
        })
    
    embeddings_df = pd.DataFrame(patient_embeddings)
    return embeddings_df


def preprocess_demographics_for_fold(train_df, val_df):
    """
    Apply sophisticated demographics preprocessing matching lightgbm_cnn_improved.py exactly.
    
    Preprocessing steps (fitted on TRAIN, applied to VAL):
    1. Age → StandardScaler → Age_normalized
    2. Sex (Male/Female) → Map to 1/0 → Map to 1/-1 → Sex_encoded
    3. SmokingStatus → Map to numerical → One-hot encode → Center → Smoking_0, Smoking_1, Smoking_2
    
    Returns:
        train_df, val_df: DataFrames with NEW demographic columns
        new_demo_cols: List of new column names [Age_normalized, Sex_encoded, Smoking_0, Smoking_1, Smoking_2]
    """
    from sklearn.preprocessing import StandardScaler
    
    new_demo_cols = []
    
    # === 1. AGE ===
    if 'Age' in train_df.columns:
        age_scaler = StandardScaler()
        age_scaler.fit(train_df[['Age']].values)
        train_df['Age_normalized'] = age_scaler.transform(train_df[['Age']].values).flatten()
        val_df['Age_normalized'] = age_scaler.transform(val_df[['Age']].values).flatten()
        new_demo_cols.append('Age_normalized')
    
    # === 2. SEX ===
    if 'Sex' in train_df.columns:
        # Convert string to 0/1 if needed
        if train_df['Sex'].dtype == 'object':
            train_df['Sex'] = train_df['Sex'].map({'Male': 1, 'Female': 0})
            val_df['Sex'] = val_df['Sex'].map({'Male': 1, 'Female': 0})
        
        # Then convert to centered -1/+1
        train_df['Sex_encoded'] = train_df['Sex'].map({0: -1, 1: 1})
        val_df['Sex_encoded'] = val_df['Sex'].map({0: -1, 1: 1})
        new_demo_cols.append('Sex_encoded')
    
    # === 3. SMOKING STATUS ===
    if 'SmokingStatus' in train_df.columns:
        # Convert string to numerical if needed
        if train_df['SmokingStatus'].dtype == 'object':
            smoking_map = {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}
            train_df['SmokingStatus'] = train_df['SmokingStatus'].map(smoking_map)
            val_df['SmokingStatus'] = val_df['SmokingStatus'].map(smoking_map)
        
        # One-hot encode
        train_smoking_dummies = pd.get_dummies(train_df['SmokingStatus'], prefix='Smoking', dtype=float)
        val_smoking_dummies = pd.get_dummies(val_df['SmokingStatus'], prefix='Smoking', dtype=float)
        
        # Center binary features (from [0,1] to [-0.5, 0.5])
        train_smoking_dummies = train_smoking_dummies - 0.5
        val_smoking_dummies = val_smoking_dummies - 0.5
        
        # Add to dataframes
        smoking_cols = sorted(train_smoking_dummies.columns.tolist())
        for col in smoking_cols:
            train_df[col] = train_smoking_dummies[col]
            if col in val_smoking_dummies.columns:
                val_df[col] = val_smoking_dummies[col]
            else:
                val_df[col] = -0.5  # Missing category
        
        new_demo_cols.extend(smoking_cols)
    
    print(f"    ✓ Demographics: {len(new_demo_cols)} features [{', '.join(new_demo_cols)}]")
    return train_df, val_df, new_demo_cols


def preprocess_fvc_baseline_for_fold(train_df, val_df):
    """
    Normalize FVC baseline matching training code exactly.
    
    Returns:
        train_df, val_df: DataFrames with FVC_normalized column
    """
    if 'baselinefvc' in train_df.columns:
        fvc_scaler = StandardScaler()
        fvc_scaler.fit(train_df[['baselinefvc']].values)
        train_df['FVC_normalized'] = fvc_scaler.transform(train_df[['baselinefvc']].values).flatten()
        val_df['FVC_normalized'] = fvc_scaler.transform(val_df[['baselinefvc']].values).flatten()
        print(f"    ✓ FVC baseline normalized: baselinefvc → FVC_normalized")
    
    return train_df, val_df


def reconstruct_validation_data(fold_idx, fold_splits, gt_df, patient_features_df, embeddings_df, preprocessing):
    """
    Reconstruct validation data for a specific fold using saved preprocessing.
    
    CRITICAL: Must replicate training preprocessing EXACTLY:
    1. Build feature matrix
    2. Preprocess demographics (fit on train, apply to val) → Creates 5 features from 3
    3. Preprocess FVC baseline (fit on train, apply to val)
    4. Apply saved PCA (if used)
    5. Apply saved StandardScaler
    
    Returns:
        X_val: Validation features (after preprocessing)
        y_val: Validation labels
    """
    
    ablation_config = preprocessing['ablation_config']
    
    # Get patient-level features (de-duplicate)
    patient_level_df = patient_features_df.groupby('Patient').first().reset_index()
    
    # Build feature matrix according to ablation config
    if ablation_config['cnn'] is not None and embeddings_df is not None:
        combined_df = patient_level_df.merge(
            embeddings_df[['patient_id', 'embeddings']],
            left_on='Patient',
            right_on='patient_id',
            how='inner'
        )
    else:
        combined_df = patient_level_df.copy()
    
    # Split into train and validation sets
    train_ids = fold_splits['train']
    val_ids = fold_splits['val']
    
    train_df = combined_df[combined_df['Patient'].isin(train_ids)].copy()
    val_df = combined_df[combined_df['Patient'].isin(val_ids)].copy()
    
    # Merge FVC baseline if needed (before demographics preprocessing)
    if ablation_config.get('fvc_baseline', False):
        fvc_data = gt_df[['PatientID', 'baselinefvc']].copy()
        train_df = train_df.merge(fvc_data, left_on='Patient', right_on='PatientID', how='left')
        val_df = val_df.merge(fvc_data, left_on='Patient', right_on='PatientID', how='left')
        train_df = train_df.drop(columns=['PatientID'], errors='ignore')
        val_df = val_df.drop(columns=['PatientID'], errors='ignore')
    
    # === STEP 1: Preprocess demographics (creates 5 features from 3) ===
    if ablation_config['demo']:
        train_df, val_df, new_demo_cols = preprocess_demographics_for_fold(train_df, val_df)
    else:
        new_demo_cols = []
    
    # === STEP 2: Preprocess FVC baseline ===
    if ablation_config.get('fvc_baseline', False) and 'baselinefvc' in val_df.columns:
        train_df, val_df = preprocess_fvc_baseline_for_fold(train_df, val_df)
    
    # Merge labels
    val_df = val_df.merge(gt_df[['PatientID', 'has_progressed']], 
                          left_on='Patient', right_on='PatientID', how='left')
    
    # Build feature column list
    feature_cols = []
    
    # Add CNN embeddings if used
    if ablation_config['cnn'] is not None and 'embeddings' in val_df.columns:
        # Unpack embeddings array into individual columns
        embedding_dim = val_df['embeddings'].iloc[0].shape[0]
        embedding_cols = [f'cnn_emb_{i}' for i in range(embedding_dim)]
        embedding_matrix = np.vstack(val_df['embeddings'].values)
        for i, col in enumerate(embedding_cols):
            val_df[col] = embedding_matrix[:, i]
        
        feature_cols.extend(embedding_cols)
    
    # Add hand-crafted features
    if ablation_config['hand']:
        hand_cols = [
            'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
            'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60', 
            'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
        ]
        feature_cols.extend(hand_cols)
    
    # Add preprocessed demographics (5 features instead of original 3)
    if ablation_config['demo']:
        feature_cols.extend(new_demo_cols)
    
    # Add FVC_normalized (not baselinefvc!)
    if ablation_config.get('fvc_baseline', False):
        feature_cols.append('FVC_normalized')
    
    # Extract features and labels
    print(f"    Final features: {len(feature_cols)}")
    print(f"    Expected by scaler: {len(preprocessing['feature_names'])}")
    
    # Check for missing features
    missing_features = [f for f in feature_cols if f not in val_df.columns]
    if missing_features:
        print(f"    ⚠️ WARNING: Missing features in val_df: {missing_features}")
    
    X_val = val_df[feature_cols].values
    y_val = val_df['has_progressed'].values.astype(int)
    
    print(f"    X_val shape: {X_val.shape}")
    
    # Apply saved preprocessing (CRITICAL: same order as training)
    # 1. PCA (if used) - applied BEFORE scaling in training
    if preprocessing['pca_model'] is not None:
        X_val = preprocessing['pca_model'].transform(X_val)
        print(f"    After PCA: {X_val.shape}")
    
    # 2. Scaling (always applied)
    X_val = preprocessing['scaler'].transform(X_val)
    print(f"    After scaling: {X_val.shape}")
    
    return X_val, y_val


def load_fold_data(fold_dir, fold_idx, fold_splits, gt_df, patient_features_df, embeddings_df):
    """
    Load saved model and reconstruct validation data for a single fold.
    
    Returns:
        model: LightGBM Booster
        X_val: Validation features
        y_val: Validation labels
        preprocessing: Dict with scaler, feature_cols, etc.
    """
    model_path = fold_dir / "lightgbm_model.txt"
    preprocessing_path = fold_dir / "preprocessing.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n  Loading fold_{fold_idx}...")
    
    # Load LightGBM model
    model = lgb.Booster(model_file=str(model_path))
    
    # Load preprocessing info
    with open(preprocessing_path, 'rb') as f:
        preprocessing = pickle.load(f)
    
    # Reconstruct validation data
    X_val, y_val = reconstruct_validation_data(
        fold_idx, fold_splits, gt_df, patient_features_df, embeddings_df, preprocessing
    )
    
    print(f"    ✓ X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    return model, X_val, y_val, preprocessing


def compute_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """
    Compute all classification metrics at a specific threshold.
    
    Returns:
        dict with accuracy, precision, recall, f1, specificity
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    return metrics


# =============================================================================
# THRESHOLD SWEEP ANALYSIS
# =============================================================================

def sweep_thresholds(fold_results, thresholds):
    """
    Sweep thresholds and compute metrics for each fold.
    
    Args:
        fold_results: List of dicts with 'fold_idx', 'y_val', 'y_pred'
        thresholds: Array of thresholds to test
        
    Returns:
        results_df: DataFrame with metrics for each threshold and fold
        aggregated_df: DataFrame with mean metrics across folds for each threshold
    """
    
    results = []
    
    print(f"\n{'='*70}")
    print("SWEEPING THRESHOLDS ACROSS VALIDATION FOLDS")
    print(f"{'='*70}")
    print(f"Thresholds: {len(thresholds)} ({thresholds[0]:.2f} to {thresholds[-1]:.2f}, step {thresholds[1]-thresholds[0]:.2f})")
    print(f"Folds: {len(fold_results)}")
    
    for fold_result in fold_results:
        fold_idx = fold_result['fold_idx']
        y_true = fold_result['y_val']
        y_pred_proba = fold_result['y_pred']
        
        for threshold in thresholds:
            metrics = compute_metrics_at_threshold(y_true, y_pred_proba, threshold)
            metrics['fold'] = f"fold_{fold_idx}"
            results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Aggregate across folds
    metric_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    aggregated = []
    
    for threshold in thresholds:
        threshold_data = results_df[results_df['threshold'] == threshold]
        
        agg_metrics = {'threshold': threshold}
        for metric in metric_cols:
            agg_metrics[f'{metric}_mean'] = threshold_data[metric].mean()
            agg_metrics[f'{metric}_std'] = threshold_data[metric].std()
        
        aggregated.append(agg_metrics)
    
    aggregated_df = pd.DataFrame(aggregated)
    
    print(f"\n✓ Sweep complete: {len(results_df)} measurements ({len(fold_results)} folds × {len(thresholds)} thresholds)")
    
    return results_df, aggregated_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_threshold_curves(aggregated_df, output_dir):
    """
    Plot comprehensive threshold optimization curves.
    
    4-panel plot:
    1. F1 vs threshold (main metric)
    2. Precision-Recall tradeoff
    3. Accuracy & Specificity
    4. All metrics overlaid
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    thresholds = aggregated_df['threshold'].values
    
    # Panel 1: F1 Score
    ax = axes[0, 0]
    ax.plot(thresholds, aggregated_df['f1_mean'], 'b-', linewidth=2, label='F1 Score')
    ax.fill_between(thresholds, 
                     aggregated_df['f1_mean'] - aggregated_df['f1_std'],
                     aggregated_df['f1_mean'] + aggregated_df['f1_std'],
                     alpha=0.2)
    
    # Mark optimal F1
    optimal_idx = aggregated_df['f1_mean'].idxmax()
    optimal_threshold = aggregated_df.loc[optimal_idx, 'threshold']
    optimal_f1 = aggregated_df.loc[optimal_idx, 'f1_mean']
    
    ax.plot(optimal_threshold, optimal_f1, 'ro', markersize=12, 
            label=f'Optimal: {optimal_threshold:.3f} (F1={optimal_f1:.3f})')
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    
    # Panel 2: Precision-Recall Tradeoff
    ax = axes[0, 1]
    ax.plot(thresholds, aggregated_df['precision_mean'], 'g-', linewidth=2, label='Precision', marker='o', markersize=3)
    ax.plot(thresholds, aggregated_df['recall_mean'], 'b-', linewidth=2, label='Recall', marker='s', markersize=3)
    ax.fill_between(thresholds,
                     aggregated_df['precision_mean'] - aggregated_df['precision_std'],
                     aggregated_df['precision_mean'] + aggregated_df['precision_std'],
                     alpha=0.15, color='g')
    ax.fill_between(thresholds,
                     aggregated_df['recall_mean'] - aggregated_df['recall_std'],
                     aggregated_df['recall_mean'] + aggregated_df['recall_std'],
                     alpha=0.15, color='b')
    
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.5, label=f'Optimal ({optimal_threshold:.3f})')
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    
    # Panel 3: Accuracy & Specificity
    ax = axes[1, 0]
    ax.plot(thresholds, aggregated_df['accuracy_mean'], 'purple', linewidth=2, label='Accuracy', marker='d', markersize=3)
    ax.plot(thresholds, aggregated_df['specificity_mean'], 'orange', linewidth=2, label='Specificity', marker='^', markersize=3)
    ax.fill_between(thresholds,
                     aggregated_df['accuracy_mean'] - aggregated_df['accuracy_std'],
                     aggregated_df['accuracy_mean'] + aggregated_df['accuracy_std'],
                     alpha=0.15, color='purple')
    ax.fill_between(thresholds,
                     aggregated_df['specificity_mean'] - aggregated_df['specificity_std'],
                     aggregated_df['specificity_mean'] + aggregated_df['specificity_std'],
                     alpha=0.15, color='orange')
    
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.5, label=f'Optimal ({optimal_threshold:.3f})')
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy & Specificity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    
    # Panel 4: All Metrics Overlaid
    ax = axes[1, 1]
    ax.plot(thresholds, aggregated_df['f1_mean'], 'b-', linewidth=2, label='F1', alpha=0.8)
    ax.plot(thresholds, aggregated_df['precision_mean'], 'g-', linewidth=2, label='Precision', alpha=0.8)
    ax.plot(thresholds, aggregated_df['recall_mean'], 'r-', linewidth=2, label='Recall', alpha=0.8)
    ax.plot(thresholds, aggregated_df['accuracy_mean'], 'purple', linewidth=2, label='Accuracy', alpha=0.8)
    ax.plot(thresholds, aggregated_df['specificity_mean'], 'orange', linewidth=2, label='Specificity', alpha=0.8)
    
    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([thresholds[0], thresholds[-1]])
    
    plt.tight_layout()
    save_path = output_dir / "threshold_optimization_curves.png"
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Threshold curves saved: {save_path}")
    
    return optimal_threshold, optimal_f1


def plot_confusion_matrices(fold_results, optimal_threshold, output_dir):
    """
    Plot confusion matrices for each fold + aggregate at optimal threshold.
    """
    
    n_folds = len(fold_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_y_true = []
    all_y_pred = []
    
    # Individual fold confusion matrices
    for i, fold_result in enumerate(fold_results):
        ax = axes[i]
        
        y_true = fold_result['y_val']
        y_pred_proba = fold_result['y_pred']
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['No Prog', 'Prog'],
                    yticklabels=['No Prog', 'Prog'])
        
        tn, fp, fn, tp = cm.ravel()
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        ax.set_title(f"Fold {fold_result['fold_idx']}\nF1={f1:.3f}, Threshold={optimal_threshold:.3f}", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Aggregate confusion matrix
    ax = axes[n_folds]
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm_agg = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    
    sns.heatmap(cm_agg, annot=True, fmt='d', cmap='RdYlGn', ax=ax, cbar=False,
                xticklabels=['No Prog', 'Prog'],
                yticklabels=['No Prog', 'Prog'],
                annot_kws={'size': 14, 'weight': 'bold'})
    
    tn, fp, fn, tp = cm_agg.ravel()
    f1_agg = f1_score(all_y_true, all_y_pred, zero_division=0)
    accuracy = accuracy_score(all_y_true, all_y_pred)
    
    ax.set_title(f"AGGREGATE (All Folds)\nF1={f1_agg:.3f}, Acc={accuracy:.3f}, Threshold={optimal_threshold:.3f}", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    
    # Hide extra subplot if 5 folds
    if n_folds < len(axes) - 1:
        for idx in range(n_folds + 1, len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "confusion_matrices_optimal_threshold.png"
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrices saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution for threshold optimization."""
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION FOR BEST LIGHTGBM MODEL")
    print("="*80)
    print(f"Model: {CONFIG['model_dir'].name}")
    print(f"Threshold range: [{CONFIG['threshold_min']}, {CONFIG['threshold_max']}], step {CONFIG['threshold_step']}")
    
    # Create output directory
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    gt_df = load_ground_truth()
    kfold_splits = load_kfold_splits()
    patient_features_df, embeddings_df = load_patient_features_and_embeddings()
    
    # Load all folds
    print(f"\n{'='*70}")
    print(f"LOADING FOLD MODELS AND VALIDATION DATA")
    print(f"{'='*70}")
    
    fold_results = []
    
    for fold_idx in sorted(kfold_splits.keys()):
        fold_dir = CONFIG['model_dir'] / f"fold_fold{fold_idx}"
        
        if not fold_dir.exists():
            print(f"  ⚠️ Skipping fold {fold_idx}: directory not found")
            continue
        
        try:
            model, X_val, y_val, preprocessing = load_fold_data(
                fold_dir, fold_idx, kfold_splits[fold_idx], 
                gt_df, patient_features_df, embeddings_df
            )
            
            # Get predictions
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            fold_results.append({
                'fold_idx': fold_idx,
                'y_val': y_val,
                'y_pred': y_pred,
                'model': model
            })
            
        except Exception as e:
            print(f"  ❌ Error loading fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(fold_results) == 0:
        print("\n❌ ERROR: No folds loaded successfully!")
        return
    
    print(f"\n✓ Successfully loaded {len(fold_results)} folds")
    
    # Generate threshold sweep
    thresholds = np.arange(CONFIG['threshold_min'], CONFIG['threshold_max'] + CONFIG['threshold_step'], CONFIG['threshold_step'])
    
    # Sweep thresholds
    results_df, aggregated_df = sweep_thresholds(fold_results, thresholds)
    
    # Save results
    results_df.to_csv(CONFIG['output_dir'] / "threshold_sweep_detailed.csv", index=False)
    aggregated_df.to_csv(CONFIG['output_dir'] / "threshold_sweep_aggregated.csv", index=False)
    
    print(f"\n✓ Results saved:")
    print(f"  - threshold_sweep_detailed.csv")
    print(f"  - threshold_sweep_aggregated.csv")
    
    # Find optimal threshold
    optimal_idx = aggregated_df['f1_mean'].idxmax()
    optimal_threshold = aggregated_df.loc[optimal_idx, 'threshold']
    optimal_metrics = aggregated_df.loc[optimal_idx]
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLD (maximizes F1)")
    print(f"{'='*70}")
    print(f"Threshold: {optimal_threshold:.3f}")
    print(f"F1:         {optimal_metrics['f1_mean']:.3f} ± {optimal_metrics['f1_std']:.3f}")
    print(f"Precision:  {optimal_metrics['precision_mean']:.3f} ± {optimal_metrics['precision_std']:.3f}")
    print(f"Recall:     {optimal_metrics['recall_mean']:.3f} ± {optimal_metrics['recall_std']:.3f}")
    print(f"Accuracy:   {optimal_metrics['accuracy_mean']:.3f} ± {optimal_metrics['accuracy_std']:.3f}")
    print(f"Specificity: {optimal_metrics['specificity_mean']:.3f} ± {optimal_metrics['specificity_std']:.3f}")
    
    # Save summary
    summary_data = {
        'optimal_threshold': optimal_threshold,
        'f1_mean': optimal_metrics['f1_mean'],
        'f1_std': optimal_metrics['f1_std'],
        'precision_mean': optimal_metrics['precision_mean'],
        'precision_std': optimal_metrics['precision_std'],
        'recall_mean': optimal_metrics['recall_mean'],
        'recall_std': optimal_metrics['recall_std'],
        'accuracy_mean': optimal_metrics['accuracy_mean'],
        'accuracy_std': optimal_metrics['accuracy_std'],
        'specificity_mean': optimal_metrics['specificity_mean'],
        'specificity_std': optimal_metrics['specificity_std'],
        'n_folds': len(fold_results),
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(CONFIG['output_dir'] / "optimization_summary.csv", index=False)
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_threshold_curves(aggregated_df, CONFIG['output_dir'])
    plot_confusion_matrices(fold_results, optimal_threshold, CONFIG['output_dir'])
    
    # Fold-level metrics at optimal threshold
    fold_metrics = []
    for fold_result in fold_results:
        metrics = compute_metrics_at_threshold(fold_result['y_val'], fold_result['y_pred'], optimal_threshold)
        metrics['fold'] = f"fold_{fold_result['fold_idx']}"
        fold_metrics.append(metrics)
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(CONFIG['output_dir'] / "fold_metrics_optimal_threshold.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ THRESHOLD OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Expected F1: {optimal_metrics['f1_mean']:.3f} ± {optimal_metrics['f1_std']:.3f}")


if __name__ == "__main__":
    main()
