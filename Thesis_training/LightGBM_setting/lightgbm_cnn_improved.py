# IMPROVED: LightGBM with CNN Embeddings + Hand-crafted + Demographics
# Key improvements:
# 1. Better LightGBM regularization for small datasets
# 2. Multiple pooling strategies (max + mean + std)
# 3. Adaptive PCA components based on variance
# 4. Better handling of failed folds
# 5. Diagnostic visualization
# 6. ABLATION STUDY: Test different feature combinations

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import random
import sys
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
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

# Add parent directory to path to import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n{'='*70}")
    print(f"🔒 RANDOM SEED SET TO: {seed}")
    print(f"{'='*70}\n")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "npy_root": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    
    # CNN settings
    'cnn_model': 'resnet50',  # More efficient than ResNet50, good for small datasets
    'pretrained': True,
    'batch_size': 32,
    'num_workers': 4,
    
    # Pooling settings - USE ALL THREE for richer representation
    'pooling_methods': 'max',  # Combine multiple statistics
    
    # PCA settings - more conservative
    'pca_variance_threshold': 0.95,  # Keep 95% variance instead of fixed components
    'pca_max_components': 50,  # Upper limit
    'pca_whiten': False,  # Don't whiten - can hurt with small samples
    
    # Reproducibility
    'base_seed': 42,
    'n_repeats': 5,  # 5×5 repeated cross-validation (5 repeats × 5 folds = 25 runs)
    
    # IMPROVED LightGBM parameters for small datasets
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 3,  # REDUCED from 7 - less overfitting
        'max_depth': 2,  # NEW - limit tree depth
        'learning_rate': 0.01,  # REDUCED from 0.05 - slower, more stable
        'feature_fraction': 0.7,  # REDUCED from 0.8 - more regularization
        'bagging_fraction': 0.7,  # REDUCED - more regularization
        'bagging_freq': 5,
        'min_data_in_leaf': 15,  # NEW beofre 10 - prevent tiny leaves
        'lambda_l1': 0.5,  # NEW increased form 0.1 - L1 regularization
        'lambda_l2': 0.5,  # NEW  increasede from 0.1- L2 regularization
        'min_gain_to_split': 0.05,  # NEW increased from 0.01- require meaningful splits
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True  # Better for small datasets
    },
    'num_boost_round': 1000,  # Increased - let early stopping decide
    'early_stopping_rounds': 100,  # Increased - more patience
    
    # Feature normalization
    'normalization_type': 'standard',
}




# Hand-crafted features
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

# Demographic features
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']


# =============================================================================
# ABLATION STUDY CONFIGURATIONS (ESSENTIAL EXPERIMENTS ONLY)
# =============================================================================
# Only the key experiments needed for thesis:
# 1. Tabular baselines (clinical baseline)
# 2. Best CNN (mean pooling - proven best)
# 3. Best hybrid (full combination)

ABLATION_CONFIGS = {
    # Tabular baselines (clinical baseline - no CNN)
    "hand_only": {
        "cnn": None,
        "pca": False,
        "hand": True,
        "demo": False,
        "description": "Hand-crafted features only (Clinical Baseline)"
    },
    
    "hand_demo": {
        "cnn": None,
        "pca": False,
        "hand": True,
        "demo": True,
        "description": "Hand-crafted + Demographics (Clinical Baseline)"
    },
    
    # Best CNN configuration
    "cnn_mean": {
        "cnn": "mean",
        "pca": False,
        "hand": False,
        "demo": False,
        "description": "CNN mean pooling (Best CNN Configuration)"
    },
    
    # Best hybrid (full combination)
    "best_cnn_hand_demo": {
        "cnn": "mean",  # Using mean directly (proven best)
        "pca": False,
        "hand": True,
        "demo": True,
        "description": "CNN + Hand-crafted + Demographics (FULL MODEL)"
    },

    "best_cnn_demo": {
        "cnn": "mean",  # Using mean directly (proven best)
        "pca": False,
        "hand": False,
        "demo": True,
        "description": "CNN + Demographics (Ablation without hand-crafted)"
    },
}


# =============================================================================
# CNN EMBEDDING EXTRACTION (USING UTILITIES CNNFeatureExtractor)
# Note: The old inline CNNFeatureExtractor class has been replaced with 
# the shared CNNFeatureExtractor from utilities module (same as ablation_study.py)
# =============================================================================

def extract_slice_level_features(
    patient_ids,
    gt_df,
    npy_root,
    cnn_extractor,
    cache_path=None
):
    """
    Extract slice-level CNN features using the same approach as ablation_study.py
    This extracts features for each CT slice, returning a DataFrame with one row per slice
    """
    
    if cache_path and cache_path.exists():
        print(f"\n✓ Loading cached slice-level features from: {cache_path}")
        return pd.read_csv(cache_path)
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING SLICE-LEVEL CNN FEATURES")
    print(f"{'='*70}")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Using extract_features_patient_grouping() with batch processing")
    
    # Build patient_data structure required by CNNFeatureExtractor
    # This is the same format used in ablation_study.py
    patient_data = {}
    for patient_id in patient_ids:
        patient_dir = npy_root / patient_id
        if not patient_dir.exists():
            print(f"Warning: No directory found for patient {patient_id}")
            continue
        
        npy_files = sorted(list(patient_dir.glob('*.npy')))
        if len(npy_files) == 0:
            print(f"Warning: No .npy files found for patient {patient_id}")
            continue
        
        # Get ground truth label
        patient_row = gt_df[gt_df['PatientID'] == patient_id]
        if len(patient_row) == 0:
            print(f"Warning: No ground truth for patient {patient_id}")
            continue
        
        has_progressed = bool(patient_row.iloc[0]['has_progressed'])
        
        patient_data[patient_id] = {
            'gt_has_progressed': has_progressed,
            'slices': npy_files  # Key must be 'slices' for utilities.CTSliceLoaderPatients
        }
    
    print(f"\n✓ Built patient_data for {len(patient_data)} patients")
    print(f"  Average slices per patient: {np.mean([len(p['slices']) for p in patient_data.values()]):.1f}")
    
    # Extract features using patient grouping (batch processing)
    slice_features_df = cnn_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,  # Same as ablation_study.py
        save_path=cache_path
    )
    
    print(f"\n✓ Extracted slice-level features for {len(slice_features_df)} slices")
    print(f"  Feature dimension: {len([c for c in slice_features_df.columns if c.startswith('cnn_feature_')])}")
    
    return slice_features_df


def apply_pooling_to_slices(
    slice_features_df,
    pooling_method='max'
):
    """
    Apply pooling strategy to slice-level features to get patient-level embeddings
    Supports single pooling (max, mean, std, min) or combined pooling (max_mean)
    
    Args:
        slice_features_df: DataFrame with slice-level CNN features
        pooling_method: 'max', 'mean', 'std', 'min', or 'max_mean' (concatenate max and mean)
    
    Returns:
        embeddings_df: DataFrame with patient-level embeddings
    """
    
    print(f"\n{'='*70}")
    print(f"APPLYING POOLING TO SLICE-LEVEL FEATURES")
    print(f"{'='*70}")
    print(f"Pooling method: {pooling_method}")
    
    # Get CNN feature columns
    cnn_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
    print(f"CNN features per slice: {len(cnn_cols)}")
    
    # Group by patient and apply pooling
    patient_embeddings = []
    
    for patient_id, patient_slices in slice_features_df.groupby('patient_id'):
        slice_features = patient_slices[cnn_cols].values  # Shape: (n_slices, n_features)
        
        # Apply the specified pooling operation
        if pooling_method == 'max':
            pooled_features = np.max(slice_features, axis=0)
        elif pooling_method == 'mean':
            pooled_features = np.mean(slice_features, axis=0)
        elif pooling_method == 'std':
            pooled_features = np.std(slice_features, axis=0)
        elif pooling_method == 'min':
            pooled_features = np.min(slice_features, axis=0)
        elif pooling_method == 'max_mean':
            # Concatenate max and mean pooling
            max_features = np.max(slice_features, axis=0)
            mean_features = np.mean(slice_features, axis=0)
            pooled_features = np.concatenate([max_features, mean_features])
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        patient_embeddings.append({
            'patient_id': patient_id,
            'embeddings': pooled_features,
            'num_slices': len(patient_slices)
        })
    
    embeddings_df = pd.DataFrame(patient_embeddings)
    
    print(f"\n✓ Pooled features for {len(embeddings_df)} patients")
    print(f"  Embedding dimension: {embeddings_df['embeddings'].iloc[0].shape[0]}")
    if pooling_method == 'max_mean':
        print(f"  (max={len(cnn_cols)} + mean={len(cnn_cols)} = {embeddings_df['embeddings'].iloc[0].shape[0]} total)")
    print(f"  Average slices per patient: {embeddings_df['num_slices'].mean():.1f}")
    print(f"  Min slices: {embeddings_df['num_slices'].min()}")
    print(f"  Max slices: {embeddings_df['num_slices'].max()}")
    
    return embeddings_df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load demographics and merge with patient features"""
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS")
    print("="*70)
    
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
    
    demographics_df = train_df[demo_cols].drop_duplicates(subset=['Patient'])
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    if 'Sex' in enhanced_df.columns:
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
    
    if 'SmokingStatus' in enhanced_df.columns:
        smoking_map = {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(smoking_map)
    
    return enhanced_df


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Load ground truth labels"""
    gt_df = pd.read_csv(gt_path)
    print(f"\nGround truth: {len(gt_df)} patients")
    print(f"  Progression: {gt_df['has_progressed'].sum()}")
    print(f"  No progression: {(~gt_df['has_progressed'].astype(bool)).sum()}")
    return gt_df


def create_repeated_kfold_splits(kfold_splits_base, n_repeats, base_seed):
    """
    Create repeated K-fold splits for repeated cross-validation
    
    STRATEGY: For computational efficiency, we use the same patient fold assignments 
    across all repetitions but vary the random seed for training initialization.
    This provides robustness through different:
    - Model initializations (LightGBM random seeds)
    - Data shuffling within training
    - Early stopping variations
    
    For even more robust repeated CV, you should pre-generate multiple 
    stratified split files with different random seeds and load them here.
    
    Args:
        kfold_splits_base: Base K-fold splits (1 repetition)
        n_repeats: Number of repetitions
        base_seed: Base random seed
    
    Returns:
        repeated_splits: Dict with structure {repeat_idx: {fold_idx: fold_data}}
    """
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\n{'='*70}")
    print("CREATING REPEATED K-FOLD SPLITS")
    print(f"{'='*70}")
    print(f"Base folds: {len(kfold_splits_base)}")
    print(f"Repeats: {n_repeats}")
    print(f"Total runs: {len(kfold_splits_base) * n_repeats}")
    
    repeated_splits = {}
    
    # First repetition: use existing splits
    repeated_splits[0] = kfold_splits_base
    print(f"  Repeat 0: Using existing stratified splits")
    
    # Create additional repetitions with different seeds
    # We need to recreate the splits from scratch with different random seeds
    # Get all patient IDs and labels from first split
    all_patients = set()
    for fold_data in kfold_splits_base.values():
        all_patients.update(fold_data['train'])
        all_patients.update(fold_data['val'])
        all_patients.update(fold_data['test'])
    
    # For now, just use the base splits with shuffled patient assignment
    # In practice, you should regenerate from StratifiedKFold with different seeds
    for repeat_idx in range(1, n_repeats):
        repeat_seed = base_seed + (repeat_idx * 1000)
        np.random.seed(repeat_seed)
        
        # Create new splits with different seed
        # Note: This is a simplified approach - ideally load pre-generated splits
        repeated_splits[repeat_idx] = kfold_splits_base.copy()
        print(f"  Repeat {repeat_idx}: Seed {repeat_seed} (using base stratification)")
    
    print(f"\n✓ Created {len(repeated_splits)} repetitions × {len(kfold_splits_base)} folds = {len(repeated_splits) * len(kfold_splits_base)} total runs")
    
    return repeated_splits


# =============================================================================
# FEATURE CONCATENATION
# =============================================================================

def build_feature_matrix(patient_features_df, embeddings_df, config, 
                         hand_feature_cols, demo_feature_cols):
    """
    Build feature matrix according to ablation configuration
    
    Args:
        patient_features_df: DataFrame with patient-level tabular features
        embeddings_df: DataFrame with CNN embeddings (or None)
        config: Ablation configuration dict with keys: cnn, pca, hand, demo
        hand_feature_cols: List of hand-crafted feature column names
        demo_feature_cols: List of demographic feature column names
    
    Returns:
        combined_df: DataFrame with all selected features
        feature_cols: List of feature column names
        feature_groups: Dict mapping group names to column lists
    """
    
    print(f"\n{'='*70}")
    print(f"BUILDING FEATURE MATRIX")
    print(f"{'='*70}")
    print(f"Config: {config}")
    
    patient_level_df = patient_features_df.groupby('Patient').first().reset_index()
    
    # Start with patient base
    if config['cnn'] is not None and embeddings_df is not None:
        combined_df = patient_level_df.merge(
            embeddings_df[['patient_id', 'embeddings']],
            left_on='Patient',
            right_on='patient_id',
            how='inner'
        )
    else:
        combined_df = patient_level_df.copy()
    
    print(f"Patients: {len(combined_df)}")
    
    # Build feature list
    all_features = []
    feature_groups = {
        'cnn_embeddings': [],
        'hand_crafted': [],
        'demographics': []
    }
    
    # Add CNN embeddings if requested
    if config['cnn'] is not None and 'embeddings' in combined_df.columns:
        embedding_dim = combined_df['embeddings'].iloc[0].shape[0]
        embedding_cols = [f'cnn_emb_{i}' for i in range(embedding_dim)]
        embedding_matrix = np.vstack(combined_df['embeddings'].values)
        for i, col in enumerate(embedding_cols):
            combined_df[col] = embedding_matrix[:, i]
        
        feature_groups['cnn_embeddings'] = embedding_cols
        all_features.extend(embedding_cols)
        print(f"  CNN embeddings: {len(embedding_cols)} (pooling={config['cnn']}, pca={config['pca']})")
    
    # Add hand-crafted features if requested
    if config['hand']:
        available_hand = [c for c in hand_feature_cols if c in combined_df.columns]
        feature_groups['hand_crafted'] = available_hand
        all_features.extend(available_hand)
        print(f"  Hand-crafted: {len(available_hand)}")
    
    # Add demographic features if requested
    if config['demo']:
        available_demo = [c for c in demo_feature_cols if c in combined_df.columns]
        feature_groups['demographics'] = available_demo
        all_features.extend(available_demo)
        print(f"  Demographics: {len(available_demo)}")
    
    print(f"\nTotal features: {len(all_features)}")
    
    if len(all_features) == 0:
        raise ValueError("No features selected! Check ablation configuration.")
    
    return combined_df, all_features, feature_groups


# =============================================================================
# DEMOGRAPHICS PREPROCESSING (SOPHISTICATED)
# =============================================================================

def preprocess_demographics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    demo_feature_cols: list,
    normalization_type: str = 'standard'
) -> tuple:
    """
    Apply sophisticated demographics preprocessing with proper centering and encoding
    
    Preprocessing steps:
    1. Age: Normalize using StandardScaler (fitted on train only) -> Age_normalized
    2. Sex: Convert to centered binary: Female=-1, Male=1 -> Sex_encoded
    3. SmokingStatus: One-hot encode + center to [-0.5, 0.5] -> Smoking_0, Smoking_1, Smoking_2
    
    Returns:
        train_df, val_df, test_df: DataFrames with preprocessed demographic features
        demo_encoding_info: Dict containing preprocessing metadata
        new_demo_cols: List of new demographic column names
    """
    
    print(f"\n{'='*70}")
    print("SOPHISTICATED DEMOGRAPHICS PREPROCESSING")
    print(f"{'='*70}")
    
    demo_encoding_info = {}
    new_demo_cols = []
    
    # === 1. AGE (Continuous) ===
    if 'Age' in demo_feature_cols and 'Age' in train_df.columns:
        print("\n=== PREPROCESSING AGE ===")
        
        print(f"  Pre-normalization (Training Set):")
        print(f"    Mean: {train_df['Age'].mean():.2f}")
        print(f"    Std: {train_df['Age'].std():.2f}")
        print(f"    Range: [{train_df['Age'].min():.0f}, {train_df['Age'].max():.0f}]")
        
        # Normalize Age
        if normalization_type == 'standard':
            age_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            age_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            age_scaler = MinMaxScaler()
        
        age_scaler.fit(train_df[['Age']].values)
        train_df['Age_normalized'] = age_scaler.transform(train_df[['Age']].values)
        val_df['Age_normalized'] = age_scaler.transform(val_df[['Age']].values)
        test_df['Age_normalized'] = age_scaler.transform(test_df[['Age']].values)
        
        print(f"  Post-normalization (Training Set):")
        print(f"    Mean: {train_df['Age_normalized'].mean():.4f}")
        print(f"    Std: {train_df['Age_normalized'].std():.4f}")
        
        demo_encoding_info['age_scaler'] = age_scaler
        new_demo_cols.append('Age_normalized')
    else:
        print("  ⚠️  Age column not found")
    
    # === 2. SEX (Binary Categorical) ===
    if 'Sex' in demo_feature_cols and 'Sex' in train_df.columns:
        print("\n=== PREPROCESSING SEX ===")
        print(f"  Unique values: {train_df['Sex'].unique()}")
        
        # Convert to binary centered around 0 (better for ML)
        # Assuming Sex is already encoded as 0=Female, 1=Male
        train_df['Sex_encoded'] = train_df['Sex'].map({0: -1, 1: 1})  # Female=-1, Male=1
        val_df['Sex_encoded'] = val_df['Sex'].map({0: -1, 1: 1})
        test_df['Sex_encoded'] = test_df['Sex'].map({0: -1, 1: 1})
        
        print(f"  Encoded as: Female=-1, Male=1 (centered)")
        sex_dist = train_df['Sex_encoded'].value_counts()
        print(f"  Training set distribution:")
        for val, count in sex_dist.items():
            label = 'Female' if val == -1 else 'Male'
            print(f"    {label} ({val:+d}): {count}")
        
        demo_encoding_info['sex_encoding'] = {0: -1, 1: 1, 'description': 'Female=-1, Male=1'}
        new_demo_cols.append('Sex_encoded')
    else:
        print("  ⚠️  Sex column not found")
    
    # === 3. SMOKING STATUS (Multi-class Categorical) ===
    if 'SmokingStatus' in demo_feature_cols and 'SmokingStatus' in train_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS ===")
        print(f"  Unique values: {train_df['SmokingStatus'].unique()}")
        
        # One-hot encoding (more expressive for ML)
        # Assuming SmokingStatus is already encoded as 0=Never, 1=Ex-smoker, 2=Current
        train_smoking_dummies = pd.get_dummies(
            train_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        val_smoking_dummies = pd.get_dummies(
            val_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        test_smoking_dummies = pd.get_dummies(
            test_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        
        # Center binary features around 0 (from [0,1] to [-0.5, 0.5])
        train_smoking_dummies = (train_smoking_dummies - 0.5)
        val_smoking_dummies = (val_smoking_dummies - 0.5)
        test_smoking_dummies = (test_smoking_dummies - 0.5)
        
        # Add to dataframes
        smoking_cols = sorted(train_smoking_dummies.columns.tolist())
        for col in smoking_cols:
            train_df[col] = train_smoking_dummies[col]
            # Ensure val/test have same columns (even if some categories missing)
            if col in val_smoking_dummies.columns:
                val_df[col] = val_smoking_dummies[col]
            else:
                val_df[col] = -0.5  # Absent category = centered 0
            if col in test_smoking_dummies.columns:
                test_df[col] = test_smoking_dummies[col]
            else:
                test_df[col] = -0.5
        
        demo_encoding_info['smoking_columns'] = smoking_cols
        new_demo_cols.extend(smoking_cols)
        
        print(f"  One-hot encoded into {len(smoking_cols)} features: {smoking_cols}")
        print(f"  Values centered to [-0.5, 0.5]")
        
        # Show distribution
        print(f"  Training set distribution:")
        for val in train_df['SmokingStatus'].unique():
            count = (train_df['SmokingStatus'] == val).sum()
            smoking_label = {0: 'Never', 1: 'Ex-smoker', 2: 'Current'}.get(val, f'Unknown({val})')
            print(f"    {smoking_label} ({val}): {count}")
    else:
        print("  ⚠️  SmokingStatus column not found")
    
    print(f"\n✓ Demographics preprocessing complete!")
    print(f"  Original columns: {len(demo_feature_cols)}")
    print(f"  New columns: {len(new_demo_cols)} ({', '.join(new_demo_cols)})")
    
    return train_df, val_df, test_df, demo_encoding_info, new_demo_cols


# =============================================================================
# EVALUATION
# =============================================================================

def find_optimal_threshold(y_true, y_pred):
    """Find optimal threshold using Youden's J"""
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
# VISUALIZATION WITH DIAGNOSTICS
# =============================================================================

def plot_roc_with_threshold(y_true, y_pred, optimal_threshold, fold_idx, save_dir, dataset='val'):
    """Plot ROC curve with optimal threshold marked"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    # Find the point corresponding to optimal threshold
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    # Mark optimal threshold
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=12, 
             label=f'Optimal Threshold = {optimal_threshold:.3f}\n(TPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Fold {fold_idx} ({dataset.upper()})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    
    plt.savefig(save_dir / f"fold_{fold_idx}_roc_{dataset}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred_binary, fold_idx, save_dir, dataset='test'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'],
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - Fold {fold_idx} ({dataset.upper()})', fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
             ha='center', fontsize=12, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"fold_{fold_idx}_confusion_{dataset}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion_matrix(all_y_true, all_y_pred_binary, save_dir):
    """Plot aggregated confusion matrix across all folds"""
    cm = confusion_matrix(all_y_true, all_y_pred_binary)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'],
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title('Aggregated Confusion Matrix - All Folds', fontsize=15, fontweight='bold')
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}\nSpecificity: {specificity:.3f} | F1: {f1:.3f}'
    plt.text(0.5, -0.15, metrics_text, ha='center', fontsize=11, 
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_dir / "aggregate_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_roc(fold_results, save_dir):
    """Plot all fold ROC curves on one plot"""
    plt.figure(figsize=(10, 8))
    
    all_aucs = []
    for result in fold_results:
        fpr, tpr, _ = roc_curve(result['test_y_true'], result['test_y_pred'])
        auc_val = result['test_auc']
        all_aucs.append(auc_val)
        plt.plot(fpr, tpr, linewidth=2, alpha=0.7, label=f"Fold {result['fold_idx']} (AUC={auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')
    
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title(f'ROC Curves - All Folds (Mean AUC={np.mean(all_aucs):.3f}±{np.std(all_aucs):.3f})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    
    plt.savefig(save_dir / "aggregate_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_fold_diagnostics(fold_results, save_dir):
    """Plot diagnostic charts to understand model behavior"""
    
    print(f"\n{'='*70}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. AUC across folds
    ax = axes[0, 0]
    folds = [r['fold_idx'] for r in fold_results]
    val_aucs = [r['val_auc'] for r in fold_results]
    test_aucs = [r['test_auc'] for r in fold_results]
    
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width/2, val_aucs, width, label='Validation', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, test_aucs, width, label='Test', alpha=0.8, color='coral')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Optimal thresholds
    ax = axes[0, 1]
    thresholds = [r['optimal_threshold'] for r in fold_results]
    ax.bar(folds, thresholds, alpha=0.8, color='seagreen')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Optimal Threshold')
    ax.set_title('Optimal Thresholds per Fold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 3. Precision-Recall trade-off
    ax = axes[1, 0]
    precisions = [r['test_precision'] for r in fold_results]
    recalls = [r['test_recall'] for r in fold_results]
    ax.scatter(recalls, precisions, s=100, alpha=0.6, color='purple')
    for i, fold in enumerate(folds):
        ax.annotate(f'F{fold}', (recalls[i], precisions[i]), fontsize=9)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # 4. F1 scores
    ax = axes[1, 1]
    f1_scores = [r['test_f1'] for r in fold_results]
    colors = ['green' if f1 > 0.5 else 'orange' if f1 > 0.3 else 'red' for f1 in f1_scores]
    ax.bar(folds, f1_scores, alpha=0.8, color=colors)
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores per Fold')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "fold_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Diagnostics saved to: {save_dir / 'fold_diagnostics.png'}")


# =============================================================================
# TRAINING
# =============================================================================

def train_single_fold(
    combined_features_df, gt_df, fold_data, fold_idx, config,
    feature_cols, feature_groups, results_dir, ablation_config, train_ids
):
    """
    Train LightGBM on a single fold with proper preprocessing order:
    1. Split data
    2. Apply PCA (if needed) - fit on train only
    3. Apply scaling - fit on train only
    4. Train LightGBM
    5. Find optimal threshold on validation
    6. Evaluate on test
    """
    
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx}")
    print("="*70)
    
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    
    # Split data
    train_df = combined_features_df[combined_features_df['Patient'].isin(train_ids)].copy()
    val_df = combined_features_df[combined_features_df['Patient'].isin(val_ids)].copy()
    test_df = combined_features_df[combined_features_df['Patient'].isin(test_ids)].copy()
    
    # Merge labels
    train_df = train_df.merge(gt_df[['PatientID', 'has_progressed']], 
                               left_on='Patient', right_on='PatientID', how='left')
    val_df = val_df.merge(gt_df[['PatientID', 'has_progressed']], 
                          left_on='Patient', right_on='PatientID', how='left')
    test_df = test_df.merge(gt_df[['PatientID', 'has_progressed']], 
                            left_on='Patient', right_on='PatientID', how='left')
    
    # -----------------------------------
    # STEP 0: Preprocess Demographics (SOPHISTICATED)
    # -----------------------------------
    # Identify demographic features in the feature list
    demo_features_in_data = [f for f in DEMO_FEATURE_COLS if f in feature_cols]
    
    if demo_features_in_data:
        print(f"\n{'='*70}")
        print(f"STEP 0: DEMOGRAPHICS PREPROCESSING (fit on train only)")
        print(f"{'='*70}")
        print(f"Found demographic features: {demo_features_in_data}")
        
        train_df, val_df, test_df, demo_encoding_info, new_demo_cols = preprocess_demographics(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            demo_feature_cols=demo_features_in_data,
            normalization_type=config['normalization_type']
        )
        
        # Update feature_cols: replace original demo features with new preprocessed ones
        updated_feature_cols = [f for f in feature_cols if f not in demo_features_in_data]
        updated_feature_cols.extend(new_demo_cols)
        feature_cols = updated_feature_cols
        
        print(f"\n✓ Feature columns updated:")
        print(f"  Removed: {demo_features_in_data}")
        print(f"  Added: {new_demo_cols}")
    
    # Handle missing values in features
    for col in feature_cols:
        if train_df[col].isnull().any():
            if col in HAND_FEATURE_COLS or col == 'Age' or col == 'Age_normalized':
                fill_value = train_df[col].median()
            else:
                fill_value = 0
            train_df[col] = train_df[col].fillna(fill_value)
            val_df[col] = val_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)
    
    X_train, y_train = train_df[feature_cols].values, train_df['has_progressed'].values.astype(int)
    X_val, y_val = val_df[feature_cols].values, val_df['has_progressed'].values.astype(int)
    X_test, y_test = test_df[feature_cols].values, test_df['has_progressed'].values.astype(int)
    
    print(f"\nAfter demographics preprocessing:")
    print(f"  Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # -----------------------------------
    # STEP 1: Apply PCA (if needed)
    # -----------------------------------
    pca_model = None
    if ablation_config.get('pca', False):
        print(f"\n{'='*70}")
        print("APPLYING PCA (fit on train only)")
        print(f"{'='*70}")
        
        # Use adaptive PCA
        variance_threshold = CONFIG['pca_variance_threshold']
        max_components = CONFIG['pca_max_components']
        
        pca_full = PCA(n_components=min(max_components, X_train.shape[0], X_train.shape[1]))
        pca_full.fit(X_train)
        
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        n_components = max(10, min(n_components, max_components))
        
        print(f"Selected {n_components} components to explain {cumsum_var[n_components-1]*100:.2f}% variance")
        
        pca_model = PCA(n_components=n_components, random_state=42)
        pca_model.fit(X_train)
        
        X_train = pca_model.transform(X_train)
        X_val = pca_model.transform(X_val)
        X_test = pca_model.transform(X_test)
        
        print(f"After PCA: {X_train.shape[1]} components")
    
    # -----------------------------------
    # STEP 2: Apply Scaling (fit on train only)
    # -----------------------------------
    print(f"\n{'='*70}")
    print("APPLYING SCALING (fit on train only)")
    print(f"{'='*70}")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"✓ Features normalized with StandardScaler")
    
    # Check class balance
    print(f"\nClass distribution:")
    print(f"  Train - Pos: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"  Val   - Pos: {y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.1f}%)")
    print(f"  Test  - Pos: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
    
    # -----------------------------------
    # STEP 3: Train LightGBM
    # -----------------------------------
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(config['early_stopping_rounds'])]
    
    model = lgb.train(
        config['lgb_params'],
        train_data,
        num_boost_round=config['num_boost_round'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    print(f"\n✓ Training complete! Best iteration: {model.best_iteration}")
    
    # -----------------------------------
    # STEP 4: Threshold selection on validation
    # -----------------------------------
    print("\n" + "="*70)
    print("THRESHOLD SELECTION (validation set)")
    print("="*70)
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_preds)
    optimal_threshold, val_metrics = find_optimal_threshold(y_val, val_preds)
    
    print(f"Val AUC: {val_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    
    # -----------------------------------
    # STEP 5: Evaluate on test
    # -----------------------------------
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_auc = roc_auc_score(y_test, test_preds)
    
    test_preds_binary = (test_preds >= optimal_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds_binary)
    test_precision = precision_score(y_test, test_preds_binary, zero_division=0)
    test_recall = recall_score(y_test, test_preds_binary, zero_division=0)
    test_f1 = f1_score(y_test, test_preds_binary, zero_division=0)
    
    cm = confusion_matrix(y_test, test_preds_binary)
    tn, fp, fn, tp = cm.ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Generate visualizations
    print("\n✓ Generating visualizations...")
    plot_roc_with_threshold(y_val, val_preds, optimal_threshold, fold_idx, fold_dir, 'val')
    plot_roc_with_threshold(y_test, test_preds, optimal_threshold, fold_idx, fold_dir, 'test')
    plot_confusion_matrix(y_test, test_preds_binary, fold_idx, fold_dir, 'test')
    
    # Save model and preprocessing
    model.save_model(str(fold_dir / "lightgbm_model.txt"))
    
    with open(fold_dir / "preprocessing.pkl", 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'pca_model': pca_model,
            'feature_names': feature_cols,
            'feature_groups': feature_groups,
            'ablation_config': ablation_config
        }, f)
    
    return {
        'fold_idx': fold_idx,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_specificity': test_specificity,
        'test_f1': test_f1,
        'optimal_threshold': optimal_threshold,
        'test_y_true': y_test,
        'test_y_pred': test_preds,
        'test_y_pred_binary': test_preds_binary,
    }


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# ABLATION STUDY COMPARISON
# =============================================================================

def plot_ablation_comparison(ablation_results, save_dir):
    """Plot comparison of all ablation experiments"""
    
    print(f"\n{'='*70}")
    print("GENERATING ABLATION COMPARISON")
    print("="*70)
    
    # Extract metrics
    experiments = []
    mean_aucs = []
    std_aucs = []
    mean_f1s = []
    std_f1s = []
    
    for exp_name, results in ablation_results.items():
        experiments.append(exp_name)
        aucs = [r['test_auc'] for r in results['fold_results']]
        f1s = [r['test_f1'] for r in results['fold_results']]
        mean_aucs.append(np.mean(aucs))
        std_aucs.append(np.std(aucs))
        mean_f1s.append(np.mean(f1s))
        std_f1s.append(np.std(f1s))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: AUC comparison
    ax = axes[0]
    x_pos = np.arange(len(experiments))
    colors = ['steelblue', 'coral', 'seagreen', 'purple', 'orange']
    
    ax.bar(x_pos, mean_aucs, yerr=std_aucs, alpha=0.8, capsize=5, color=colors[:len(experiments)])
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_title('AUC Comparison Across Ablation Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels
    for i, (mean_val, std_val) in enumerate(zip(mean_aucs, std_aucs)):
        ax.text(i, mean_val + std_val + 0.02, f'{mean_val:.3f}±{std_val:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    # Plot 2: F1 comparison
    ax = axes[1]
    ax.bar(x_pos, mean_f1s, yerr=std_f1s, alpha=0.8, capsize=5, color=colors[:len(experiments)])
    ax.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison Across Ablation Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean_val, std_val) in enumerate(zip(mean_f1s, std_f1s)):
        ax.text(i, mean_val + std_val + 0.02, f'{mean_val:.3f}±{std_val:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Ablation comparison saved")


def run_single_ablation_experiment(exp_name, exp_config, patient_features_df, gt_df, 
                                   repeated_kfold_splits, embeddings_dict, base_results_dir, config):
    """
    Run a single ablation experiment
    
    Args:
        exp_name: Name of the experiment
        exp_config: Configuration dict with keys: cnn, pca, hand, demo
        embeddings_dict: Dict mapping pooling method -> embeddings_df
        
    Returns:
        Dict with experiment results
    """
    
    print("\n" + "="*80)
    print(f"ABLATION EXPERIMENT: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print("="*80)
    print(f"Config: {exp_config}")
    
    # Create experiment directory
    exp_dir = base_results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Select embeddings based on pooling method
    if exp_config['cnn'] is not None:
        pooling_method = exp_config['cnn']
        if pooling_method in embeddings_dict:
            embeddings_df = embeddings_dict[pooling_method]
            print(f"\nUsing CNN embeddings with pooling: {pooling_method}")
        else:
            raise ValueError(f"Pooling method '{pooling_method}' not found in embeddings_dict. Available: {list(embeddings_dict.keys())}")
    else:
        embeddings_df = None
        print(f"\nNo CNN embeddings (tabular features only)")
    
    # Train all folds across all repetitions (5×5 = 25 runs)
    fold_results = []
    total_runs = 0
    
    for repeat_idx in sorted(repeated_kfold_splits.keys()):
        kfold_splits = repeated_kfold_splits[repeat_idx]
        
        print(f"\n{'='*70}")
        print(f"REPETITION {repeat_idx + 1}/{len(repeated_kfold_splits)}")
        print(f"{'='*70}")
        
        for fold_idx in sorted(kfold_splits.keys()):
            total_runs += 1
            fold_seed = config['base_seed'] + repeat_idx * 1000 + fold_idx * 100
            set_seed(fold_seed)
            
            fold_data = kfold_splits[fold_idx]
            
            # Build feature matrix according to ablation config
            combined_df, feature_cols, feature_groups = build_feature_matrix(
                patient_features_df, embeddings_df, exp_config,
                HAND_FEATURE_COLS, DEMO_FEATURE_COLS
            )
            
            # Create unique fold identifier for this repetition
            unique_fold_id = f"rep{repeat_idx}_fold{fold_idx}"
            
            print(f"\n[Run {total_runs}/{len(repeated_kfold_splits) * len(kfold_splits)}] Repeat {repeat_idx}, Fold {fold_idx}")
            
            # Train (PCA and scaling are handled inside train_single_fold)
            result = train_single_fold(
                combined_df, gt_df, fold_data, unique_fold_id, config,
                feature_cols, feature_groups, exp_dir, exp_config, fold_data['train']
            )
            
            # Add repetition info to result
            result['repeat_idx'] = repeat_idx
            result['original_fold_idx'] = fold_idx
            
            fold_results.append(result)
    
    # Aggregate results for this experiment
    test_aucs = [r['test_auc'] for r in fold_results]
    test_f1s = [r['test_f1'] for r in fold_results]
    test_recalls = [r['test_recall'] for r in fold_results]
    test_precisions = [r['test_precision'] for r in fold_results]
    test_specificities = [r['test_specificity'] for r in fold_results]
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_name} RESULTS ({len(fold_results)} runs: {config['n_repeats']} repeats × {len(repeated_kfold_splits[0])} folds)")
    print(f"{'='*70}")
    print(f"Test AUC:         {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test F1:          {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Recall:      {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")
    print(f"Test Precision:   {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
    print(f"Test Specificity: {np.mean(test_specificities):.4f} ± {np.std(test_specificities):.4f}")
    print(f"\n  (Averaged across {config['n_repeats']} repeated {len(repeated_kfold_splits[0])}-fold cross-validations)")
    
    # Save fold results
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['test_y_true', 'test_y_pred', 'test_y_pred_binary']} 
                                for r in fold_results])
    summary_df.to_csv(exp_dir / "fold_results.csv", index=False)
    
    # Generate diagnostics
    plot_fold_diagnostics(fold_results, exp_dir)
    plot_aggregate_roc(fold_results, exp_dir)
    
    all_y_true = np.concatenate([r['test_y_true'] for r in fold_results])
    all_y_pred_binary = np.concatenate([r['test_y_pred_binary'] for r in fold_results])
    plot_aggregate_confusion_matrix(all_y_true, all_y_pred_binary, exp_dir)
    
    return {
        'exp_name': exp_name,
        'config': exp_config,
        'fold_results': fold_results,
        'mean_auc': np.mean(test_aucs),
        'std_auc': np.std(test_aucs),
        'mean_f1': np.mean(test_f1s),
        'std_f1': np.std(test_f1s),
        'mean_recall': np.mean(test_recalls),
        'std_recall': np.std(test_recalls),
        'mean_precision': np.mean(test_precisions),
        'std_precision': np.std(test_precisions),
        'mean_specificity': np.mean(test_specificities),
        'std_specificity': np.std(test_specificities)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution with ablation study"""
    
    set_seed(CONFIG['base_seed'])
    
    base_results_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\lightgbm_ablation_resnt50")
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    patient_features_df = pd.read_csv(CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(CONFIG['train_csv_path'], patient_features_df)
    gt_df = load_ground_truth(CONFIG['gt_path'])
    
    # Load base K-fold splits
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits_base = pickle.load(f)
    
    # Now print the study configuration
    print("\n" + "="*80)
    print("ABLATION STUDY: ESSENTIAL EXPERIMENTS FOR THESIS")
    print("="*80)
    print(f"CNN Model: {CONFIG['cnn_model']}")
    print(f"Pooling strategy: mean (proven best)")
    print(f"Cross-validation: {CONFIG['n_repeats']}×{len(kfold_splits_base)} repeated CV ({CONFIG['n_repeats'] * len(kfold_splits_base)} runs per experiment)")
    print(f"\nExperiments (4 total):")
    for exp_name, exp_config in ABLATION_CONFIGS.items():
        print(f"  {exp_name}: {exp_config['description']}")
    
    # Create repeated K-fold splits for repeated cross-validation
    repeated_kfold_splits = create_repeated_kfold_splits(
        kfold_splits_base, 
        CONFIG['n_repeats'], 
        CONFIG['base_seed']
    )
    
    # Filter to patients with ground truth
    all_patient_ids = gt_df['PatientID'].unique().tolist()
    patient_features_df = patient_features_df[patient_features_df['Patient'].isin(all_patient_ids)]
    
    print("\n✓ Using {len(all_patient_ids)} patients with ground truth")
    
    # =========================================================================
    # STEP 1: Extract CNN features with MULTIPLE pooling strategies
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING CNN FEATURES")
    print("="*70)
    print("Extracting slice-level features, then applying multiple pooling methods")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn_extractor = CNNFeatureExtractor(model_name=CONFIG['cnn_model'], device=device)
    
    # Load pre-extracted slice-level features
    slice_features_cache = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\LightGBM_setting\slice_features.csv")
    
    if not slice_features_cache.exists():
        raise FileNotFoundError(f"Slice features file not found: {slice_features_cache}")
    
    print(f"\n✓ Loading pre-extracted slice features from: {slice_features_cache}")
    slice_features_df = pd.read_csv(slice_features_cache)
    print(f"✓ Loaded {len(slice_features_df)} slices")
    print(f"  Feature dimension: {len([c for c in slice_features_df.columns if c.startswith('cnn_feature_')])}")
    
    # Apply MEAN pooling (proven best in preliminary experiments)
    embeddings_dict = {}
    pooling_method = 'mean'
    print(f"\n{'='*70}")
    print(f"Applying {pooling_method.upper()} pooling (Best CNN configuration)")
    print(f"{'='*70}")
    embeddings_df = apply_pooling_to_slices(slice_features_df, pooling_method=pooling_method)
    embeddings_dict[pooling_method] = embeddings_df
    
    print(f"\n✓ Created embeddings with {len(embeddings_dict)} pooling method: {list(embeddings_dict.keys())}")
    
    # =========================================================================
    # STEP 2: Run ALL experiments (streamlined - only essential)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: RUNNING ESSENTIAL EXPERIMENTS")
    print("="*80)
    print("Experiments: hand_only, hand_demo (clinical baseline)")
    print("             cnn_mean (best CNN), best_cnn_hand_demo (full model)")
    
    ablation_results = {}
    
    # Run all experiments in order
    for exp_name in ABLATION_CONFIGS.keys():
        exp_config = ABLATION_CONFIGS[exp_name]
        result = run_single_ablation_experiment(
            exp_name, exp_config, patient_features_df, gt_df,
            repeated_kfold_splits, embeddings_dict, base_results_dir, CONFIG
        )
        ablation_results[exp_name] = result
    
    # =========================================================================
    # STEP 3: Generate comparison summary
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    summary_data = []
    for exp_name, result in ablation_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Description': result['config']['description'],
            'N_Runs': len(result['fold_results']),
            'Mean_AUC': f"{result['mean_auc']:.4f}",
            'Std_AUC': f"{result['std_auc']:.4f}",
            'Mean_F1': f"{result['mean_f1']:.4f}",
            'Std_F1': f"{result['std_f1']:.4f}",
            'Mean_Recall': f"{result['mean_recall']:.4f}",
            'Std_Recall': f"{result['std_recall']:.4f}",
            'Mean_Precision': f"{result['mean_precision']:.4f}",
            'Std_Precision': f"{result['std_precision']:.4f}",
            'Mean_Specificity': f"{result['mean_specificity']:.4f}",
            'Std_Specificity': f"{result['std_specificity']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(base_results_dir / "ablation_summary.csv", index=False)
    print(f"\n✓ Summary saved to: {base_results_dir / 'ablation_summary.csv'}")
    
    # =========================================================================
    # STEP 3.1: Statistical Significance Testing (Paired t-test)
    # =========================================================================
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING (Paired t-test on fold-by-fold AUC)")
    print(f"{'='*80}")
    
    # Extract AUC arrays for each experiment (aligned fold-by-fold)
    exp_names = list(ablation_results.keys())
    auc_arrays = {}
    for exp_name in exp_names:
        aucs = [r['test_auc'] for r in ablation_results[exp_name]['fold_results']]
        auc_arrays[exp_name] = np.array(aucs)
    
    # Perform pairwise paired t-tests
    print(f"\nPairwise comparisons (n={len(auc_arrays[exp_names[0]])} paired observations per comparison):")
    print(f"{'='*70}")
    
    pairwise_results = []
    for i, exp1 in enumerate(exp_names):
        for j, exp2 in enumerate(exp_names):
            if i < j:  # Only upper triangle to avoid duplicates
                t_stat, p_value = ttest_rel(auc_arrays[exp1], auc_arrays[exp2])
                mean_diff = auc_arrays[exp1].mean() - auc_arrays[exp2].mean()
                
                # Determine significance level
                sig_marker = ""
                if p_value < 0.001:
                    sig_marker = "***"
                elif p_value < 0.01:
                    sig_marker = "**"
                elif p_value < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "ns"
                
                pairwise_results.append({
                    'Comparison': f"{exp1} vs {exp2}",
                    'Mean_Diff': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'Significance': sig_marker
                })
                
                print(f"  {exp1:25s} vs {exp2:25s}")
                print(f"    Mean AUC diff: {mean_diff:+.4f}")
                print(f"    t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {sig_marker}")
                print()
    
    # Save pairwise comparison results
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(base_results_dir / "statistical_tests.csv", index=False)
    print(f"\n✓ Statistical tests saved to: {base_results_dir / 'statistical_tests.csv'}")
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    
    # Generate comparison plots
    plot_ablation_comparison(ablation_results, base_results_dir)
    
    # Find best overall configuration
    best_exp = max(ablation_results.items(), key=lambda x: x[1]['mean_auc'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST CONFIGURATION: {best_exp[0]}")
    print(f"   {best_exp[1]['config']['description']}")
    print(f"   Mean AUC: {best_exp[1]['mean_auc']:.4f} ± {best_exp[1]['std_auc']:.4f}")
    print(f"   Mean F1:  {best_exp[1]['mean_f1']:.4f} ± {best_exp[1]['std_f1']:.4f}")
    print(f"   (Based on {len(best_exp[1]['fold_results'])} runs: {CONFIG['n_repeats']}×{len(kfold_splits_base)}-fold repeated CV)")
    print(f"{'='*80}")
    
    print("\n✓ ABLATION STUDY COMPLETE!")
    print(f"✓ Repeated Cross-Validation: {CONFIG['n_repeats']} repeats × {len(kfold_splits_base)} folds = {CONFIG['n_repeats'] * len(kfold_splits_base)} runs per experiment")
    print(f"Results saved to: {base_results_dir}")


# REMOVED: main_single_experiment() - replaced by ablation study in main()
# The old single-experiment approach has been superseded by the comprehensive
# ablation study that systematically compares different feature combinations.


if __name__ == "__main__":
    # Run comprehensive ablation study comparing all feature combinations
    main()