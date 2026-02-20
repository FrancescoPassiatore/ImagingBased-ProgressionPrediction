# Ablation Study con Normalizzazione Corretta
# IMPORTANTE: Ogni fold normalizza usando SOLO il proprio training set

from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import random
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    CNNFeatureExtractor,
    IPFDataLoader,create_dataloaders,compute_class_weights
    )
from model_train import (
    ProgressionPredictionModel,
    ModelTrainer,
    HAND_FEATURE_COLS,
    DEMO_FEATURE_COLS,
    plot_evaluation_metrics,
    evaluate_with_threshold,
    plot_validation_roc_with_thresholds,
    aggregate_fold_results,
    train_single_fold
    )


# ============================================================================
# ABLATION STUDY STRUCTURE - 3 BLOCKS
# ============================================================================

ABLATION_CONFIGS = {
    # ========== BLOCK 1: CLINICAL BASELINE ==========
    # Traditional clinical prediction without imaging
    
    '1_hand_only': {
        'use_cnn_features': False,
        'use_hand_features': True,
        'use_demographics': False,
        'description': '[BLOCK 1] Hand-crafted features only',
        'block': 'Clinical Baseline'
    },
    
    '1_hand_demo': {
        'use_cnn_features': False,
        'use_hand_features': True,
        'use_demographics': True,
        'description': '[BLOCK 1] Hand-crafted + Demographics',
        'block': 'Clinical Baseline'
    },
    
    # ========== BLOCK 2: IMAGING ONLY ==========
    # Deep learning on small dataset - show instability
    
    '2_cnn_mean': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': '[BLOCK 2] CNN with mean pooling',
        'block': 'Imaging Only',
        'pooling_type': 'mean'
    },
    
    '2_cnn_max': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': '[BLOCK 2] CNN with max pooling',
        'block': 'Imaging Only',
        'pooling_type': 'max'
    },
    
    '2_cnn_max_mean': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': '[BLOCK 2] CNN with max+mean pooling (concatenated)',
        'block': 'Imaging Only',
        'pooling_type': 'max_mean'
    },
    
    # ========== BLOCK 3: FULL MODEL ==========
    # Multimodal fusion - imaging + clinical
    # Uses pooling_type from BASE_CONFIG (set manually after BLOCK 2 results)
    
    '3_cnn_hand': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': False,
        'description': '[BLOCK 3] CNN + Hand-crafted',
        'block': 'Full Model'
        # No pooling_type override - uses BASE_CONFIG['pooling_type']
    },
    
    '3_cnn_hand_demo': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': True,
        'description': '[BLOCK 3] CNN + Hand-crafted + Demographics (FULL)',
        'block': 'Full Model'
        # No pooling_type override - uses BASE_CONFIG['pooling_type']
    },
}


BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    'base_seed': 42,
    # Model parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'pooling_type': 'max',
    'use_feature_branches': False,  # Simplified architecture for small dataset
    'use_ktop': False,
    
    # Training parameters
    'batch_size': 16,
    'learning_rate': 3.86e-05,
    'weight_decay': 0.003, #0.003
    'epochs': 60,
    'early_stopping_patience': 15,
    'label_smoothing': 0,
    'use_scheduler': True,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    
    # Model architecture (simplified: CNN reduction → concat → small classifier)
    'hidden_dims': [32],  # Not used anymore - architecture is now fixed
    'dropout': 0.3,
    'use_batch_norm': False,
    
    'resume_from_checkpoint': True,
    'normalization_type': 'standard',  # 'standard', 'minmax', o 'robust'
}


# Hand-crafted features da normalizzare
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

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*70}")
    print(f"🔒 RANDOM SEED SET TO: {seed}")
    print(f"{'='*70}")

def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Carica demographics da train.csv e merge con patient_features
    """
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS FROM TRAIN.CSV")
    print("="*70)
    print(f"Train CSV shape: {train_df.shape}")
    print(f"Columns available: {train_df.columns.tolist()}")
    
    # Identifica colonne demographics disponibili
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  Found: {col}")
        else:
            print(f"  ⚠️  Missing: {col}")
    
    # Merge
    demographics_df = train_df[demo_cols].copy()
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    print(f"\nEnhanced features shape: {enhanced_df.shape}")
    
    # Encode categorical variables to numeric
    if 'Sex' in enhanced_df.columns:
        print(f"\nEncoding Sex (categorical -> numeric)")
        print(f"  Unique values: {enhanced_df['Sex'].unique()}")
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"  Encoded as: Male=1, Female=0")
    
    if 'SmokingStatus' in enhanced_df.columns:
        print(f"\nEncoding SmokingStatus (categorical -> numeric)")
        print(f"  Unique values: {enhanced_df['SmokingStatus'].unique()}")
        # Codifica: Never smoked=0, Ex-smoker=1, Currently smokes=2
        smoking_map = {
            'Never smoked': 0,
            'Ex-smoker': 1,
            'Currently smokes': 2
        }
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(smoking_map)
        print(f"  Encoded as: Never smoked=0, Ex-smoker=1, Currently smokes=2")
    
    # Check missing values
    missing = enhanced_df[demo_cols[1:]].isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing demographic values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} missing")
    
    return enhanced_df


def preprocess_demographics_improved(
    result_df: pd.DataFrame,
    train_patient_ids: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Improved demographics preprocessing with proper centering and encoding
    
    Returns:
        result_df: DataFrame with preprocessed demographic features
        encoding_info: Dict containing preprocessing metadata
    """
    encoding_info = {}
    
    # Get training set for fitting
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)]
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    
    # === 1. AGE (Continuous) ===
    if 'Age' in result_df.columns:
        print("\n=== PREPROCESSING AGE ===")
        
        print(f"  Pre-normalization:")
        print(f"    Mean: {train_patient_df['Age'].mean():.2f}")
        print(f"    Std: {train_patient_df['Age'].std():.2f}")
        print(f"    Range: [{train_patient_df['Age'].min():.0f}, {train_patient_df['Age'].max():.0f}]")
        
        # Normalize Age
        if normalization_type == 'standard':
            age_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            age_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            age_scaler = MinMaxScaler(feature_range=(-1, 1))  # Center around 0
        
        age_scaler.fit(train_patient_df[['Age']].values)
        result_df['Age_normalized'] = age_scaler.transform(result_df[['Age']].values)
        
        # Verify
        train_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)]
        train_patient_normalized = train_normalized.groupby('patient_id').first().reset_index()
        print(f"  Post-normalization:")
        print(f"    Mean: {train_patient_normalized['Age_normalized'].mean():.4f}")
        print(f"    Std: {train_patient_normalized['Age_normalized'].std():.4f}")
        
        encoding_info['age_scaler'] = age_scaler
    else:
        print("  ⚠️  Age column not found")
    
    # === 2. SEX (Binary Categorical) ===
    if 'Sex' in result_df.columns:
        print("\n=== PREPROCESSING SEX ===")
        print(f"  Unique values: {result_df['Sex'].unique()}")
        
        # Convert to binary centered around 0 (better for neural networks)
        result_df['Sex_encoded'] = result_df['Sex'].map({0: -1, 1: 1})  # Female=-1, Male=1
        
        print(f"  Encoded as: Female=-1, Male=1 (centered)")
        sex_dist = result_df.groupby('patient_id')['Sex_encoded'].first().value_counts()
        print(f"  Distribution:")
        for val, count in sex_dist.items():
            label = 'Female' if val == -1 else 'Male'
            print(f"    {label}: {count}")
        
        encoding_info['sex_encoding'] = {0: -1, 1: 1, 'description': 'Female=-1, Male=1'}
    
    # === 3. SMOKING STATUS (Multi-class Categorical) ===
    if 'SmokingStatus' in result_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS ===")
        print(f"  Unique values: {result_df['SmokingStatus'].unique()}")
        
        # One-hot encoding (more expressive for neural networks)
        smoking_dummies = pd.get_dummies(
            result_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        
        # Center binary features around 0 (from [0,1] to [-0.5, 0.5])
        # This helps neural network learning
        smoking_dummies = (smoking_dummies - 0.5)
        
        # Add to result_df
        for col in smoking_dummies.columns:
            result_df[col] = smoking_dummies[col]
        
        smoking_cols = sorted(smoking_dummies.columns.tolist())
        encoding_info['smoking_columns'] = smoking_cols
        
        print(f"  One-hot encoded into {len(smoking_cols)} features: {smoking_cols}")
        print(f"  Values centered to [-0.5, 0.5]")
        
        # Show distribution
        print(f"  Training set distribution:")
        orig_smoking = result_df[result_df['patient_id'].isin(train_patient_ids)].groupby('patient_id')['SmokingStatus'].first()
        for val in orig_smoking.unique():
            count = (orig_smoking == val).sum()
            print(f"    {val}: {count}")
    
    # === REMOVE ORIGINAL DEMOGRAPHIC COLUMNS ===
    # Keep only the preprocessed versions to avoid duplication
    cols_to_remove = []
    if 'Age_normalized' in result_df.columns and 'Age' in result_df.columns:
        cols_to_remove.append('Age')
    if 'Sex_encoded' in result_df.columns and 'Sex' in result_df.columns:
        cols_to_remove.append('Sex')
    if encoding_info.get('smoking_columns') and 'SmokingStatus' in result_df.columns:
        cols_to_remove.append('SmokingStatus')
    
    if cols_to_remove:
        result_df.drop(cols_to_remove, axis=1, inplace=True)
        print(f"\n  Removed original demographic columns: {cols_to_remove}")
        print(f"  ✓ Using only preprocessed versions")
    
    return result_df, encoding_info


def get_preprocessed_demo_features(
    row: pd.Series,
    encoding_info: dict
) -> np.ndarray:
    """
    Extract preprocessed demographic features for a single patient
    Used by the dataset to extract demo features from a row
    
    Returns:
        features: numpy array of shape (n_demo_features,)
    """
    features = []
    
    # Age (normalized)
    if 'Age_normalized' in row:
        features.append(row['Age_normalized'])
    
    # Sex (binary, centered)
    if 'Sex_encoded' in row:
        features.append(row['Sex_encoded'])
    
    # Smoking (one-hot, centered)
    smoking_cols = encoding_info.get('smoking_columns', [])
    for col in smoking_cols:
        if col in row:
            features.append(row[col])
    
    return np.array(features, dtype=np.float32)


def normalize_features_per_fold(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    hand_feature_cols: list,
    demo_feature_cols: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Normalizza features usando SOLO statistiche dal training set
    Uses improved demographics preprocessing with proper centering
    Returns the normalized dataframe AND the scalers/encoding info for inverse transform
    
    CRUCIALE: Fit su train, transform su tutto il fold (train+val+test)
    """
    result_df = features_df.copy()
    scalers = {}
    
    # Identifica features disponibili
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
    
    if not available_hand and not available_demo:
        return result_df, scalers  # Nessuna feature da normalizzare
    
    print(f"\nNormalizing features ({normalization_type} scaler):")
    
    # Prendi statistiche SOLO dal training set (un sample per paziente)
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    print(f"  Training set: {len(train_patient_df)} patients (not slices)")
    # === HAND-CRAFTED FEATURES ===
    if available_hand:
        print(f"  Hand-crafted: {len(available_hand)} features")
        
        # Statistiche pre-normalizzazione (train only)
        print(f"\n  Pre-normalization (Training Set):")
        for col in available_hand:
            mean = train_patient_df[col].mean()
            std = train_patient_df[col].std()
            print(f"    {col:30s}: mean={mean:10.4f}, std={std:10.4f}")
        
        # Crea scaler e fit su training
        if normalization_type == 'standard':
            hand_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            hand_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            hand_scaler = MinMaxScaler()
        
        hand_scaler.fit(train_patient_df[available_hand].values)
        
        # Transform tutto il dataset
        result_df[available_hand] = hand_scaler.transform(result_df[available_hand].values)
        
        # Verifica post-normalizzazione (train only)
        train_df_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
        train_patient_normalized = train_df_normalized.groupby('patient_id').first().reset_index()
        
        print(f"\n  Post-normalization (Training Set):")
        for col in available_hand:
            mean = train_patient_normalized[col].mean()
            std = train_patient_normalized[col].std()
            print(f"    {col:30s}: mean={mean:10.4f}, std={std:10.4f}")
            
            # Verifica che sia corretto (mean~0, std~1)
            if abs(mean) > 0.01:
                print(f"      ⚠️ WARNING: Mean not close to 0!")
            if abs(std - 1.0) > 0.01:
                print(f"      ⚠️ WARNING: Std not close to 1!")
        
        # Store scaler in scalers dict
        scalers['hand_scaler'] = hand_scaler
    
    # === DEMOGRAPHIC FEATURES (IMPROVED PREPROCESSING) ===
    if available_demo:
        print(f"\n=== DEMOGRAPHIC FEATURES (IMPROVED) ===")
        print(f"  Found {len(available_demo)} demographic columns: {available_demo}")
        
        # Apply improved preprocessing
        result_df, encoding_info = preprocess_demographics_improved(
            result_df,
            train_patient_ids,
            normalization_type
        )
        
        # Store encoding info in scalers dict
        scalers['demo_encoding'] = encoding_info
        
        # DEBUG: Verify encoding_info after preprocessing
        print(f"  [DEBUG] encoding_info keys after preprocessing: {list(encoding_info.keys())}")
        if 'smoking_columns' in encoding_info:
            print(f"  [DEBUG] Smoking columns stored: {encoding_info['smoking_columns']}")
        
        print(f"  ✓ Demographics preprocessed with proper centering and encoding")
    
    return result_df, scalers


def create_feature_set_for_fold(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    ablation_config: dict,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Crea feature set per un fold specifico con normalizzazione corretta
    Returns: (features_df, encoding_info)
    """
    print("\n" + "="*70)
    print(f"FEATURE SET: {ablation_config['description']}")
    print("="*70)
    
    # Start con slice-level info (patient_id, slice_idx, label)
    result_df = slice_features_df.copy()
    
    # Rimuovi CNN features se non richieste (CRUCIAL per BLOCK 1)
    if not ablation_config['use_cnn_features']:
        cnn_cols_to_drop = [c for c in result_df.columns if c.startswith('cnn_feature_')]
        if cnn_cols_to_drop:
            result_df.drop(cnn_cols_to_drop, axis=1, inplace=True)
            print(f"  Removed {len(cnn_cols_to_drop)} CNN features (use_cnn_features=False)")
    
    # Determina quali features aggiungere
    patient_level_cols = ['Patient']
    hand_to_add = []
    demo_to_add = []
    
    if ablation_config['use_hand_features']:
        available_hand = [f for f in HAND_FEATURE_COLS if f in patient_features_df.columns]
        patient_level_cols.extend(available_hand)
        hand_to_add = available_hand
        print(f"  Adding {len(available_hand)} hand-crafted features")
    
    if ablation_config['use_demographics']:
        available_demo = [f for f in DEMO_FEATURE_COLS if f in patient_features_df.columns]
        patient_level_cols.extend(available_demo)
        demo_to_add = available_demo
        print(f"  Adding {len(available_demo)} demographic features")
    
    # Initialize encoding_info
    encoding_info = {}
    
    # Merge patient-level features
    if len(patient_level_cols) > 1:
        result_df = result_df.merge(
            patient_features_df[patient_level_cols],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        result_df.drop('Patient', axis=1, inplace=True)
        
        # Handle missing values PRIMA della normalizzazione
        all_features_to_check = hand_to_add + demo_to_add
        missing = result_df[all_features_to_check].isnull().sum()
        
        if missing.any():
            print(f"\n  ⚠️ Handling missing values:")
            
            # Calcola statistiche dal training set
            train_stats = result_df[result_df['patient_id'].isin(fold_data['train'])].groupby('patient_id').first()
            
            for col in all_features_to_check:
                if result_df[col].isnull().any():
                    if col == 'Age' or col in hand_to_add:
                        # Per continue: usa mediana del training
                        fill_value = train_stats[col].median()
                        result_df[col].fillna(fill_value, inplace=True)
                        print(f"    {col}: filled {missing[col]} values with train median ({fill_value:.2f})")
                    else:
                        # Per categoriche: usa 0 (unknown)
                        result_df[col].fillna(0, inplace=True)
                        print(f"    {col}: filled {missing[col]} values with 0 (unknown)")
        
        # NORMALIZZAZIONE (usando solo training set)
        if hand_to_add or demo_to_add:
            result_df, scalers = normalize_features_per_fold(
                features_df=result_df,
                train_patient_ids=fold_data['train'],
                hand_feature_cols=hand_to_add,
                demo_feature_cols=demo_to_add,
                normalization_type=normalization_type
            )
            
            # Extract encoding info
            encoding_info = scalers.get('demo_encoding', {})
            
            # DEBUG: Verify encoding_info extraction
            if demo_to_add:
                print(f"\n[DEBUG] Extracted encoding_info keys: {list(encoding_info.keys())}")
                if 'smoking_columns' in encoding_info:
                    print(f"[DEBUG] Smoking columns in encoding_info: {encoding_info['smoking_columns']}")
    
    # Feature dimension summary
    cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
    
    # Calculate actual demographic features (may be expanded by one-hot encoding)
    actual_demo_features = 0
    if demo_to_add:
        if 'Age' in demo_to_add:
            actual_demo_features += 1  # Age_normalized
        if 'Sex' in demo_to_add:
            actual_demo_features += 1  # Sex_encoded
        if 'SmokingStatus' in demo_to_add:
            smoking_cols = encoding_info.get('smoking_columns', [])
            actual_demo_features += len(smoking_cols)  # Smoking one-hot (3 features)
    
    print(f"\nFinal feature composition:")
    print(f"  CNN features: {len(cnn_cols)}")
    print(f"  Hand-crafted features: {len(hand_to_add)}")
    print(f"  Demographic features: {actual_demo_features} (from {len(demo_to_add)} original columns)")
    print(f"  Total: {len(cnn_cols) + len(hand_to_add) + actual_demo_features}")
    
    return result_df, encoding_info


def run_ablation_study(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
):
    """Run ablation study with 3 blocks: clinical baseline + imaging pooling + multimodal"""
    
    print("\n" + "="*70)
    print("ABLATION STUDY - 3 BLOCKS")
    print("="*70)
    print(f"Total configurations: {len(ABLATION_CONFIGS)}")
    
    all_ablation_results = {}
    
    # Separate experiments by block
    block_1_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k.startswith('1_')}
    block_2_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k.startswith('2_')}
    block_3_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k.startswith('3_')}
    
    print(f"\n  BLOCK 1 (Clinical):  {len(block_1_configs)} experiments")
    print(f"  BLOCK 2 (Imaging):   {len(block_2_configs)} experiments")
    print(f"  BLOCK 3 (Full):      {len(block_3_configs)} experiments")
    print(f"\n  BLOCK 3 will use pooling type: '{base_config['pooling_type']}'")
    print(f"  (Update BASE_CONFIG['pooling_type'] after BLOCK 2 if needed)")
    """
    # ========== BLOCK 1: Clinical Baseline ==========
    print("\n" + "="*80)
    print("EXECUTING BLOCK 1 - CLINICAL BASELINE")
    print("="*80)
    
    for config_name, ablation_config in block_1_configs.items():
        all_ablation_results[config_name] = _run_single_experiment(
            config_name, ablation_config, 
            slice_features_df, patient_features_df, kfold_splits, base_config, results_base_dir
        )
    
    # ========== BLOCK 2: Imaging Only (Test Pooling Strategies) ==========
    print("\n" + "="*80)
    print("EXECUTING BLOCK 2 - IMAGING ONLY (POOLING COMPARISON)")
    print("="*80)
    
    for config_name, ablation_config in block_2_configs.items():
        all_ablation_results[config_name] = _run_single_experiment(
            config_name, ablation_config,
            slice_features_df, patient_features_df, kfold_splits, base_config, results_base_dir
        )
    
    # ========== Analyze BLOCK 2 Results ==========
    print("\n" + "="*80)
    print("BLOCK 2 POOLING COMPARISON RESULTS")
    print("="*80)
    
    block_2_results = {}
    for config_name in block_2_configs.keys():
        fold_results = all_ablation_results[config_name]['fold_results']
        mean_auc = np.mean([fold['test_metrics_optimal']['auc'] for fold in fold_results])
        std_auc = np.std([fold['test_metrics_optimal']['auc'] for fold in fold_results])
        pooling_type = ABLATION_CONFIGS[config_name]['pooling_type']
        
        block_2_results[pooling_type] = {
            'config_name': config_name,
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
        print(f"  {pooling_type:12s}: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Identify best pooling for user's reference
    best_pooling = max(block_2_results.items(), key=lambda x: x[1]['mean_auc'])[0]
    best_auc = block_2_results[best_pooling]['mean_auc']
    
    print(f"\n→ Best performing pooling: {best_pooling} (AUC = {best_auc:.4f})")
    print(f"  Recommended for BLOCK 3 multimodal experiments")
    
    # Save pooling comparison for reference
    pooling_info = {
        'best_pooling': best_pooling,
        'best_auc': best_auc,
        'block_2_results': block_2_results
    }
    import json
    with open(results_base_dir / "pooling_comparison.json", 'w') as f:
        json.dump(pooling_info, f, indent=2, default=str)
    
    print(f"\n✓ Pooling comparison saved to: {results_base_dir / 'pooling_comparison.json'}")
    """
    # ========== BLOCK 3: Full Model (Multimodal) ==========
    print("\n" + "="*80)
    print(f"EXECUTING BLOCK 3 - FULL MODEL (using '{base_config['pooling_type']}' pooling)")
    print("="*80)
    
    for config_name, ablation_config in block_3_configs.items():
        all_ablation_results[config_name] = _run_single_experiment(
            config_name, ablation_config,
            slice_features_df, patient_features_df, kfold_splits, base_config, results_base_dir
        )
    
    # Create comparison
    create_ablation_comparison(all_ablation_results, results_base_dir)
    
    # Statistical testing (fold-by-fold paired t-tests)
    perform_statistical_testing(all_ablation_results, results_base_dir)
    
    return all_ablation_results


def _run_single_experiment(
    config_name: str,
    ablation_config: dict,
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
) -> dict:
    """Run a single ablation experiment"""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {config_name.upper()}")
    print("="*80)
    print(f"Description: {ablation_config['description']}")
    
    ablation_results_dir = results_base_dir / f"ablation_{config_name}"
    ablation_results_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    fold_keys = sorted(kfold_splits.keys())
    
    for fold_idx in fold_keys:
        fold_seed = base_config['base_seed'] + fold_idx * 100
        set_seed(fold_seed)
        
        print(f"\nFOLD {fold_idx} - SEED: {fold_seed}")
        fold_data = kfold_splits[fold_idx]
        
        # CRUCIAL: Normalize per fold using only its training set
        features_df, encoding_info = create_feature_set_for_fold(
            slice_features_df=slice_features_df,
            patient_features_df=patient_features_df,
            fold_data=fold_data,
            ablation_config=ablation_config,
            normalization_type=base_config['normalization_type']
        )
        
        # Configure
        config = base_config.copy()
        config['results_save_dir'] = ablation_results_dir
        
        # Override pooling_type if specified in ablation_config
        if 'pooling_type' in ablation_config:
            config['pooling_type'] = ablation_config['pooling_type']
            print(f"  Using pooling: {ablation_config['pooling_type']}")
        
        # Train
        result = train_single_fold(
            features_df=features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=config,
            results_dir=ablation_results_dir,
            resume_from_checkpoint=config['resume_from_checkpoint'],
            encoding_info=encoding_info
        )
        
        fold_results.append(result)
    
    summary_df, detailed_df = aggregate_fold_results(
        fold_results=fold_results,
        save_path=ablation_results_dir
    )
    
    print(f"\n✓ '{config_name}' complete!")
    
    return {
        'config': ablation_config,
        'summary': summary_df,
        'detailed': detailed_df,
        'fold_results': fold_results
    }


def create_ablation_comparison(all_results: dict, results_dir: Path):
    """Create comparison summary across ablation experiments"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("ABLATION COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    for config_name, results in all_results.items():
        summary = results['summary']
        
        val_auc = summary[summary['Metric'] == 'Validation AUC']['Mean'].values[0]
        test_auc_optimal = summary[summary['Metric'] == 'Test AUC (Optimal)']['Mean'].values[0]
        test_acc_optimal = summary[summary['Metric'] == 'Test Accuracy (Optimal)']['Mean'].values[0]
        test_f1_optimal = summary[summary['Metric'] == 'Test F1 (Optimal)']['Mean'].values[0]
        
        comparison_data.append({
            'Configuration': config_name,
            'Description': results['config']['description'],
            'Val_AUC': val_auc,
            'Test_AUC_Optimal': test_auc_optimal,
            'Test_Acc_Optimal': test_acc_optimal,
            'Test_F1_Optimal': test_f1_optimal
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_AUC_Optimal', ascending=False)
    comparison_df.to_csv(results_dir / "ablation_comparison.csv", index=False)
    
    print("\nResults (sorted by Test AUC):")
    print(comparison_df.to_string(index=False))
    
    # Best configuration
    best = comparison_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best['Configuration']}")
    print(f"   {best['Description']}")
    print(f"   Test AUC: {best['Test_AUC_Optimal']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    configs = comparison_df['Configuration'].values
    x = range(len(configs))
    
    axes[0].bar(x, comparison_df['Test_AUC_Optimal'], alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Test AUC')
    axes[0].set_title('AUC Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, comparison_df['Test_Acc_Optimal'], alpha=0.7, color='seagreen')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, comparison_df['Test_F1_Optimal'], alpha=0.7, color='coral')
    axes[2].set_ylabel('Test F1-Score')
    axes[2].set_title('F1-Score Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(configs, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved!")

def perform_statistical_testing(all_results: dict, results_dir: Path):
    """
    Perform fold-by-fold paired t-tests between experiments
    Tests hypothesis: Is experiment A significantly better than experiment B?
    """
    print("\n" + "="*70)
    print("STATISTICAL TESTING - PAIRED T-TESTS (FOLD-BY-FOLD)")
    print("="*70)
    
    # Extract fold-by-fold test AUCs for each experiment
    experiment_names = list(all_results.keys())
    experiment_aucs = {}
    
    for exp_name in experiment_names:
        fold_results = all_results[exp_name]['fold_results']
        # Get test AUC with optimal threshold for each fold
        aucs = [fold['test_metrics_optimal']['auc'] for fold in fold_results]
        experiment_aucs[exp_name] = np.array(aucs)
        print(f"  {exp_name}: {len(aucs)} folds, AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # Perform pairwise comparisons
    print("\n" + "="*70)
    print("PAIRWISE COMPARISONS (Paired T-Test)")
    print("="*70)
    
    comparisons = []
    n_experiments = len(experiment_names)
    
    for i in range(n_experiments):
        for j in range(i + 1, n_experiments):
            exp1 = experiment_names[i]
            exp2 = experiment_names[j]
            
            aucs1 = experiment_aucs[exp1]
            aucs2 = experiment_aucs[exp2]
            
            # Paired t-test (requires same number of folds)
            if len(aucs1) != len(aucs2):
                print(f"\n⚠ {exp1} vs {exp2}: Different number of folds, skipping")
                continue
            
            t_stat, p_value = ttest_rel(aucs1, aucs2)
            mean_diff = np.mean(aucs1 - aucs2)
            
            # Determine significance
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            print(f"\n{exp1} vs {exp2}:")
            print(f"  Mean AUC difference: {mean_diff:.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f} {sig}")
            
            comparisons.append({
                'Experiment_1': exp1,
                'Experiment_2': exp2,
                'Mean_AUC_1': np.mean(aucs1),
                'Mean_AUC_2': np.mean(aucs2),
                'Mean_Diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'Significance': sig
            })
    
    # Save to CSV
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(results_dir / "statistical_tests.csv", index=False)
    
    print("\n" + "="*70)
    print("STATISTICAL TESTING COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_dir / 'statistical_tests.csv'}")
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")

def main():
    set_seed(BASE_CONFIG['base_seed'])
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\third_block_analysis")
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path']
    )
    patient_data, _ = data_loader.get_patient_data()
    
    # Extract CNN features
    print("\n" + "="*70)
    print("EXTRACTING CNN FEATURES")
    print("="*70)
    
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=results_base_dir / "slice_features.csv"
    )
    
    # Load patient features and demographics
    print("\n" + "="*70)
    print("LOADING PATIENT FEATURES")
    print("="*70)
    
    patient_features_df = pd.read_csv(BASE_CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=BASE_CONFIG['train_csv_path'],
        patient_features_df=patient_features_df
    )
    
    # Load K-Fold splits
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"\nLoaded {len(kfold_splits)} folds")
    
    # Run ablation study
    all_results = run_ablation_study(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        base_config=BASE_CONFIG,
        results_base_dir=results_base_dir
    )
    
    print("\n" + "="*70)
    print("✓ ABLATION STUDY COMPLETE!")
    print("="*70)
    print(f"Results: {results_base_dir}")


if __name__ == "__main__":
    main()