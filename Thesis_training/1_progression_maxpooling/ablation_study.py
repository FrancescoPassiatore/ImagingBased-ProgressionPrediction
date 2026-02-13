# Ablation Study con Normalizzazione Corretta
# IMPORTANTE: Ogni fold normalizza usando SOLO il proprio training set

from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    aggregate_fold_results
    
    )

"""'cnn_only': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': 'CNN features only (baseline)'
    },
    'cnn_demo': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': True,
        'description': 'CNN + Demographics'
    },"""

# Feature configurations
ABLATION_CONFIGS = {

    
    
    'cnn_hand': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': False,
        'description': 'CNN + Hand-crafted features'
    },
    
    'full': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': True,
        'description': 'CNN + Hand-crafted + Demographics (full)'
    },
}


BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_all_slices.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # Model parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'pooling_type': 'max',
    'use_feature_branches': True,
    'use_ktop': False,
    
    # Training parameters
    'batch_size': 16,
    'learning_rate': 3.86e-05,
    'weight_decay': 0.0305,
    'epochs': 100,
    'early_stopping_patience': 20,
    'label_smoothing': 0.2,
    'use_scheduler': True,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    
    # Model architecture
    'hidden_dims': [256, 128,64],
    'dropout': 0.7,
    'use_batch_norm': True,
    
    'resume_from_checkpoint': True,
    'normalization_type': 'standard',  # 'standard', 'minmax', o 'robust'
}


# Hand-crafted features da normalizzare
HAND_FEATURE_COLS = [
    'ApproxVol',
    'Avg_NumTissuePixel',
    'Avg_Tissue',
    'Avg_Tissue_thickness',
    'Avg_TissueByTotal',
    'Avg_TissueByLung',
    'Mean',
    'Skew',
    'Kurtosis'
]

# Demographic features
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']


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
    
    # Start con CNN features
    result_df = slice_features_df.copy()
    
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


def train_single_fold(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path,
    resume_from_checkpoint: bool = True,
    hand_feature_cols: list = None,
    demo_feature_cols: list = None,
    encoding_info: dict = None
):
    """Train model on a single fold with slice-level predictions"""
    
    if hand_feature_cols is None:
        hand_feature_cols = HAND_FEATURE_COLS
    if demo_feature_cols is None:
        demo_feature_cols = DEMO_FEATURE_COLS
    if encoding_info is None:
        encoding_info = {}
    
    # DEBUG: Verify encoding_info content
    if encoding_info:
        print(f"\n[DEBUG] encoding_info received: {list(encoding_info.keys())}")
        if 'smoking_columns' in encoding_info:
            print(f"[DEBUG] Smoking columns: {encoding_info['smoking_columns']}")
    else:
        print("\n[DEBUG] WARNING: encoding_info is empty!")
    
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = fold_dir / "best_model.pth"
    
    # Check for existing checkpoint
    if resume_from_checkpoint and checkpoint_path.exists():
        print("\n" + "="*70)
        print(f"CHECKPOINT FOUND FOR FOLD {fold_idx}")
        print("="*70)
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        is_complete = (
            'test_metrics_default' in checkpoint and 
            'test_metrics_optimal' in checkpoint and
            'val_auc' in checkpoint and
            'optimal_threshold' in checkpoint
        )
        
        if is_complete:
            print("\n✓ Fold already completed! Loading saved results...")
            print(f"  Val AUC: {checkpoint.get('val_auc'):.4f}")
            print(f"  Test AUC (Optimal): {checkpoint['test_metrics_optimal'].get('auc'):.4f}")
            
            return {
                'fold_idx': fold_idx,
                'val_auc': checkpoint.get('val_auc'),
                'test_metrics_default': checkpoint.get('test_metrics_default', {}),
                'test_metrics_optimal': checkpoint.get('test_metrics_optimal', {}),
                'optimal_threshold': checkpoint.get('optimal_threshold'),
                'loaded_from_checkpoint': True
            }
    
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx}")
    print("="*70)
    
    # Create dataloaders
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    # Identify available features
    available_hand_cols = [c for c in hand_feature_cols if c in features_df.columns]
    
    # For demographics, DON'T filter by column names since they've been preprocessed
    # (Age -> Age_normalized, Sex -> Sex_encoded, SmokingStatus -> Smoking_0/1/2)
    # The dataset will handle mapping original names to preprocessed columns
    
    print(f"\nFeature availability:")
    print(f"  Hand-crafted: {len(available_hand_cols)}/{len(hand_feature_cols)}")
    print(f"  Demographics: {len(demo_feature_cols)}/{len(demo_feature_cols)}")
    
    # Compute class weights
    class_weights = compute_class_weights(features_df, train_ids)
    
    # Create dataloaders
    # Pass ORIGINAL demographic column names - the dataset handles preprocessing mapping
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=available_hand_cols,
        demo_feature_cols=demo_feature_cols,  # Pass original names, not filtered
        encoding_info=encoding_info
    )
    
    # Get actual dimensions from first batch
    sample_batch = next(iter(train_loader))
    actual_cnn_dim = sample_batch['cnn_features'].shape[2]
    
    # Get actual hand-crafted and demographic dimensions
    actual_hand_dim = sample_batch['hand_features'].shape[1] if sample_batch['hand_features'] is not None else 0
    actual_demo_dim = sample_batch['demo_features'].shape[1] if sample_batch['demo_features'] is not None else 0
    actual_patient_dim = actual_hand_dim + actual_demo_dim
    
    print(f"\nActual feature dimensions:")
    print(f"  CNN features: {actual_cnn_dim}")
    print(f"  Hand-crafted features: {actual_hand_dim}")
    print(f"  Demographic features: {actual_demo_dim}")
    print(f"  Total patient-level: {actual_patient_dim}")
    
    # Create model
    print(f"\nInitializing model:")
    print(f"  Hidden dimensions: {config['hidden_dims']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Pooling: {config.get('pooling_type', 'max')}")
    print(f"  Feature branches: {config.get('use_feature_branches', True)}")
    
    model = ProgressionPredictionModel(
        cnn_feature_dim=actual_cnn_dim,
        hand_feature_dim=actual_hand_dim,  # Use ACTUAL dimensions from batch
        demo_feature_dim=actual_demo_dim,   # Use ACTUAL dimensions from batch
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        pooling_type=config.get('pooling_type', 'max'),  # NEW
        use_feature_branches=config.get('use_feature_branches', True),  # NEW,
        use_ktop=config.get('use_ktop', True)  # NEW
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights,
        use_scheduler=config['use_scheduler']
    )
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    
    best_val_auc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    trainer.plot_training_history(save_path=str(fold_dir / "training_history.png"))
    
    # Validation analysis
    print(f"\n{'='*70}")
    print("VALIDATION SET THRESHOLD ANALYSIS")
    print("="*70)
    
    val_results = trainer.evaluate(val_loader)
    threshold_analysis = plot_validation_roc_with_thresholds(
        y_true=val_results['labels'],
        y_pred=val_results['predictions'],
        save_path=str(fold_dir / "validation_roc_threshold_analysis.png")
    )
    
    optimal_threshold = threshold_analysis['Youden']['threshold']
    print(f"\nSelected Optimal Threshold: {optimal_threshold:.4f} (Youden's J)")
    
    # Test evaluation
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_results = trainer.evaluate(test_loader)
    
    test_metrics_default = evaluate_with_threshold(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=0.5
    )
    
    test_metrics_optimal = evaluate_with_threshold(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=optimal_threshold
    )
    
    # Plot evaluations
    plot_evaluation_metrics(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=0.5,
        save_path=str(fold_dir / "test_evaluation_default_threshold.png")
    )
    
    plot_evaluation_metrics(
        y_true=test_results['labels'],
        y_pred=test_results['predictions'],
        threshold=optimal_threshold,
        save_path=str(fold_dir / "test_evaluation_optimal_threshold.png")
    )
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'optimal_threshold': optimal_threshold,
        'threshold_analysis': threshold_analysis,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
    }, checkpoint_path)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': test_ids,
        'true_label': test_results['labels'],
        'predicted_prob': test_results['predictions'],
        'predicted_label_default': (test_results['predictions'] >= 0.5).astype(int),
        'predicted_label_optimal': (test_results['predictions'] >= optimal_threshold).astype(int)
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    print(f"\n✓ Fold {fold_idx} complete! Results saved to: {fold_dir}")
    
    return {
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
        'optimal_threshold': optimal_threshold,
        'loaded_from_checkpoint': False
    }
def run_ablation_study(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
):
    """Run ablation study with correct architecture and normalization"""
    
    print("\n" + "="*70)
    print("ABLATION STUDY - SLICE-LEVEL PREDICTIONS + PROPER NORMALIZATION")
    print("="*70)
    print(f"Configurations: {len(ABLATION_CONFIGS)}")
    
    all_ablation_results = {}
    
    for config_name, ablation_config in ABLATION_CONFIGS.items():
        print("\n" + "="*80)
        print(f"ABLATION EXPERIMENT: {config_name.upper()}")
        print("="*80)
        print(f"Description: {ablation_config['description']}")
        
        ablation_results_dir = results_base_dir / f"ablation_{config_name}"
        ablation_results_dir.mkdir(parents=True, exist_ok=True)
        
        fold_results = []
        fold_keys = sorted(kfold_splits.keys())
        
        for fold_idx in fold_keys:
            #fold_idx = int(fold_key.split('_')[1])
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
            
            # Train
            result = train_single_fold(
                features_df=features_df,
                fold_data=fold_data,
                fold_idx=fold_idx,
                config=config,
                results_dir=ablation_results_dir,
                resume_from_checkpoint=config['resume_from_checkpoint'],
                encoding_info=encoding_info  # Pass encoding info with smoking_columns
            )
            
            fold_results.append(result)
        
        summary_df, detailed_df = aggregate_fold_results(
            fold_results=fold_results,
            save_path=ablation_results_dir
        )
        
        all_ablation_results[config_name] = {
            'config': ablation_config,
            'summary': summary_df,
            'detailed': detailed_df,
            'fold_results': fold_results
        }
        
        print(f"\n✓ '{config_name}' complete!")
    
    # Create comparison
    create_ablation_comparison(all_ablation_results, results_base_dir)
    
    return all_ablation_results


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

def main():
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\full_slices_training_3_max")
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