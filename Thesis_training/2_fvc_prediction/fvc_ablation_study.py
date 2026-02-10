# FVC Ablation Study with Correct Normalization
# IMPORTANT: Each fold normalizes using ONLY its own training set

from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader, create_fvc_dataloaders
from fvc_prediction_model import (
    FVCPredictionModel,
    FVCModelTrainer,
    plot_evaluation_metrics
)


# Feature configurations for ablation study
ABLATION_CONFIGS = {
    'cnn_only': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': 'CNN features + FVC baseline only'
    },
    'cnn_hand': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': False,
        'description': 'CNN + Hand-crafted features + FVC baseline'
    },
    'cnn_demo': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': True,
        'description': 'CNN + Demographics + FVC baseline'
    },
    'full': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': True,
        'description': 'CNN + Hand-crafted + Demographics + FVC baseline (full)'
    },
}


BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # Model parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'pooling_type': 'max',  # 'max', 'mean', or 'attention'
    'use_fvc_branch':True,
    
    # Training parameters
    'batch_size': 8,
    'learning_rate': 5e-4,
    'weight_decay': 0.05,
    'epochs': 100,
    'early_stopping_patience': 20,
    'use_scheduler': True,
    
    # Model architecture
    'hidden_dims': [256, 128],
    'dropout': 0.7,
    'use_batch_norm': True,
    
    'resume_from_checkpoint': True,
    'normalization_type': 'standard',  # 'standard', 'minmax', or 'robust'
}


# Feature column names
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

# Flag to use one-hot encoding for smoking (recommended for neural networks)
USE_ONEHOT_SMOKING = True


def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load demographics from train.csv and merge with patient_features
    """
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS FROM TRAIN.CSV")
    print("="*70)
    print(f"Train CSV shape: {train_df.shape}")
    print(f"Columns available: {train_df.columns.tolist()}")
    
    # Identify available demographic columns
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  Found: {col}")
        else:
            print(f"  ⚠️  Missing: {col}")
    
    # Merge
    demographics_df = train_df[demo_cols].copy().drop_duplicates('Patient')
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    print(f"\nEnhanced features shape: {enhanced_df.shape}")
    
    # Encode categorical variables
    if 'Sex' in enhanced_df.columns:
        print(f"\nEncoding Sex (categorical -> numeric)")
        print(f"  Unique values: {enhanced_df['Sex'].unique()}")
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"  Encoded as: Male=1, Female=0")
    
    if 'SmokingStatus' in enhanced_df.columns:
        print(f"\nEncoding SmokingStatus (categorical -> numeric)")
        print(f"  Unique values: {enhanced_df['SmokingStatus'].unique()}")
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
    features_df: pd.DataFrame,
    train_patient_ids: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Improved demographics preprocessing:
    1. Separates continuous (Age) from categorical (Sex, Smoking)
    2. Normalizes Age properly using training set only
    3. Centers Sex encoding: Female=-1, Male=1
    4. One-hot encodes SmokingStatus and centers around 0
    5. Returns encoding_info for dataset usage
    """
    result_df = features_df.copy()
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
        
        result_df = pd.concat([result_df, smoking_dummies], axis=1)
        
        smoking_cols = smoking_dummies.columns.tolist()
        print(f"  One-hot encoded into {len(smoking_cols)} features (centered at 0):")
        for col in smoking_cols:
            print(f"    {col}")
        
        encoding_info['smoking_columns'] = smoking_cols
        encoding_info['smoking_type'] = 'onehot_centered'
    
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
    Normalize features using ONLY statistics from the training set
    Uses improved demographics preprocessing with proper centering
    Returns the normalized dataframe AND the scalers/encoding info for inverse transform
    """
    result_df = features_df.copy()
    scalers = {}
    
    # Get training set statistics
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    print(f"\nNormalizing features ({normalization_type} scaler):")
    print(f"  Training set: {len(train_patient_df)} patients")
    
    # === CRITICAL: NORMALIZE FVC VALUES ===
    fvc_cols = ['baselinefvc', 'gt_fvc52']
    print(f"\n=== FVC VALUES (CRITICAL - must normalize!) ===")
    print(f"  Pre-normalization:")
    for col in fvc_cols:
        mean = train_patient_df[col].mean()
        std = train_patient_df[col].std()
        print(f"    {col:20s}: mean={mean:10.2f}, std={std:10.2f}")
    
    # Create FVC scaler
    if normalization_type == 'standard':
        fvc_scaler = StandardScaler()
    elif normalization_type == 'robust':
        from sklearn.preprocessing import RobustScaler
        fvc_scaler = RobustScaler()
    elif normalization_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        fvc_scaler = MinMaxScaler()
    
    # Fit on training FVC values
    fvc_scaler.fit(train_patient_df[fvc_cols].values)
    
    # Transform ALL FVC values
    result_df[fvc_cols] = fvc_scaler.transform(result_df[fvc_cols].values)
    scalers['fvc'] = fvc_scaler
    
    # Verify normalization
    train_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_normalized = train_normalized.groupby('patient_id').first().reset_index()
    print(f"  Post-normalization:")
    for col in fvc_cols:
        mean = train_patient_normalized[col].mean()
        std = train_patient_normalized[col].std()
        print(f"    {col:20s}: mean={mean:10.4f}, std={std:10.4f}")
        if abs(mean) > 0.01:
            print(f"      ⚠️ WARNING: Mean not close to 0!")
    
    # === HAND-CRAFTED FEATURES ===
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    if available_hand:
        print(f"\n=== HAND-CRAFTED FEATURES ===")
        print(f"  Normalizing {len(available_hand)} features")
        
        if normalization_type == 'standard':
            hand_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            hand_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            hand_scaler = MinMaxScaler()
        
        hand_scaler.fit(train_patient_df[available_hand].values)
        result_df[available_hand] = hand_scaler.transform(result_df[available_hand].values)
        scalers['hand'] = hand_scaler
        
        print(f"  ✓ Normalized using training set statistics")
    
    # === DEMOGRAPHIC FEATURES (IMPROVED PREPROCESSING) ===
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
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
        
        print(f"  ✓ Demographics preprocessed with proper centering and encoding")
    
    return result_df, scalers


def create_feature_set_for_fold(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    ablation_config: dict,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame,dict]:
    """
    Create feature set for a specific fold with normalization
    """
    print("\n" + "="*70)
    print(f"FEATURE SET: {ablation_config['description']}")
    print("="*70)
    
    # Start with CNN features (already includes baselinefvc and gt_fvc52 from feature extraction)
    result_df = slice_features_df.copy()
    
    # Remove 'Patient' column if it exists (to avoid conflict during merge)
    if 'Patient' in result_df.columns:
        result_df.drop('Patient', axis=1, inplace=True)
    
    # Determine which features to add
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
    
    # Merge patient-level features
    if len(patient_level_cols) > 1:
        result_df = result_df.merge(
            patient_features_df[patient_level_cols],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        result_df.drop('Patient', axis=1, inplace=True)
        
        # Handle missing values BEFORE normalization
        all_features_to_check = hand_to_add + demo_to_add
        missing = result_df[all_features_to_check].isnull().sum()
        
        if missing.any():
            print(f"\n  ⚠️ Handling missing values:")
            
            train_stats = result_df[result_df['patient_id'].isin(fold_data['train'])].groupby('patient_id').first()
            
            for col in all_features_to_check:
                if result_df[col].isnull().any():
                    if col == 'Age' or col in hand_to_add:
                        fill_value = train_stats[col].median()
                        result_df[col].fillna(fill_value, inplace=True)
                        print(f"    {col}: filled {missing[col]} values with train median ({fill_value:.2f})")
                    else:
                        result_df[col].fillna(0, inplace=True)
                        print(f"    {col}: filled {missing[col]} values with 0 (unknown)")
        
        
    # ALWAYS NORMALIZE (including FVC)
    result_df, scalers = normalize_features_per_fold(
        features_df=result_df,
        train_patient_ids=fold_data['train'],
        hand_feature_cols=hand_to_add,
        demo_feature_cols=demo_to_add,
        normalization_type=normalization_type
    )

    # Feature dimension summary
    cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
    print(f"\nFinal feature composition:")
    print(f"  CNN features: {len(cnn_cols)}")
    print(f"  FVC baseline: 1 (normalized)")
    print(f"  Hand-crafted features: {len(hand_to_add)}")
    print(f"  Demographic features: {len(demo_to_add)}")
    print(f"  Total input: {len(cnn_cols) + 1 + len(hand_to_add) + len(demo_to_add)}")
    


    return result_df,scalers
    



def train_single_fold(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path,
    scalers:dict,
    resume_from_checkpoint: bool = True,
    hand_feature_cols: list = None,
    demo_feature_cols: list = None
):
    """Train FVC prediction model on a single fold"""
    
    if hand_feature_cols is None:
        hand_feature_cols = HAND_FEATURE_COLS
    if demo_feature_cols is None:
        demo_feature_cols = DEMO_FEATURE_COLS
    
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
            'test_metrics' in checkpoint and
            'val_mae' in checkpoint
        )
        
        if is_complete:
            print("\n✓ Fold already completed! Loading saved results...")
            print(f"  Val MAE: {checkpoint.get('val_mae'):.2f} mL")
            print(f"  Test MAE: {checkpoint['test_metrics'].get('mae'):.2f} mL")
            print(f"  Test R²: {checkpoint['test_metrics'].get('r2'):.4f}")
            
            return {
                'fold_idx': fold_idx,
                'val_mae': checkpoint.get('val_mae'),
                'test_metrics': checkpoint.get('test_metrics', {}),
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
    available_demo_cols = [c for c in demo_feature_cols if c in features_df.columns]
    
    print(f"\nFeature availability:")
    print(f"  Hand-crafted: {len(available_hand_cols)}/{len(hand_feature_cols)}")
    print(f"  Demographics: {len(available_demo_cols)}/{len(demo_feature_cols)}")
    
    # Extract encoding info for demographics (from scalers dict)
    encoding_info = scalers.get('demo_encoding', {})
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_fvc_dataloaders(
        features_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=available_hand_cols,
        demo_feature_cols=available_demo_cols,
        encoding_info=encoding_info
    )
    
    # Get actual dimensions
    sample_batch = next(iter(train_loader))
    actual_cnn_dim = sample_batch['cnn_features'].shape[2]
    
    # Get actual hand-crafted and demographic dimensions from sample
    actual_hand_dim = sample_batch['hand_features'].shape[1] if sample_batch['hand_features'] is not None else 0
    actual_demo_dim = sample_batch['demo_features'].shape[1] if sample_batch['demo_features'] is not None else 0
    
    print(f"\nActual feature dimensions:")
    print(f"  CNN features: {actual_cnn_dim}")
    print(f"  Hand-crafted features: {actual_hand_dim}")
    print(f"  Demographic features: {actual_demo_dim}")
    print(f"  Total patient-level: {actual_hand_dim + actual_demo_dim}")
    
    # Create model
    print(f"\nInitializing FVC prediction model:")
    print(f"  Hidden dimensions: {config['hidden_dims']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Pooling: {config['pooling_type']}")
    
    model = FVCPredictionModel(
        cnn_feature_dim=actual_cnn_dim,
        hand_feature_dim=actual_hand_dim,
        demo_feature_dim=actual_demo_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        pooling_type=config['pooling_type'],
        use_fvc_branch=config.get('use_fvc_branch',True)
    )
    
    # Create trainer
    trainer = FVCModelTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        use_scheduler=config['use_scheduler']
    )
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    
    best_val_mae = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    trainer.plot_training_history(save_path=str(fold_dir / "training_history.png"))
    
    # Test evaluation
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_results = trainer.evaluate(test_loader)

    #Denormalize FVC predictions
    fvc_scaler = scalers['fvc']

    # Create dummy array with both baseline and 52-week columns
    dummy_baseline = np.zeros((len(test_results['targets']), 1))
    
    # Denormalize targets
    targets_with_dummy = np.column_stack([dummy_baseline, test_results['targets']])
    targets_denorm = fvc_scaler.inverse_transform(targets_with_dummy)[:, 1]
    
    # Denormalize predictions
    preds_with_dummy = np.column_stack([dummy_baseline, test_results['predictions']])
    preds_denorm = fvc_scaler.inverse_transform(preds_with_dummy)[:, 1]
    
    # Plot evaluation
    plot_evaluation_metrics(
        y_true=targets_denorm,
        y_pred=preds_denorm,
        save_path=str(fold_dir / "test_evaluation.png")
    )

    # Calculate metrics on denormalized values
    mae_denorm = mean_absolute_error(targets_denorm, preds_denorm)
    rmse_denorm = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
    r2_denorm = r2_score(targets_denorm, preds_denorm)

    print(f"\nDenormalized Test Metrics (mL):")
    print(f"  MAE: {mae_denorm:.2f} mL")
    print(f"  RMSE: {rmse_denorm:.2f} mL")
    print(f"  R²: {r2_denorm:.4f}")
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'fold_idx': fold_idx,
        'val_mae': best_val_mae,
        'test_metrics': {
            'mae': test_results['mae'],
            'rmse': test_results['rmse'],
            'r2': test_results['r2']
        },
    }, checkpoint_path)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': test_ids,
        'true_fvc': test_results['targets'],
        'predicted_fvc': test_results['predictions']
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    print(f"\n✓ Fold {fold_idx} complete! Results saved to: {fold_dir}")
    
    return {
        'fold_idx': fold_idx,
        'val_mae': best_val_mae,
        'test_metrics': {
            'mae': test_results['mae'],
            'rmse': test_results['rmse'],
            'r2': test_results['r2']
        },
        'loaded_from_checkpoint': False
    }


def aggregate_fold_results(fold_results: list, save_path: Path):
    """Aggregate results across folds"""
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    # Collect metrics
    val_maes = [r['val_mae'] for r in fold_results]
    test_maes = [r['test_metrics']['mae'] for r in fold_results]
    test_rmses = [r['test_metrics']['rmse'] for r in fold_results]
    test_r2s = [r['test_metrics']['r2'] for r in fold_results]
    
    # Create summary
    summary_data = {
        'Metric': ['Val MAE', 'Test MAE', 'Test RMSE', 'Test R²'],
        'Mean': [
            np.mean(val_maes),
            np.mean(test_maes),
            np.mean(test_rmses),
            np.mean(test_r2s)
        ],
        'Std': [
            np.std(val_maes),
            np.std(test_maes),
            np.std(test_rmses),
            np.std(test_r2s)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(save_path / "aggregate_metrics_summary.csv", index=False)
    
    # Detailed results
    detailed_results = []
    for r in fold_results:
        fold_data = {
            'fold': r['fold_idx'],
            'val_mae': r['val_mae'],
            'test_mae': r['test_metrics']['mae'],
            'test_rmse': r['test_metrics']['rmse'],
            'test_r2': r['test_metrics']['r2']
        }
        detailed_results.append(fold_data)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(save_path / "detailed_fold_results.csv", index=False)
    
    return summary_df, detailed_df


def run_ablation_study(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
):
    """Run ablation study for FVC prediction"""
    
    print("\n" + "="*70)
    print("FVC PREDICTION ABLATION STUDY")
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
            fold_data = kfold_splits[fold_idx]
            
            # Create feature set with normalization
            features_df,scalers = create_feature_set_for_fold(
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
                scalers=scalers,
                resume_from_checkpoint=config['resume_from_checkpoint']
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
        
        val_mae = summary[summary['Metric'] == 'Val MAE']['Mean'].values[0]
        test_mae = summary[summary['Metric'] == 'Test MAE']['Mean'].values[0]
        test_rmse = summary[summary['Metric'] == 'Test RMSE']['Mean'].values[0]
        test_r2 = summary[summary['Metric'] == 'Test R²']['Mean'].values[0]
        
        comparison_data.append({
            'Configuration': config_name,
            'Description': results['config']['description'],
            'Val_MAE': val_mae,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_MAE')  # Lower is better
    comparison_df.to_csv(results_dir / "ablation_comparison.csv", index=False)
    
    print("\nResults (sorted by Test MAE):")
    print(comparison_df.to_string(index=False))
    
    # Best configuration
    best = comparison_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {best['Configuration']}")
    print(f"   {best['Description']}")
    print(f"   Test MAE: {best['Test_MAE']:.2f} mL")
    print(f"   Test R²: {best['Test_R2']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    configs = comparison_df['Configuration'].values
    x = range(len(configs))
    
    axes[0].bar(x, comparison_df['Test_MAE'], alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Test MAE (mL)')
    axes[0].set_title('MAE Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, comparison_df['Test_RMSE'], alpha=0.7, color='seagreen')
    axes[1].set_ylabel('Test RMSE (mL)')
    axes[1].set_title('RMSE Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, comparison_df['Test_R2'], alpha=0.7, color='coral')
    axes[2].set_ylabel('Test R²')
    axes[2].set_title('R² Score Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(configs, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved!")


def main():
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\2_fvc_prediction\ablation_study_results_stratified")
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
        save_path=None
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