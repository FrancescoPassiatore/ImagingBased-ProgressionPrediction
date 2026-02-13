# Main Execution Script for Three-Expert Ensemble
# Runs K-fold cross-validation with CNN Expert, LightGBM Expert, and Fusion

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import torch
import sys

# Import components
from ensemble_training import train_single_fold_ensemble
from ensemble_experts import plot_expert_comparison

sys.path.append(str(Path(__file__).parent.parent))
from utilities import IPFDataLoader, CNNFeatureExtractor


# Configuration
CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # CNN Expert parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'pooling_type': 'max',
    'hidden_dims': [256, 128, 64],
    'dropout': 0.65,
    'use_batch_norm': True,
    
    # Training parameters (CNN)
    'batch_size': 16,
    'learning_rate': 3.86e-05,
    'weight_decay': 0.0305,
    'epochs': 100,
    'early_stopping_patience': 25,
    
    # LightGBM parameters
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    },
    'lgb_num_boost_round': 500,
    'lgb_early_stopping': 50,
    
    # Fusion parameters
    'fusion_method': 'learned',  # 'fixed', 'learned', or 'stacking'
    
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


def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load demographics from train.csv and merge with patient_features"""
    
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS")
    print("="*70)
    print(f"Train CSV shape: {train_df.shape}")
    
    # Identify available demographic columns
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  ✓ Found: {col}")
        else:
            print(f"  ✗ Missing: {col}")
    
    # Merge
    demographics_df = train_df[demo_cols].copy()
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    print(f"\nEnhanced features shape: {enhanced_df.shape}")
    
    # Encode categorical variables
    if 'Sex' in enhanced_df.columns:
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"\n✓ Encoded Sex (Male=1, Female=0)")
    
    if 'SmokingStatus' in enhanced_df.columns:
        smoking_map = {
            'Never smoked': 0,
            'Ex-smoker': 1,
            'Currently smokes': 2
        }
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(smoking_map)
        print(f"✓ Encoded SmokingStatus (Never=0, Ex=1, Current=2)")
    
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
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    
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
        
        # Normalize Age
        if normalization_type == 'standard':
            age_scaler = StandardScaler()
        elif normalization_type == 'robust':
            age_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            age_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        age_scaler.fit(train_patient_df[['Age']].values)
        result_df['Age_normalized'] = age_scaler.transform(result_df[['Age']].values)
        
        print(f"  Post-normalization:")
        train_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)]
        train_patient_normalized = train_normalized.groupby('patient_id').first().reset_index()
        print(f"    Mean: {train_patient_normalized['Age_normalized'].mean():.4f}")
        print(f"    Std: {train_patient_normalized['Age_normalized'].std():.4f}")
        
        encoding_info['age_scaler'] = age_scaler
    
    # === 2. SEX (Binary Categorical) ===
    if 'Sex' in result_df.columns:
        print("\n=== PREPROCESSING SEX ===")
        
        # Convert to binary centered around 0
        result_df['Sex_encoded'] = result_df['Sex'].map({0: -1, 1: 1})  # Female=-1, Male=1
        
        print(f"  Encoded as: Female=-1, Male=1 (centered)")
        encoding_info['sex_encoding'] = {0: -1, 1: 1}
    
    # === 3. SMOKING STATUS (Multi-class Categorical) ===
    if 'SmokingStatus' in result_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS ===")
        
        # One-hot encoding
        smoking_dummies = pd.get_dummies(
            result_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        
        # Center binary features around 0
        smoking_dummies = (smoking_dummies - 0.5)
        
        # Add to result_df
        for col in smoking_dummies.columns:
            result_df[col] = smoking_dummies[col]
        
        smoking_cols = sorted(smoking_dummies.columns.tolist())
        encoding_info['smoking_columns'] = smoking_cols
        
        print(f"  One-hot encoded into {len(smoking_cols)} features")
        print(f"  Values centered to [-0.5, 0.5]")
    
    return result_df, encoding_info


def normalize_features_per_fold(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    hand_feature_cols: list,
    demo_feature_cols: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize features using ONLY training set statistics
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    
    result_df = features_df.copy()
    scalers = {}
    
    # Identify available features
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
    
    if not available_hand and not available_demo:
        return result_df, scalers
    
    print(f"\nNormalizing features ({normalization_type} scaler):")
    
    # Get training set statistics (one sample per patient)
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    print(f"  Training set: {len(train_patient_df)} patients")
    
    # === HAND-CRAFTED FEATURES ===
    if available_hand:
        print(f"  Hand-crafted: {len(available_hand)} features")
        
        # Create scaler
        if normalization_type == 'standard':
            hand_scaler = StandardScaler()
        elif normalization_type == 'robust':
            hand_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            hand_scaler = MinMaxScaler()
        
        # Fit on training
        hand_scaler.fit(train_patient_df[available_hand].values)
        
        # Transform entire dataset
        result_df[available_hand] = hand_scaler.transform(result_df[available_hand].values)
        
        # Verify
        train_df_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
        train_patient_normalized = train_df_normalized.groupby('patient_id').first().reset_index()
        
        print(f"\n  Post-normalization (Training Set):")
        for col in available_hand:
            mean = train_patient_normalized[col].mean()
            std = train_patient_normalized[col].std()
            print(f"    {col:30s}: mean={mean:10.4f}, std={std:10.4f}")
        
        scalers['hand_scaler'] = hand_scaler
    
    # === DEMOGRAPHIC FEATURES ===
    if available_demo:
        print(f"\n=== DEMOGRAPHIC FEATURES ===")
        print(f"  Found {len(available_demo)} demographic columns")
        
        result_df, encoding_info = preprocess_demographics_improved(
            result_df,
            train_patient_ids,
            normalization_type
        )
        
        scalers['demo_encoding'] = encoding_info
        print(f"  ✓ Demographics preprocessed")
    
    return result_df, scalers


def create_feature_set_for_fold(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """Create feature set for a specific fold with normalization"""
    
    print("\n" + "="*70)
    print("CREATING FEATURE SET FOR FOLD")
    print("="*70)
    
    # Start with CNN features
    result_df = slice_features_df.copy()
    
    # Merge patient-level features
    patient_level_cols = ['Patient'] + HAND_FEATURE_COLS + DEMO_FEATURE_COLS
    available_cols = [c for c in patient_level_cols if c in patient_features_df.columns]
    
    result_df = result_df.merge(
        patient_features_df[available_cols],
        left_on='patient_id',
        right_on='Patient',
        how='left'
    )
    result_df.drop('Patient', axis=1, inplace=True)
    
    # Handle missing values BEFORE normalization
    all_features_to_check = HAND_FEATURE_COLS + DEMO_FEATURE_COLS
    available_features = [f for f in all_features_to_check if f in result_df.columns]
    
    missing = result_df[available_features].isnull().sum()
    
    if missing.any():
        print(f"\n⚠️ Handling missing values:")
        
        train_stats = result_df[result_df['patient_id'].isin(fold_data['train'])].groupby('patient_id').first()
        
        for col in available_features:
            if result_df[col].isnull().any():
                if col == 'Age' or col in HAND_FEATURE_COLS:
                    fill_value = train_stats[col].median()
                    result_df[col].fillna(fill_value, inplace=True)
                    print(f"    {col}: filled with train median ({fill_value:.2f})")
                else:
                    result_df[col].fillna(0, inplace=True)
                    print(f"    {col}: filled with 0 (unknown)")
    
    # NORMALIZATION (using only training set)
    result_df, scalers = normalize_features_per_fold(
        features_df=result_df,
        train_patient_ids=fold_data['train'],
        hand_feature_cols=HAND_FEATURE_COLS,
        demo_feature_cols=DEMO_FEATURE_COLS,
        normalization_type=normalization_type
    )
    
    encoding_info = scalers.get('demo_encoding', {})
    
    # Feature summary
    cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
    hand_cols = [c for c in HAND_FEATURE_COLS if c in result_df.columns]
    
    actual_demo_features = 0
    if DEMO_FEATURE_COLS:
        if 'Age' in result_df.columns:
            actual_demo_features += 1
        if 'Sex' in result_df.columns:
            actual_demo_features += 1
        if 'SmokingStatus' in result_df.columns:
            smoking_cols = encoding_info.get('smoking_columns', [])
            actual_demo_features += len(smoking_cols)
    
    print(f"\nFinal feature composition:")
    print(f"  CNN features: {len(cnn_cols)}")
    print(f"  Hand-crafted features: {len(hand_cols)}")
    print(f"  Demographic features: {actual_demo_features}")
    
    return result_df, encoding_info


def run_kfold_ensemble(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    config: dict,
    results_base_dir: Path
):
    """Run K-fold cross-validation with three-expert ensemble"""
    
    print("\n" + "="*70)
    print("THREE-EXPERT ENSEMBLE - K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    fold_keys = sorted(kfold_splits.keys())
    
    for fold_idx in fold_keys:
        fold_data = kfold_splits[fold_idx]
        
        # Create feature set for this fold (with proper normalization)
        features_df, encoding_info = create_feature_set_for_fold(
            slice_features_df=slice_features_df,
            patient_features_df=patient_features_df,
            fold_data=fold_data,
            normalization_type=config['normalization_type']
        )
        
        # Train three-expert ensemble for this fold
        result = train_single_fold_ensemble(
            features_df=features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=config,
            results_dir=results_base_dir,
            hand_feature_cols=HAND_FEATURE_COLS,
            demo_feature_cols=DEMO_FEATURE_COLS,
            encoding_info=encoding_info
        )
        
        fold_results.append(result)
    
    # Aggregate results
    aggregate_results(fold_results, results_base_dir)
    
    return fold_results


def aggregate_results(fold_results: list, save_path: Path):
    """Aggregate results across all folds"""
    
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    # Collect metrics for each expert
    cnn_aucs = [r['test_results']['cnn']['auc'] for r in fold_results]
    lgb_aucs = [r['test_results']['lgb']['auc'] for r in fold_results]
    fusion_aucs = [r['test_results']['fusion']['auc'] for r in fold_results]
    
    # Create summary
    summary_data = {
        'Expert': ['CNN', 'LightGBM', 'Fusion'],
        'Mean_AUC': [
            np.mean(cnn_aucs),
            np.mean(lgb_aucs),
            np.mean(fusion_aucs)
        ],
        'Std_AUC': [
            np.std(cnn_aucs),
            np.std(lgb_aucs),
            np.std(fusion_aucs)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Save
    summary_df.to_csv(save_path / "aggregate_summary.csv", index=False)
    
    # Detailed results
    detailed_results = []
    for r in fold_results:
        detailed_results.append({
            'fold': r['fold_idx'],
            'cnn_auc': r['test_results']['cnn']['auc'],
            'lgb_auc': r['test_results']['lgb']['auc'],
            'fusion_auc': r['test_results']['fusion']['auc'],
            'fusion_weight_cnn': r['fusion_weights'][0],
            'fusion_weight_lgb': r['fusion_weights'][1]
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(save_path / "detailed_fold_results.csv", index=False)
    
    print(f"\nResults saved to: {save_path}")


def main():
    """Main execution function"""
    
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_experts\three_expert_ensemble")
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=CONFIG['gt_path'],
        features_path=CONFIG['patient_features_path'],
        npy_dir=CONFIG['ct_scan_path']
    )
    patient_data, _ = data_loader.get_patient_data()
    
    # Extract CNN features
    print("\n" + "="*70)
    print("EXTRACTING CNN FEATURES")
    print("="*70)
    
    feature_extractor = CNNFeatureExtractor(
        model_name=CONFIG['backbone'],
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
    
    patient_features_df = pd.read_csv(CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=CONFIG['train_csv_path'],
        patient_features_df=patient_features_df
    )
    
    # Load K-Fold splits
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"\nLoaded {len(kfold_splits)} folds")
    
    # Run three-expert ensemble with K-fold CV
    fold_results = run_kfold_ensemble(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        config=CONFIG,
        results_base_dir=results_base_dir
    )
    
    print("\n" + "="*70)
    print("✓ THREE-EXPERT ENSEMBLE COMPLETE!")
    print("="*70)
    print(f"Results: {results_base_dir}")


if __name__ == "__main__":
    main()