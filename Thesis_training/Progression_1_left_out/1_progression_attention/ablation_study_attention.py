"""
Ablation Study with Attention MIL
==================================

This script extends the original ablation study to compare:
1. Max pooling (baseline)
2. Gated attention MIL
3. Multi-head attention MIL

For each pooling method, we test different feature combinations:
- CNN only
- CNN + Hand-crafted
- CNN + Demographics  
- CNN + Hand-crafted + Demographics (full)
"""

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
    IPFDataLoader,
    create_dataloaders,
    compute_class_weights
)
from model_train import (
    ModelTrainer,
    HAND_FEATURE_COLS,
    DEMO_FEATURE_COLS,
    plot_evaluation_metrics,
    evaluate_with_threshold,
    plot_validation_roc_with_thresholds,
    aggregate_fold_results
)

# Import attention models
from attention_model import (
    AttentionMILProgressionModel,
    PoolingComparisonModel,
    visualize_attention_weights
)


# =============================================================================
# POOLING CONFIGURATIONS
# =============================================================================

POOLING_CONFIGS = {
    'gated_attention': {
        'pooling_type': 'attention',
        'use_attention': True,
        'attention_type': 'gated',
        'attention_hidden_dim': 128,
        'description': 'Gated attention MIL'
    },
    'multihead_attention_4': {
        'pooling_type': 'attention',
        'use_attention': True,
        'attention_type': 'multihead',
        'num_attention_heads': 4,
        'attention_hidden_dim': 128,
        'description': 'Multi-head attention MIL (4 heads)'
    },
    'multihead_attention_8': {
        'pooling_type': 'attention',
        'use_attention': True,
        'attention_type': 'multihead',
        'num_attention_heads': 8,
        'attention_hidden_dim': 128,
        'description': 'Multi-head attention MIL (8 heads)'
    }
}


# Feature configurations (same as before)
ABLATION_CONFIGS = {
    'cnn_only': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': False,
        'description': 'CNN features only'
    },
    'cnn_hand': {
        'use_cnn_features': True,
        'use_hand_features': True,
        'use_demographics': False,
        'description': 'CNN + Hand-crafted features'
    },
    'cnn_demo': {
        'use_cnn_features': True,
        'use_hand_features': False,
        'use_demographics': True,
        'description': 'CNN + Demographics'
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
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # Model parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    'use_feature_branches': True,
    
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
    'hidden_dims': [256, 128, 64],
    'dropout': 0.4,
    'use_batch_norm': True,
    
    'resume_from_checkpoint': True,
    'normalization_type': 'standard',
}


def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load demographics from train.csv and merge with patient_features"""
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS FROM TRAIN.CSV")
    print("="*70)
    print(f"Train CSV shape: {train_df.shape}")
    
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  Found: {col}")
        else:
            print(f"  ⚠️  Missing: {col}")
    
    demographics_df = train_df[demo_cols].copy()
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    # Encode categorical variables
    if 'Sex' in enhanced_df.columns:
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
    
    if 'SmokingStatus' in enhanced_df.columns:
        smoking_map = {
            'Never smoked': 0,
            'Ex-smoker': 1,
            'Currently smokes': 2
        }
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(smoking_map)
    
    return enhanced_df


def preprocess_demographics_improved(
    result_df: pd.DataFrame,
    train_patient_ids: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """Improved demographics preprocessing with proper centering and encoding"""
    encoding_info = {}
    
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)]
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    
    # Age (Continuous)
    if 'Age' in result_df.columns:
        print("\n=== PREPROCESSING AGE ===")
        
        if normalization_type == 'standard':
            age_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            age_scaler = RobustScaler()
        elif normalization_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            age_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        age_scaler.fit(train_patient_df[['Age']].values)
        result_df['Age_normalized'] = age_scaler.transform(result_df[['Age']].values)
        
        encoding_info['age_scaler'] = age_scaler
    
    # Sex (Binary Categorical)
    if 'Sex' in result_df.columns:
        print("\n=== PREPROCESSING SEX ===")
        result_df['Sex_encoded'] = result_df['Sex'].map({0: -1, 1: 1})
        encoding_info['sex_encoding'] = {0: -1, 1: 1, 'description': 'Female=-1, Male=1'}
    
    # Smoking Status (Multi-class Categorical)
    if 'SmokingStatus' in result_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS ===")
        smoking_dummies = pd.get_dummies(
            result_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        smoking_dummies = (smoking_dummies - 0.5)
        
        for col in smoking_dummies.columns:
            result_df[col] = smoking_dummies[col]
        
        smoking_cols = sorted(smoking_dummies.columns.tolist())
        encoding_info['smoking_columns'] = smoking_cols
    
    return result_df, encoding_info


def normalize_features_per_fold(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    hand_feature_cols: list,
    demo_feature_cols: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """Normalize features using ONLY statistics from training set"""
    result_df = features_df.copy()
    scalers = {}
    
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
    
    if not available_hand and not available_demo:
        return result_df, scalers
    
    print(f"\nNormalizing features ({normalization_type} scaler):")
    
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    
    # Hand-crafted features
    if available_hand:
        print(f"  Hand-crafted: {len(available_hand)} features")
        
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
        scalers['hand_scaler'] = hand_scaler
    
    # Demographic features
    if available_demo:
        print(f"\n=== DEMOGRAPHIC FEATURES (IMPROVED) ===")
        result_df, encoding_info = preprocess_demographics_improved(
            result_df,
            train_patient_ids,
            normalization_type
        )
        scalers['demo_encoding'] = encoding_info
    
    return result_df, scalers


def create_feature_set_for_fold(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    ablation_config: dict,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """Create feature set for a specific fold with correct normalization"""
    print("\n" + "="*70)
    print(f"FEATURE SET: {ablation_config['description']}")
    print("="*70)
    
    result_df = slice_features_df.copy()
    
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
    
    encoding_info = {}
    
    if len(patient_level_cols) > 1:
        result_df = result_df.merge(
            patient_features_df[patient_level_cols],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        result_df.drop('Patient', axis=1, inplace=True)
        
        # Handle missing values
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
                        print(f"    {col}: filled with train median ({fill_value:.2f})")
                    else:
                        result_df[col].fillna(0, inplace=True)
                        print(f"    {col}: filled with 0 (unknown)")
        
        # Normalization
        if hand_to_add or demo_to_add:
            result_df, scalers = normalize_features_per_fold(
                features_df=result_df,
                train_patient_ids=fold_data['train'],
                hand_feature_cols=hand_to_add,
                demo_feature_cols=demo_to_add,
                normalization_type=normalization_type
            )
            encoding_info = scalers.get('demo_encoding', {})
    
    # Feature dimension summary
    cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
    
    actual_demo_features = 0
    if demo_to_add:
        if 'Age' in demo_to_add:
            actual_demo_features += 1
        if 'Sex' in demo_to_add:
            actual_demo_features += 1
        if 'SmokingStatus' in demo_to_add:
            smoking_cols = encoding_info.get('smoking_columns', [])
            actual_demo_features += len(smoking_cols)
    
    print(f"\nFinal feature composition:")
    print(f"  CNN features: {len(cnn_cols)}")
    print(f"  Hand-crafted features: {len(hand_to_add)}")
    print(f"  Demographic features: {actual_demo_features}")
    print(f"  Total: {len(cnn_cols) + len(hand_to_add) + actual_demo_features}")
    
    return result_df, encoding_info


def train_single_fold_with_attention(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    pooling_config: dict,
    results_dir: Path,
    resume_from_checkpoint: bool = True,
    hand_feature_cols: list = None,
    demo_feature_cols: list = None,
    encoding_info: dict = None
):
    """Train model on a single fold with specified pooling method"""
    
    if hand_feature_cols is None:
        hand_feature_cols = HAND_FEATURE_COLS
    if demo_feature_cols is None:
        demo_feature_cols = DEMO_FEATURE_COLS
    if encoding_info is None:
        encoding_info = {}
    
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
            return {
                'fold_idx': fold_idx,
                'val_auc': checkpoint.get('val_auc'),
                'test_metrics_default': checkpoint.get('test_metrics_default', {}),
                'test_metrics_optimal': checkpoint.get('test_metrics_optimal', {}),
                'optimal_threshold': checkpoint.get('optimal_threshold'),
                'loaded_from_checkpoint': True
            }
    
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx} - {pooling_config['description']}")
    print("="*70)
    
    # Create dataloaders
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    available_hand_cols = [c for c in hand_feature_cols if c in features_df.columns]
    available_demo_cols = [c for c in demo_feature_cols if c in features_df.columns]
    
    # Compute class weights
    class_weights = compute_class_weights(features_df, train_ids)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
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
    actual_hand_dim = sample_batch['hand_features'].shape[1] if sample_batch['hand_features'] is not None else 0
    actual_demo_dim = sample_batch['demo_features'].shape[1] if sample_batch['demo_features'] is not None else 0
    
    print(f"\nActual feature dimensions:")
    print(f"  CNN features: {actual_cnn_dim}")
    print(f"  Hand-crafted features: {actual_hand_dim}")
    print(f"  Demographic features: {actual_demo_dim}")
    
    # Create model with attention
    print(f"\nInitializing model with {pooling_config['description']}:")
    
    if pooling_config['use_attention']:
        model = AttentionMILProgressionModel(
            cnn_feature_dim=actual_cnn_dim,
            hand_feature_dim=actual_hand_dim,  # Use ACTUAL dimensions from batch
            demo_feature_dim=actual_demo_dim,   # Use ACTUAL dimensions from batch
            attention_hidden_dim=pooling_config.get('attention_hidden_dim', 128),
            attention_type=pooling_config.get('attention_type', 'gated'),
            num_attention_heads=pooling_config.get('num_attention_heads', 4),
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
            use_feature_branches=config.get('use_feature_branches', True)
        )
    else:
        # Use regular model with max/mean pooling
        from model_train import ProgressionPredictionModel
        model = ProgressionPredictionModel(
            cnn_feature_dim=actual_cnn_dim,
            hand_feature_dim=actual_hand_dim,  # Use ACTUAL dimensions from batch
            demo_feature_dim=actual_demo_dim,   # Use ACTUAL dimensions from batch
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
            pooling_type=pooling_config['pooling_type'],
            use_feature_branches=config.get('use_feature_branches', True)
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
        'pooling_config': pooling_config,
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
    
    print(f"\n✓ Fold {fold_idx} complete!")
    
    return {
        'fold_idx': fold_idx,
        'val_auc': best_val_auc,
        'test_metrics_default': test_metrics_default,
        'test_metrics_optimal': test_metrics_optimal,
        'optimal_threshold': optimal_threshold,
        'loaded_from_checkpoint': False
    }


def run_comprehensive_ablation_study(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path,
    pooling_configs_to_test: list = None,
    feature_configs_to_test: list = None
):
    """
    Run comprehensive ablation study testing both pooling methods and feature combinations
    """
    
    if pooling_configs_to_test is None:
        pooling_configs_to_test = list(POOLING_CONFIGS.keys())
    
    if feature_configs_to_test is None:
        feature_configs_to_test = list(ABLATION_CONFIGS.keys())
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ABLATION STUDY")
    print("="*70)
    print(f"Testing {len(pooling_configs_to_test)} pooling methods")
    print(f"Testing {len(feature_configs_to_test)} feature combinations")
    print(f"Total experiments: {len(pooling_configs_to_test) * len(feature_configs_to_test)}")
    
    all_results = {}
    
    for pooling_name in pooling_configs_to_test:
        pooling_config = POOLING_CONFIGS[pooling_name]
        
        print(f"\n{'='*80}")
        print(f"POOLING METHOD: {pooling_name.upper()}")
        print(f"{'='*80}")
        
        pooling_results = {}
        
        for feature_name in feature_configs_to_test:
            ablation_config = ABLATION_CONFIGS[feature_name]
            
            print(f"\n{'-'*80}")
            print(f"Feature Configuration: {feature_name}")
            print(f"{'-'*80}")
            
            experiment_name = f"{pooling_name}_{feature_name}"
            experiment_results_dir = results_base_dir / experiment_name
            experiment_results_dir.mkdir(parents=True, exist_ok=True)
            
            fold_results = []
            
            for fold_idx in sorted(kfold_splits.keys()):
                fold_data = kfold_splits[fold_idx]
                
                # Create features for this fold
                features_df, encoding_info = create_feature_set_for_fold(
                    slice_features_df=slice_features_df,
                    patient_features_df=patient_features_df,
                    fold_data=fold_data,
                    ablation_config=ablation_config,
                    normalization_type=base_config['normalization_type']
                )
                
                # Train
                config = base_config.copy()
                config['results_save_dir'] = experiment_results_dir
                
                result = train_single_fold_with_attention(
                    features_df=features_df,
                    fold_data=fold_data,
                    fold_idx=fold_idx,
                    config=config,
                    pooling_config=pooling_config,
                    results_dir=experiment_results_dir,
                    resume_from_checkpoint=config['resume_from_checkpoint'],
                    encoding_info=encoding_info
                )
                
                fold_results.append(result)
            
            # Aggregate results
            summary_df, detailed_df = aggregate_fold_results(
                fold_results=fold_results,
                save_path=experiment_results_dir
            )
            
            pooling_results[feature_name] = {
                'config': ablation_config,
                'summary': summary_df,
                'detailed': detailed_df,
                'fold_results': fold_results
            }
        
        all_results[pooling_name] = pooling_results
    
    # Create comprehensive comparison
    create_comprehensive_comparison(all_results, results_base_dir)
    
    return all_results


def create_comprehensive_comparison(all_results: dict, results_dir: Path):
    """Create comprehensive comparison across all experiments"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    for pooling_name, pooling_results in all_results.items():
        for feature_name, results in pooling_results.items():
            summary = results['summary']
            
            val_auc = summary[summary['Metric'] == 'Validation AUC']['Mean'].values[0]
            test_auc = summary[summary['Metric'] == 'Test AUC (Optimal)']['Mean'].values[0]
            test_acc = summary[summary['Metric'] == 'Test Accuracy (Optimal)']['Mean'].values[0]
            test_f1 = summary[summary['Metric'] == 'Test F1 (Optimal)']['Mean'].values[0]
            
            comparison_data.append({
                'Pooling': pooling_name,
                'Features': feature_name,
                'Val_AUC': val_auc,
                'Test_AUC': test_auc,
                'Test_Acc': test_acc,
                'Test_F1': test_f1
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
    comparison_df.to_csv(results_dir / "comprehensive_comparison.csv", index=False)
    
    print("\nTop 10 Configurations:")
    print(comparison_df.head(10).to_string(index=False))
    
    # Best configuration
    best = comparison_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Pooling: {best['Pooling']}")
    print(f"   Features: {best['Features']}")
    print(f"   Test AUC: {best['Test_AUC']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pooling_methods = comparison_df['Pooling'].unique()
    feature_configs = comparison_df['Features'].unique()
    
    # Heatmap for Test AUC
    pivot_auc = comparison_df.pivot(index='Features', columns='Pooling', values='Test_AUC')
    im = axes[0, 0].imshow(pivot_auc.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    axes[0, 0].set_xticks(range(len(pooling_methods)))
    axes[0, 0].set_xticklabels(pooling_methods, rotation=45, ha='right')
    axes[0, 0].set_yticks(range(len(feature_configs)))
    axes[0, 0].set_yticklabels(feature_configs)
    axes[0, 0].set_title('Test AUC Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Add values to heatmap
    for i in range(len(feature_configs)):
        for j in range(len(pooling_methods)):
            text = axes[0, 0].text(j, i, f"{pivot_auc.values[i, j]:.3f}",
                                  ha="center", va="center", color="black", fontsize=9)
    
    # Bar plot: Best feature config per pooling
    for pooling in pooling_methods:
        pooling_data = comparison_df[comparison_df['Pooling'] == pooling]
        best_idx = pooling_data['Test_AUC'].idxmax()
        best_row = pooling_data.loc[best_idx]
        axes[0, 1].bar(pooling, best_row['Test_AUC'], alpha=0.7)
        axes[0, 1].text(pooling, best_row['Test_AUC'] + 0.01, best_row['Features'],
                       ha='center', fontsize=8, rotation=0)
    
    axes[0, 1].set_ylabel('Test AUC')
    axes[0, 1].set_title('Best Feature Config per Pooling Method', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0.5, 1.0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Line plot: Performance across feature configs
    for pooling in pooling_methods:
        pooling_data = comparison_df[comparison_df['Pooling'] == pooling].sort_values('Features')
        axes[1, 0].plot(pooling_data['Features'], pooling_data['Test_AUC'], 
                       marker='o', label=pooling, linewidth=2)
    
    axes[1, 0].set_xlabel('Feature Configuration')
    axes[1, 0].set_ylabel('Test AUC')
    axes[1, 0].set_title('Performance Across Feature Configurations', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Box plot: Distribution per pooling method
    pooling_data_list = [comparison_df[comparison_df['Pooling'] == p]['Test_AUC'].values 
                         for p in pooling_methods]
    axes[1, 1].boxplot(pooling_data_list, labels=pooling_methods)
    axes[1, 1].set_ylabel('Test AUC')
    axes[1, 1].set_title('Performance Distribution per Pooling Method', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(results_dir / "comprehensive_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: {results_dir / 'comprehensive_comparison.png'}")


def main():
    # Setup paths
    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_attention\attention_mil_results")
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
    
    # Load patient features
    patient_features_df = pd.read_csv(BASE_CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=BASE_CONFIG['train_csv_path'],
        patient_features_df=patient_features_df
    )
    
    # Load K-Fold splits
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    # Run comprehensive study
    # You can specify which configs to test
    pooling_to_test = [ 'gated_attention', 'multihead_attention_4']
    features_to_test = ['cnn_only', 'full']  # Start with these two
    
    all_results = run_comprehensive_ablation_study(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        base_config=BASE_CONFIG,
        results_base_dir=results_base_dir,
        pooling_configs_to_test=pooling_to_test,
        feature_configs_to_test=features_to_test
    )
    
    print("\n" + "="*70)
    print("✓ COMPREHENSIVE ABLATION STUDY COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()