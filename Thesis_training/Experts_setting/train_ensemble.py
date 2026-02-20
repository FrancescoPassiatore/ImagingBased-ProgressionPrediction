"""
Main Training Script for Expert Ensemble System
Trains CNN + LightGBM with Meta-Model Fusion using 5-Fold Cross-Validation
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import pickle

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / '1_progression_maxpooling'))

# Import utilities
from utilities import (
    IPFDataLoader,
    CNNFeatureExtractor,
    create_dataloaders,
    compute_class_weights,
    HAND_FEATURE_ORDER,
    DEMOGRAPHIC_FEATURES
)

# Import expert models
from cnn_expert import CNNExpert, CNNExpertTrainer
from lgbm_expert import LightGBMExpert, LightGBMExpertTrainer
from meta_model import (
    MetaModel,
    compute_correlation,
    find_optimal_threshold,
    compute_metrics,
    print_metrics,
    plot_roc_curves,
    plot_prediction_distribution
)
from ensemble_utils import (
    prepare_lgbm_features,
    normalize_features_per_fold,
    aggregate_results,
    save_fold_results,
    print_fold_summary
)


def train_cnn_expert(
    train_loader,
    val_loader,
    test_loader,
    cnn_feature_dim: int,
    hand_feature_dim: int,
    demo_feature_dim: int,
    class_weights: torch.Tensor,
    device: str,
    num_epochs: int = 50,
    early_stopping_patience: int = 10,
    fold_idx: int = 0,
    output_dir: Path = None,
    pooling_type: str = 'mean'
) -> Dict:
    """
    Train CNN expert model
    
    Returns:
        results: Dict with predictions and metrics
    """
    print("\n" + "="*80)
    print(f"TRAINING CNN EXPERT - FOLD {fold_idx}")
    print("="*80)
    
    # Initialize model
    print(f"  Pooling type: {pooling_type}")
    print(f"  CNN feature dim: {cnn_feature_dim}")
    print(f"  Hand feature dim: {hand_feature_dim}")
    print(f"  Demo feature dim: {demo_feature_dim}")
    model = CNNExpert(
        cnn_feature_dim=cnn_feature_dim,
        hand_feature_dim=hand_feature_dim,
        demo_feature_dim=demo_feature_dim,
        hidden_dim=256,
        dropout=0.3,
        pooling_type=pooling_type
    )
    
    trainer = CNNExpertTrainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=class_weights
    )
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_loss, y_val_true, y_val_pred = trainer.evaluate(val_loader)
        
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y_val_true, y_val_pred)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            if output_dir:
                trainer.save_model(output_dir / f'cnn_fold{fold_idx}_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if output_dir:
        trainer.load_model(output_dir / f'cnn_fold{fold_idx}_best.pth')
    
    # Get predictions
    y_val_true, p_cnn_val, val_patient_ids = trainer.predict(val_loader)
    y_test_true, p_cnn_test, test_patient_ids = trainer.predict(test_loader)
    
    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    
    return {
        'p_cnn_val': p_cnn_val,
        'p_cnn_test': p_cnn_test,
        'y_val_true': y_val_true,
        'y_test_true': y_test_true,
        'val_patient_ids': val_patient_ids,
        'test_patient_ids': test_patient_ids,
        'best_val_auc': best_val_auc
    }


def train_lgbm_expert(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_weights: np.ndarray,
    fold_idx: int = 0,
    output_dir: Path = None
) -> Dict:
    """
    Train LightGBM expert model
    
    Returns:
        results: Dict with predictions and metrics
    """
    print("\n" + "="*80)
    print(f"TRAINING LIGHTGBM EXPERT - FOLD {fold_idx}")
    print("="*80)
    
    # Initialize model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42
    }
    
    model = LightGBMExpert(params=params, class_weights=class_weights)
    
    # Train
    model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_names,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Save model
    if output_dir:
        model.save_model(output_dir / f'lgbm_fold{fold_idx}.txt')
    
    # Get predictions
    p_lgbm_val = model.predict_proba(X_val)
    p_lgbm_test = model.predict_proba(X_test)
    
    # Print feature importance
    print("\nTop 10 Feature Importances:")
    importance_df = model.get_feature_importance(importance_type='gain')
    print(importance_df.head(10))
    
    return {
        'p_lgbm_val': p_lgbm_val,
        'p_lgbm_test': p_lgbm_test,
        'y_val_true': y_val,
        'y_test_true': y_test,
        'feature_importance': importance_df
    }


def train_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    features_df: pd.DataFrame,
    features_data: Dict,
    device: str,
    output_dir: Path,
    config: Dict
) -> Dict:
    """
    Train all experts and meta-model for one fold
    
    Returns:
        fold_results: Dict with all predictions, metrics, and models
    """
    print("\n" + "="*80)
    print(f"FOLD {fold_idx + 1} / {config['n_folds']}")
    print("="*80)
    print(f"Train: {len(train_ids)} patients")
    print(f"Val:   {len(val_ids)} patients")
    print(f"Test:  {len(test_ids)} patients")
    
    # =========================================================================
    # 1. Prepare Data
    # =========================================================================
    
    # Normalize features per fold (to avoid data leakage)
    features_df_normalized, encoding_info = normalize_features_per_fold(
        features_df,
        features_data,
        train_ids,
        val_ids,
        test_ids
    )
    
    # Compute class weights
    class_weights_torch = compute_class_weights(features_df_normalized, train_ids)
    class_weights_np = class_weights_torch.numpy()
    
    # Create dataloaders for CNN (imaging features only)
    cnn_hand_feature_cols = config.get('cnn_hand_feature_cols', [])
    cnn_demo_feature_cols = config.get('cnn_demo_feature_cols', [])
    
    train_loader, val_loader, test_loader = create_dataloaders(
        features_df_normalized,
        train_ids,
        val_ids,
        test_ids,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        hand_feature_cols=cnn_hand_feature_cols,
        demo_feature_cols=cnn_demo_feature_cols,
        encoding_info=encoding_info
    )
    
    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    cnn_feature_dim = sample_batch['cnn_features'].shape[-1]
    hand_feature_dim = sample_batch['hand_features'].shape[1] if sample_batch['hand_features'] is not None else 0
    demo_feature_dim = sample_batch['demo_features'].shape[1] if sample_batch['demo_features'] is not None else 0
    
    print(f"\nFeature Dimensions:")
    print(f"  CNN: {cnn_feature_dim}")
    print(f"  Hand-crafted: {hand_feature_dim}")
    print(f"  Demographics: {demo_feature_dim}")
    
    # Prepare LightGBM features (hand-crafted + demographics)
    lgbm_hand_feature_cols = config.get('lgbm_hand_feature_cols', [])
    lgbm_demo_feature_cols = config.get('lgbm_demo_feature_cols', [])
    
    X_train_lgbm, y_train_lgbm, feature_names = prepare_lgbm_features(
        features_df_normalized, train_ids, lgbm_hand_feature_cols, lgbm_demo_feature_cols, encoding_info
    )
    X_val_lgbm, y_val_lgbm, _ = prepare_lgbm_features(
        features_df_normalized, val_ids, lgbm_hand_feature_cols, lgbm_demo_feature_cols, encoding_info
    )
    X_test_lgbm, y_test_lgbm, _ = prepare_lgbm_features(
        features_df_normalized, test_ids, lgbm_hand_feature_cols, lgbm_demo_feature_cols, encoding_info
    )
    
    # =========================================================================
    # 2. Train CNN Expert
    # =========================================================================
    
    cnn_results = train_cnn_expert(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cnn_feature_dim=cnn_feature_dim,
        hand_feature_dim=hand_feature_dim,
        demo_feature_dim=demo_feature_dim,
        class_weights=class_weights_torch,
        device=device,
        num_epochs=config['cnn_epochs'],
        early_stopping_patience=config['cnn_patience'],
        fold_idx=fold_idx,
        output_dir=output_dir,
        pooling_type=config.get('pooling_type', 'mean')
    )
    
    # =========================================================================
    # 3. Train LightGBM Expert
    # =========================================================================
    
    lgbm_results = train_lgbm_expert(
        X_train=X_train_lgbm,
        y_train=y_train_lgbm,
        X_val=X_val_lgbm,
        y_val=y_val_lgbm,
        X_test=X_test_lgbm,
        y_test=y_test_lgbm,
        feature_names=feature_names,
        class_weights=class_weights_np,
        fold_idx=fold_idx,
        output_dir=output_dir
    )
    
    # =========================================================================
    # 4. Compute Correlation Between Experts
    # =========================================================================
    
    correlation = compute_correlation(
        cnn_results['p_cnn_val'],
        lgbm_results['p_lgbm_val']
    )
    
    # =========================================================================
    # 5. Train Meta-Model on Validation Set
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"TRAINING META-MODEL - FOLD {fold_idx}")
    print("="*80)
    
    meta_model = MetaModel(fusion_type=config.get('fusion_type', 'weighted'))
    meta_model.fit(
        p_cnn=cnn_results['p_cnn_val'],
        p_lgbm=lgbm_results['p_lgbm_val'],
        y_true=cnn_results['y_val_true']
    )
    
    # =========================================================================
    # 6. Find Optimal Threshold on Validation Set
    # =========================================================================
    
    p_fused_val = meta_model.predict_proba(
        cnn_results['p_cnn_val'],
        lgbm_results['p_lgbm_val']
    )
    
    threshold, threshold_metrics = find_optimal_threshold(
        y_true=cnn_results['y_val_true'],
        y_pred_proba=p_fused_val,
        strategy=config.get('threshold_strategy', 'youden')
    )
    
    # =========================================================================
    # 7. Evaluate on Test Set
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"TEST SET EVALUATION - FOLD {fold_idx}")
    print("="*80)
    
    # Get fused predictions
    p_fused_test = meta_model.predict_proba(
        cnn_results['p_cnn_test'],
        lgbm_results['p_lgbm_test']
    )
    
    # Compute metrics for all models
    metrics_cnn = compute_metrics(cnn_results['y_test_true'], cnn_results['p_cnn_test'], threshold=threshold)
    metrics_lgbm = compute_metrics(lgbm_results['y_test_true'], lgbm_results['p_lgbm_test'], threshold=threshold)
    metrics_fused = compute_metrics(cnn_results['y_test_true'], p_fused_test, threshold=threshold)
    
    # Print results
    print_metrics(metrics_cnn, "CNN Expert")
    print_metrics(metrics_lgbm, "LightGBM Expert")
    print_metrics(metrics_fused, "Fused Model")
    
    # Plot ROC curves
    plot_roc_curves(
        y_true=cnn_results['y_test_true'],
        p_cnn=cnn_results['p_cnn_test'],
        p_lgbm=lgbm_results['p_lgbm_test'],
        p_fused=p_fused_test,
        save_path=output_dir / f'roc_curve_fold{fold_idx}.png'
    )
    
    # Plot prediction distributions
    plot_prediction_distribution(
        p_cnn=cnn_results['p_cnn_test'],
        p_lgbm=lgbm_results['p_lgbm_test'],
        p_fused=p_fused_test,
        y_true=cnn_results['y_test_true'],
        save_path=output_dir / f'prediction_dist_fold{fold_idx}.png'
    )
    
    # =========================================================================
    # 8. Compile Results
    # =========================================================================
    
    fold_results = {
        'fold_idx': fold_idx,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'correlation': correlation,
        'threshold': threshold,
        'threshold_metrics': threshold_metrics,
        'optimal_weight': meta_model.optimal_weight if hasattr(meta_model, 'optimal_weight') else None,
        'cnn_metrics': metrics_cnn,
        'lgbm_metrics': metrics_lgbm,
        'fused_metrics': metrics_fused,
        'test_predictions': {
            'patient_ids': cnn_results['test_patient_ids'],
            'y_true': cnn_results['y_test_true'],
            'p_cnn': cnn_results['p_cnn_test'],
            'p_lgbm': lgbm_results['p_lgbm_test'],
            'p_fused': p_fused_test
        }
    }
    
    # Save fold results
    save_fold_results(fold_results, output_dir / f'fold{fold_idx}_results.json')
    
    return fold_results


def main():
    """Main training pipeline"""
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    config = {
        # Data paths
        'ground_truth_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv',
        'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv',
        'demographics_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\train.csv',
        'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset',
        'cnn_features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_maxpooling\slice_features.csv',
        'kfold_splits_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl',
        
        # Output
        'output_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Experts_setting\results_max',
        
        # Cross-validation
        'n_folds': 5,
        'random_seed': 42,
        
        # Features for CNN (imaging only)
        'cnn_hand_feature_cols': [],  # CNN uses only imaging features
        'cnn_demo_feature_cols': [],  # CNN uses only imaging features
        
        # Features for LightGBM (hand-crafted + demographics)
        'lgbm_hand_feature_cols': [
            'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
            'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60',
            'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
        ],
        'lgbm_demo_feature_cols': ['Age', 'Sex', 'SmokingStatus'],
        
        # CNN training
        'cnn_epochs': 50,
        'cnn_patience': 10,
        'batch_size': 8,
        'num_workers': 4,
        'pooling_type': 'max',  # 'mean', 'max', or 'max_mean'
        
        # Threshold
        'threshold_strategy': 'youden',  # 'youden', 'f1', or 'precision_recall'
        
        # Fusion
        'fusion_type': 'weighted',  # 'weighted' or 'logistic'
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERT ENSEMBLE TRAINING")
    print("="*80)
    print(f"Device: {config['device']}")
    print(f"Output: {output_dir}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load patient data and features
    data_loader = IPFDataLoader(
        ground_truth_path=config['ground_truth_path'],
        features_path=config['features_path'],
        npy_dir=config['npy_dir'],
        demographics_path=config['demographics_path']
    )
    patient_data, features_data = data_loader.get_patient_data()
    
    # Load CNN features
    print(f"\nLoading CNN features from: {config['cnn_features_path']}")
    features_df = pd.read_csv(config['cnn_features_path'])
    print(f"Loaded {len(features_df)} slice features")
    
    # Get patient IDs and labels
    patient_ids = list(patient_data.keys())
    labels = np.array([patient_data[pid]['gt_has_progressed'] for pid in patient_ids])
    
    print(f"\nTotal patients: {len(patient_ids)}")
    print(f"Progression: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"No progression: {(1-labels).sum()} ({(1-labels).sum()/len(labels)*100:.1f}%)")
    
    # =========================================================================
    # K-Fold Cross-Validation (Load Pre-defined Splits)
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"LOADING PRE-DEFINED {config['n_folds']}-FOLD SPLITS")
    print("="*80)
    
    # Load pre-defined splits
    print(f"Loading splits from: {config['kfold_splits_path']}")
    with open(config['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"Loaded splits for {len(kfold_splits)} folds")
    
    all_fold_results = []
    
    for fold_idx in range(config['n_folds']):
        # Get pre-defined splits for this fold
        fold_split = kfold_splits[fold_idx]
        train_ids = fold_split['train']
        val_ids = fold_split['val']
        test_ids = fold_split['test']
        
        # Filter to only include patients that exist in our data
        train_ids = [pid for pid in train_ids if pid in patient_data]
        val_ids = [pid for pid in val_ids if pid in patient_data]
        test_ids = [pid for pid in test_ids if pid in patient_data]
        
        # Train fold
        fold_results = train_fold(
            fold_idx=fold_idx,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            features_df=features_df,
            features_data=features_data,
            device=config['device'],
            output_dir=output_dir,
            config=config
        )
        
        all_fold_results.append(fold_results)
        
        # Print fold summary
        print_fold_summary(fold_results, fold_idx)
    
    # =========================================================================
    # Aggregate Results Across Folds
    # =========================================================================
    
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS FOLDS")
    print("="*80)
    
    summary = aggregate_results(all_fold_results)
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS (Mean ± Std)")
    print("="*80)
    
    for model_name in ['cnn', 'lgbm', 'fused']:
        print(f"\n{model_name.upper()} Expert:")
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']:
            mean = summary[f'{model_name}_{metric}_mean']
            std = summary[f'{model_name}_{metric}_std']
            print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    print(f"\nOptimal Threshold: {summary['threshold_mean']:.4f} ± {summary['threshold_std']:.4f}")
    
    if 'optimal_weight_mean' in summary:
        print(f"\nWeighted Fusion:")
        print(f"  CNN weight: {summary['optimal_weight_mean']:.4f} ± {summary['optimal_weight_std']:.4f}")
        print(f"  LGBM weight: {1 - summary['optimal_weight_mean']:.4f} ± {summary['optimal_weight_std']:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
