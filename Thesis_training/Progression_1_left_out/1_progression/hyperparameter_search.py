"""
Hyperparameter optimization for progression prediction model
Focuses on combating overfitting while improving AUC
"""
from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
from itertools import product
import json

sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader
from model_train import train_single_fold

# Base paths (same as main training)
BASE_CONFIG = {
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl"),
    "results_base_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\hyperparam_search"),
    "backbone": 'resnet50',
    "image_size": (224, 224),
    "feature_dim": 2048,
    "batch_size": 16,
    "epochs": 100,
    "resume_from_checkpoint": False  # Set False for clean search
}

# Hyperparameter search space - FOCUSED ON OVERFITTING SOLUTIONS
SEARCH_SPACE = {
    # Architecture (avoid too narrow bottlenecks)
    'hidden_dims': [
        [512, 256],      # Wider
        [256, 128],      # Medium
        [128, 64],       # Narrower
        [256],           # Single layer - wider
        [128],           # Single layer - medium
        [64, 32],        # Narrower single layer
        [64],            # Very narrow single layer
        [32]             # Extremely narrow single layer
    ],
    
    # Note: Aggregation removed - model now always uses max pooling on per-slice predictions
    
    # Dropout (key for overfitting)
    'dropout': [0.3, 0.5, 0.7],
    
    # Learning rate (escape local minima)
    'learning_rate': [1e-4, 5e-4, 1e-3],
    
    # Weight decay (L2 regularization)
    'weight_decay': [0.001, 0.01, 0.05],
    
    # Label smoothing (prevent overconfidence)
    'label_smoothing': [0.0, 0.1, 0.2],
    
    # Batch size (smaller = more regularization, better generalization)
    'batch_size': [8, 16, 32],
    
    # Batch norm
    'use_batch_norm': [True, False],
    
    # Early stopping (prevent overtraining)
    'early_stopping_patience': [10, 15, 20],
}


def generate_random_configs(n_configs=20):
    """Generate random hyperparameter configurations"""
    import random
    
    configs = []
    for i in range(n_configs):
        config = {
            'config_id': i,
            'hidden_dims': random.choice(SEARCH_SPACE['hidden_dims']),
            'dropout': random.choice(SEARCH_SPACE['dropout']),
            'learning_rate': random.choice(SEARCH_SPACE['learning_rate']),
            'weight_decay': random.choice(SEARCH_SPACE['weight_decay']),
            'label_smoothing': random.choice(SEARCH_SPACE['label_smoothing']),
            'batch_size': random.choice(SEARCH_SPACE['batch_size']),
            'use_batch_norm': random.choice(SEARCH_SPACE['use_batch_norm']),
            'early_stopping_patience': random.choice(SEARCH_SPACE['early_stopping_patience']),
        }
        configs.append(config)
    
    return configs


def generate_focused_configs():
    """Generate focused configurations based on overfitting insights"""
    configs = [
        # Config 1: High regularization + small batch
        {
            'config_id': 'high_reg',
            'hidden_dims': [256, 128],
            'dropout': 0.7,
            'learning_rate': 5e-4,
            'weight_decay': 0.05,
            'label_smoothing': 0.2,
            'batch_size': 8,
            'use_batch_norm': True,
            'early_stopping_patience': 10,
        },
        # Config 2: Medium regularization, wider network, medium batch
        {
            'config_id': 'med_reg_wide',
            'hidden_dims': [512, 256],
            'dropout': 0.5,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'batch_size': 16,
            'use_batch_norm': True,
            'early_stopping_patience': 15,
        },
        # Config 3: Single layer, high dropout, small batch
        {
            'config_id': 'single_layer_high_drop',
            'hidden_dims': [256],
            'dropout': 0.6,
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'batch_size': 8,
            'use_batch_norm': True,
            'early_stopping_patience': 10,
        },
        # Config 4: Aggressive learning, strong regularization, large batch
        {
            'config_id': 'aggressive_learn',
            'hidden_dims': [256, 128],
            'dropout': 0.5,
            'learning_rate': 1e-3,
            'weight_decay': 0.05,
            'label_smoothing': 0.1,
            'batch_size': 32,
            'use_batch_norm': False,
            'early_stopping_patience': 10,
        },
        # Config 5: Conservative (your improved settings), medium batch
        {
            'config_id': 'conservative',
            'hidden_dims': [256, 128],
            'dropout': 0.5,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'batch_size': 16,
            'use_batch_norm': True,
            'early_stopping_patience': 10,
        },
    ]
    
    return configs


def run_hyperparameter_search(slice_features_df, kfold_splits, search_type='focused'):
    """
    Run hyperparameter search
    
    Args:
        slice_features_df: Extracted CNN features
        kfold_splits: K-fold split data
        search_type: 'focused' (5 configs), 'random_20' (20 random), 'random_50' (50 random)
    """
    # Generate configurations
    if search_type == 'focused':
        configs = generate_focused_configs()
        print(f"\n{'='*70}")
        print(f"FOCUSED HYPERPARAMETER SEARCH - {len(configs)} configurations")
        print(f"{'='*70}")
    elif search_type == 'random_20':
        configs = generate_random_configs(n_configs=20)
        print(f"\n{'='*70}")
        print(f"RANDOM HYPERPARAMETER SEARCH - 20 configurations")
        print(f"{'='*70}")
    elif search_type == 'random_50':
        configs = generate_random_configs(n_configs=50)
        print(f"\n{'='*70}")
        print(f"RANDOM HYPERPARAMETER SEARCH - 50 configurations")
        print(f"{'='*70}")
    else:
        raise ValueError(f"Unknown search_type: {search_type}")
    
    # Create results directory
    BASE_CONFIG['results_base_dir'].mkdir(parents=True, exist_ok=True)
    
    # Track results
    all_results = []
    
    # Run each configuration on first fold only (for speed)
    fold_key = sorted(kfold_splits.keys())[0]
    fold_data = kfold_splits[fold_key]
    fold_idx = 0
    
    print(f"\nTesting all configs on Fold 0 (for computational efficiency)")
    print(f"Train: {len(fold_data['train'])} | Val: {len(fold_data['val'])} | Test: {len(fold_data['test'])}")
    
    for i, hp_config in enumerate(configs):
        config_id = hp_config['config_id']
        print(f"\n{'='*70}")
        print(f"CONFIG {i+1}/{len(configs)} - ID: {config_id}")
        print(f"{'='*70}")
        
        # Print hyperparameters
        print("\nHyperparameters:")
        for key, value in hp_config.items():
            if key != 'config_id':
                print(f"  {key}: {value}")
        
        # Merge with base config
        full_config = {**BASE_CONFIG, **hp_config}
        full_config['use_scheduler'] = True
        full_config['scheduler_patience'] = 5
        full_config['scheduler_factor'] = 0.5
        full_config['scheduler_min_lr'] = 1e-6
        
        # Create results directory for this config
        config_dir = BASE_CONFIG['results_base_dir'] / f"config_{config_id}"
        
        try:
            # Train on single fold
            result = train_single_fold(
                features_df=slice_features_df,
                fold_data=fold_data,
                fold_idx=fold_idx,
                config=full_config,
                results_dir=config_dir,
                resume_from_checkpoint=False
            )
            
            # Store results
            result_summary = {
                'config_id': config_id,
                **hp_config,
                'val_auc': result['val_auc'],
                'test_auc_default': result['test_metrics_default']['auc'],
                'test_auc_optimal': result['test_metrics_optimal']['auc'],
                'test_acc_default': result['test_metrics_default']['accuracy'],
                'test_acc_optimal': result['test_metrics_optimal']['accuracy'],
                'test_f1_default': result['test_metrics_default']['f1'],
                'test_f1_optimal': result['test_metrics_optimal']['f1'],
                'optimal_threshold': result['optimal_threshold'],
                'status': 'success'
            }
            
            print(f"\n✓ Config {config_id} completed:")
            print(f"  Val AUC: {result['val_auc']:.4f}")
            print(f"  Test AUC (default): {result['test_metrics_default']['auc']:.4f}")
            print(f"  Test AUC (optimal): {result['test_metrics_optimal']['auc']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Config {config_id} FAILED: {e}")
            result_summary = {
                'config_id': config_id,
                **hp_config,
                'status': 'failed',
                'error': str(e)
            }
        
        all_results.append(result_summary)
        
        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(BASE_CONFIG['results_base_dir'] / "search_results_intermediate.csv", index=False)
    
    # Final results analysis
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")
    
    # Filter successful runs
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        print("\n❌ No successful configurations!")
        return all_results
    
    results_df = pd.DataFrame(successful_results)
    
    # Sort by validation AUC
    results_df = results_df.sort_values('val_auc', ascending=False)
    
    # Save final results
    results_df.to_csv(BASE_CONFIG['results_base_dir'] / "search_results_final.csv", index=False)
    
    # Print top 5 configurations
    print("\n" + "="*70)
    print("TOP 5 CONFIGURATIONS (by Validation AUC)")
    print("="*70)
    
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(f"\nRank {idx+1}: Config {row['config_id']}")
        print(f"  Val AUC: {row['val_auc']:.4f}")
        print(f"  Test AUC (optimal): {row['test_auc_optimal']:.4f}")
        print(f"  Hidden dims: {row['hidden_dims']}")
        print(f"  Dropout: {row['dropout']}")
        print(f"  Learning rate: {row['learning_rate']}")
        print(f"  Weight decay: {row['weight_decay']}")
        print(f"  Label smoothing: {row['label_smoothing']}")
    
    # Analyze overfitting (val vs test gap)
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS (Val-Test Gap)")
    print("="*70)
    
    results_df['overfit_gap'] = results_df['val_auc'] - results_df['test_auc_optimal']
    results_df_by_overfit = results_df.sort_values('overfit_gap', ascending=True)
    
    print("\nTop 3 configs with LEAST overfitting:")
    for idx, (_, row) in enumerate(results_df_by_overfit.head(3).iterrows()):
        print(f"\n{idx+1}. Config {row['config_id']}")
        print(f"   Val AUC: {row['val_auc']:.4f} | Test AUC: {row['test_auc_optimal']:.4f} | Gap: {row['overfit_gap']:.4f}")
        print(f"   Dropout: {row['dropout']} | Weight decay: {row['weight_decay']} | Label smooth: {row['label_smoothing']}")
    
    # Best overall (balancing val AUC and low overfitting)
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)
    
    # Score: 0.7 * val_auc + 0.3 * (1 - overfit_gap)
    results_df['combined_score'] = 0.7 * results_df['val_auc'] + 0.3 * (1 - results_df['overfit_gap'])
    best_config = results_df.loc[results_df['combined_score'].idxmax()]
    
    print(f"\nBest Config: {best_config['config_id']}")
    print(f"  Val AUC: {best_config['val_auc']:.4f}")
    print(f"  Test AUC: {best_config['test_auc_optimal']:.4f}")
    print(f"  Overfit gap: {best_config['overfit_gap']:.4f}")
    print(f"  Combined score: {best_config['combined_score']:.4f}")
    print("\nHyperparameters:")
    print(f"  hidden_dims: {best_config['hidden_dims']}")
    print(f"  dropout: {best_config['dropout']}")
    print(f"  learning_rate: {best_config['learning_rate']}")
    print(f"  weight_decay: {best_config['weight_decay']}")
    print(f"  label_smoothing: {best_config['label_smoothing']}")
    print(f"  use_batch_norm: {best_config['use_batch_norm']}")
    print(f"  early_stopping_patience: {best_config['early_stopping_patience']}")
    print("  Note: Model uses per-slice prediction with max aggregation")
    
    # Save best config as JSON
    best_config_dict = {
        'hidden_dims': best_config['hidden_dims'],
        'dropout': float(best_config['dropout']),
        'learning_rate': float(best_config['learning_rate']),
        'weight_decay': float(best_config['weight_decay']),
        'label_smoothing': float(best_config['label_smoothing']),
        'use_batch_norm': bool(best_config['use_batch_norm']),
        'early_stopping_patience': int(best_config['early_stopping_patience']),
        'note': 'Model uses per-slice prediction with max aggregation'
    }
    
    with open(BASE_CONFIG['results_base_dir'] / "best_config.json", 'w') as f:
        json.dump(best_config_dict, f, indent=2)
    
    print(f"\n✓ Best config saved to: {BASE_CONFIG['results_base_dir'] / 'best_config.json'}")
    print(f"✓ All results saved to: {BASE_CONFIG['results_base_dir'] / 'search_results_final.csv'}")
    
    return all_results


def main():
    """Main hyperparameter search pipeline"""
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path']
    )
    
    patient_data, features_data = data_loader.get_patient_data()
    print(f"Loaded data for {len(patient_data)} patients.")
    
    # Step 2: Extract CNN features
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CNN FEATURES")
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
    
    # Step 3: Load K-Fold splits
    print("\n" + "="*70)
    print("STEP 3: LOADING K-FOLD SPLITS")
    print("="*70)
    
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"Loaded {len(kfold_splits)} folds")
    
    # Step 4: Run hyperparameter search
    print("\n" + "="*70)
    print("STEP 4: HYPERPARAMETER SEARCH")
    print("="*70)
    
    # Choose search type: 'focused', 'random_20', or 'random_50'
    search_type = 'focused'  # Start with focused search
    
    results = run_hyperparameter_search(
        slice_features_df=slice_features_df,
        kfold_splits=kfold_splits,
        search_type=search_type
    )
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
