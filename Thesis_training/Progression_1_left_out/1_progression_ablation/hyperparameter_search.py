# Hyperparameter Search for Ablation Study using Optuna
# Find optimal hyperparameters on "full" configuration (CNN + hand-crafted + demographics)
# Then use these parameters for all ablation configurations

from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
import numpy as np
import json
from typing import Dict, List
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader
from ablation_study import (
    create_feature_set_for_fold,
    load_and_merge_demographics,
    ABLATION_CONFIGS
)
from model_train import (train_single_fold, aggregate_fold_results,HAND_FEATURE_COLS,DEMO_FEATURE_COLS)

# Use only first 3 folds for speed during search
NUM_SEARCH_FOLDS = 3

# Global variables for Optuna objective function
GLOBAL_SLICE_FEATURES_DF = None
GLOBAL_PATIENT_FEATURES_DF = None
GLOBAL_KFOLD_SPLITS = None
GLOBAL_RESULTS_DIR = None

# Base configuration (fixed across all trials)
BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # Model parameters
    'backbone': 'resnet50',
    'image_size': (224, 224),
    
    # Training parameters (fixed - reduced for faster search)
    'epochs': 30,  # Reduced from 100 for faster hyperparameter search
    'early_stopping_patience': 8,  # Reduced from 20
    # 'label_smoothing': 0.2,  # Now tuned by Optuna
    'use_scheduler': True,
    'scheduler_patience': 3,  # Reduced from 5
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    'use_batch_norm': True,
    
    'resume_from_checkpoint': False,  # No checkpoint for hyperparam search
    'normalization_type': 'standard',
}


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function to maximize validation AUC
    """
    # Sample hyperparameters
    hidden_config = trial.suggest_categorical('hidden_config', [
        'small',      # [256, 128]
        'medium',     # [512, 256]
        'deep',       # [512, 256, 128]
        'very_deep'   # [256, 128, 64]
    ])
    
    # Map to actual hidden_dims
    hidden_dims_map = {
        'small': [256, 128],
        'medium': [512, 256],
        'deep': [512, 256, 128],
        'very_deep': [256, 128, 64]
    }
    hidden_dims = hidden_dims_map[hidden_config]
    
    dropout = trial.suggest_float('dropout', 0.4, 0.8, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3, step=0.05)
    
    config_params = {
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'label_smoothing': label_smoothing
    }
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: Testing configuration")
    print("="*70)
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    
    # Evaluate configuration
    result = evaluate_config(
        config_params=config_params,
        slice_features_df=GLOBAL_SLICE_FEATURES_DF,
        patient_features_df=GLOBAL_PATIENT_FEATURES_DF,
        kfold_splits=GLOBAL_KFOLD_SPLITS,
        results_dir=GLOBAL_RESULTS_DIR / f"trial_{trial.number:03d}"
    )
    
    # Return validation AUC to maximize
    val_auc = result['val_auc_mean']
    
    # Report intermediate value for pruning
    trial.report(val_auc, step=0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    print(f"\nTrial {trial.number} Result: Val AUC = {val_auc:.4f}")
    
    return val_auc


def evaluate_config(
    config_params: Dict,
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: Dict,
    results_dir: Path
) -> Dict:
    """
    Evaluate a single hyperparameter configuration
    """
    # Merge base config with search params
    config = BASE_CONFIG.copy()
    config.update(config_params)
    config['results_save_dir'] = results_dir
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use "full" ablation config (CNN + hand + demo)
    ablation_config = ABLATION_CONFIGS['full']
    
    fold_results = []
    
    # Use only first N folds for faster search
    fold_keys = sorted(list(kfold_splits.keys()))[:NUM_SEARCH_FOLDS]
    
    for fold_idx in fold_keys:
        fold_data = kfold_splits[fold_idx]
        
        # Create feature set for this fold
        features_df = create_feature_set_for_fold(
            slice_features_df=slice_features_df,
            patient_features_df=patient_features_df,
            fold_data=fold_data,
            ablation_config=ablation_config,
            normalization_type=config['normalization_type']
        )
        
        # Train
        result = train_single_fold(
            features_df=features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=config,
            results_dir=results_dir,
            resume_from_checkpoint=False
        )
        
        fold_results.append(result)
    
    # Compute average validation AUC
    val_aucs = [r['val_auc'] for r in fold_results]
    test_aucs_optimal = [r['test_metrics_optimal']['auc'] for r in fold_results]
    
    avg_val_auc = np.mean(val_aucs)
    std_val_auc = np.std(val_aucs)
    avg_test_auc = np.mean(test_aucs_optimal)
    std_test_auc = np.std(test_aucs_optimal)
    
    return {
        'config': config_params,
        'val_auc_mean': avg_val_auc,
        'val_auc_std': std_val_auc,
        'test_auc_mean': avg_test_auc,
        'test_auc_std': std_test_auc,
        'fold_results': fold_results
    }


def run_hyperparameter_search(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: Dict,
    base_results_dir: Path,
    n_trials: int = 50
):
    """
    Run Optuna hyperparameter search
    """
    global GLOBAL_SLICE_FEATURES_DF, GLOBAL_PATIENT_FEATURES_DF, GLOBAL_KFOLD_SPLITS, GLOBAL_RESULTS_DIR
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH WITH OPTUNA - FULL CONFIGURATION")
    print("="*80)
    print(f"Optimization algorithm: TPE (Tree-structured Parzen Estimator)")
    print(f"Number of trials: {n_trials}")
    print(f"Using {NUM_SEARCH_FOLDS} folds for evaluation")
    print(f"\nSearch space:")
    print(f"  hidden_dims: [256,128], [512,256], [512,256,128], [256,128,64]")
    print(f"  dropout: [0.4, 0.8]")
    print(f"  learning_rate: [1e-5, 1e-3] (log scale)")
    print(f"  weight_decay: [0.001, 0.2] (log scale)")
    print(f"  batch_size: [8, 16]")
    print(f"  label_smoothing: [0.0, 0.3]")
    
    # Create search directory
    search_dir = base_results_dir / "hyperparameter_search_optuna"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global variables for objective function
    GLOBAL_SLICE_FEATURES_DF = slice_features_df
    GLOBAL_PATIENT_FEATURES_DF = patient_features_df
    GLOBAL_KFOLD_SPLITS = kfold_splits
    GLOBAL_RESULTS_DIR = search_dir
    
    # Use SQLite storage for persistence and resume capability
    storage_path = search_dir / "optuna_study.db"
    storage_name = f"sqlite:///{storage_path}"
    study_name = 'ablation_hyperparam_search'
    
    # Try to load existing study or create new one
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n✓ Resumed existing study: {n_completed} trials already completed")
        print(f"  Continuing from trial {len(study.trials)}...")
    except KeyError:
        # Study doesn't exist yet, create new one
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='maximize',  # Maximize validation AUC
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            load_if_exists=True
        )
        print(f"\n✓ Created new study")
    
    # Define callback to save progress after each trial
    def save_progress_callback(study, trial):
        # Save intermediate results after each trial
        trials_df = study.trials_dataframe()
        trials_df.to_csv(search_dir / "optuna_trials_progress.csv", index=False)
        
        # Save best config so far
        if study.best_trial.number == trial.number:
            hidden_dims_map = {
                'small': [256, 128],
                'medium': [512, 256],
                'deep': [512, 256, 128],
                'very_deep': [256, 128, 64]
            }
            best_config = {
                'hidden_dims': hidden_dims_map[study.best_trial.params['hidden_config']],
                'dropout': study.best_trial.params['dropout'],
                'learning_rate': study.best_trial.params['learning_rate'],
                'weight_decay': study.best_trial.params['weight_decay'],
                'batch_size': study.best_trial.params['batch_size'],
                'label_smoothing': study.best_trial.params['label_smoothing']
            }
            with open(search_dir / "best_config_current.json", 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"\n  ✓ New best trial {trial.number}: Val AUC = {trial.value:.4f}")
    
    # Run optimization
    print("\n" + "="*80)
    print("STARTING OPTUNA OPTIMIZATION")
    print("="*80)
    print(f"Total trials to run: {n_trials}")
    print(f"Storage: {storage_path}")
    print(f"\nYou can stop and resume anytime - progress is automatically saved!\n")
    
    study.optimize(objective, n_trials=n_trials, callbacks=[save_progress_callback], show_progress_bar=True)
    
    # Get best trial
    best_trial = study.best_trial
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best validation AUC: {best_trial.value:.4f}")
    
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Convert to actual config
    hidden_dims_map = {
        'small': [256, 128],
        'medium': [512, 256],
        'deep': [512, 256, 128],
        'very_deep': [256, 128, 64]
    }
    
    best_config = {
        'hidden_dims': hidden_dims_map[best_trial.params['hidden_config']],
        'dropout': best_trial.params['dropout'],
        'learning_rate': best_trial.params['learning_rate'],
        'weight_decay': best_trial.params['weight_decay'],
        'batch_size': best_trial.params['batch_size'],
        'label_smoothing': best_trial.params['label_smoothing']
    }
    
    # Save best config
    with open(search_dir / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(search_dir / "optuna_trials.csv", index=False)
    
    # Save study object
    joblib.dump(study, search_dir / "optuna_study.pkl")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(search_dir / "optimization_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(search_dir / "param_importances.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Plots saved: optimization_history.png, param_importances.png")
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
    
    # Show top 10 trials
    print(f"\n{'='*70}")
    print("TOP 10 TRIALS:")
    print("="*70)
    top_trials = trials_df.nlargest(10, 'value')[['number', 'value', 'params_hidden_config', 
                                                     'params_dropout', 'params_learning_rate', 
                                                     'params_weight_decay', 'params_batch_size']]
    print(top_trials.to_string(index=False))
    
    print(f"\n✓ Results saved to: {search_dir}")
    print(f"✓ Best config saved to: {search_dir / 'best_config.json'}")
    print(f"✓ Optuna study saved to: {search_dir / 'optuna_study.pkl'}")
    
    return best_config, study


def main():
    """Main execution function"""
    
    # Configure paths
    base_dir = Path(r"d:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression_ablation")
    
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER SEARCH FOR ABLATION STUDY")
    print("="*80)
    print(f"Configuration: FULL (CNN + Hand-crafted + Demographics)")
    print(f"Folds used for evaluation: {NUM_SEARCH_FOLDS}")
    print(f"Optimization: TPE algorithm with adaptive pruning")
    
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
    
    # Create results directory
    results_base_dir = base_dir / "hyperparam_search_optuna"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Optuna hyperparameter search
    best_config, study = run_hyperparameter_search(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        base_results_dir=results_base_dir,
        n_trials=30  # Reduced from 50 for faster completion
    )
    
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"\nBest trial value: {study.best_value:.4f}")
    print(f"Total trials completed: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    
    print(f"\nNext steps:")
    print(f"1. Review results in: {results_base_dir}")
    print(f"2. Check optimization plots: optimization_history.png, param_importances.png")
    print(f"3. Update BASE_CONFIG in ablation_study.py with best parameters")
    print(f"4. Run full ablation study with optimized hyperparameters")
    
    print(f"\n{'='*70}")
    print("BEST HYPERPARAMETERS TO USE IN ABLATION_STUDY.PY:")
    print("="*70)
    print(f"  'hidden_dims': {best_config['hidden_dims']},")
    print(f"  'dropout': {best_config['dropout']:.1f},")
    print(f"  'learning_rate': {best_config['learning_rate']:.2e},")
    print(f"  'label_smoothing': {best_config['label_smoothing']:.2f},")
    print(f"  'weight_decay': {best_config['weight_decay']:.4f},")
    print(f"  'batch_size': {best_config['batch_size']},")


if __name__ == "__main__":
    main()

