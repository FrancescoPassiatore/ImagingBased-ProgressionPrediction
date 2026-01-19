"""
MLP Corrector Hyperparameter Optimization with Optuna
======================================================

Optimizes MLP corrector hyperparameters for different feature combinations:
- demographics_only: CNN + age, sex, smoking
- handcrafted_only: CNN + 9 handcrafted features
- full: CNN + demographics + handcrafted (13 features total)

The corrector architecture and regularization auto-adapt based on input size.

Hyperparameters to optimize:
- Hidden layer sizes (adaptive to input size)
- Dropout rates (higher for more features)
- Learning rate
- Weight decay
- Batch size
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.trial import Trial
import yaml
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader,
    create_kfold_splits,
    CorrectorDataset,
    FeatureNormalizer,
    compute_metrics
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    'cnn_predictions_path': Path('Training_2/CNN_Training/predictions_densenet_strat/cnn_predictions_fold0.pkl'),  # From best CNN
    
    # Optuna
    'n_trials': 50,  # Fewer trials than CNN (corrector is simpler)
    'fold_for_optimization': 0,
    
    # Feature types to optimize
    'feature_types': ['demographics', 'handcrafted', 'full'],
    
    # Fixed parameters
    'n_epochs': 100,
    'patience': 15,
    'normalize_features': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Output
    'results_dir': Path('Training_2/MLP_Corrector/optuna_results'),
    'best_params_dir': Path('Training_2/MLP_Corrector/best_params')
}

CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['best_params_dir'].mkdir(parents=True, exist_ok=True)

# =============================================================================
# ADAPTIVE MLP CORRECTOR
# =============================================================================

class AdaptiveSlopeCorrector(nn.Module):
    """
    MLP corrector that adapts architecture based on input size
    More features → deeper network, more regularization
    """
    
    def __init__(self, input_dim: int, hidden_sizes: list, dropout_rates: list, use_batch_norm: bool=True):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_size = input_dim
        
        for i, (hidden_size, dropout) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) where first feature is always CNN slope
        Returns:
            final_slope: (batch, 1) = CNN slope + correction
        """
        slope_cnn = x[:, 0:1]  # First feature is CNN slope
        correction = self.network(x)
        final_slope = slope_cnn + correction
        return final_slope.squeeze(-1)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, gradient_clip=1.0, l1_lambda=0.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        slopes = batch['slope'].to(device)
        
        # Forward
        predictions = model(features)
        
        # MSE loss
        mse_loss = criterion(predictions, slopes)
        
        # L1 regularization on first layer (features)
        l1_loss = 0.0
        if l1_lambda > 0:
            first_layer = model.network[0]  # First Linear layer
            l1_loss = l1_lambda * torch.abs(first_layer.weight).sum()
        
        loss = mse_loss + l1_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * len(slopes)
        n_samples += len(slopes)
    
    return total_loss / n_samples if n_samples > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            slopes = batch['slope'].to(device)
            
            predictions = model(features).squeeze()
            loss = criterion(predictions, slopes)
            
            total_loss += loss.item() * len(slopes)
            n_samples += len(slopes)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(slopes.cpu().numpy())
    
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, metrics


def objective(trial: Trial, config: dict, feature_type: str, 
              train_ids: list, val_ids: list, patient_data: dict, 
              features_data: dict, cnn_train_slopes: dict, cnn_val_slopes: dict):
    """Optuna objective for one feature type"""
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number} - Feature type: {feature_type}")
    print(f"{'='*80}")
    
    # Get input dimension
    if feature_type == 'demographics':
        input_dim = 1 + 3  # CNN + age, sex, smoking
    elif feature_type == 'handcrafted':
        input_dim = 1 + 9  # CNN + 9 handcrafted
    else:  # full
        input_dim = 1 + 3 + 9  # CNN + demographics + handcrafted
    
    # =========================================================================
    # ADAPTIVE HYPERPARAMETERS (scale with input size)
    # =========================================================================
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    # Learning rate (lower for more features)
    if input_dim <= 4:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    else:
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Batch size
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Weight decay (higher for more features)
    if input_dim <= 4:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    else:
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    
    # Architecture depth (more layers for more features)
    if input_dim <= 4:
        n_layers = trial.suggest_int('n_layers', 1, 2)
    else:
        n_layers = trial.suggest_int('n_layers', 2, 3)
    
    # Hidden sizes (adaptive using suggest_int for power-of-2 values)
    hidden_sizes = []
    for i in range(n_layers):
        if i == 0:
            # First layer: adaptive to input size
            if input_dim <= 4:
                # Suggest log2 of size: 5->32, 6->64
                log_size = trial.suggest_int(f'hidden_{i}_log', 5, 6)
            else:
                # Suggest log2 of size: 6->64, 7->128
                log_size = trial.suggest_int(f'hidden_{i}_log', 6, 7)
            size = 2 ** log_size
        else:
            # Subsequent layers: at most half of previous
            prev_size = hidden_sizes[-1]
            max_log = int(np.log2(prev_size // 2))
            min_log = 3  # minimum 8
            if max_log < min_log:
                max_log = min_log
            log_size = trial.suggest_int(f'hidden_{i}_log', min_log, max_log)
            size = 2 ** log_size
        
        hidden_sizes.append(size)
    
    # Dropout (higher for more features)
    dropout_rates = []
    for i in range(n_layers):
        if input_dim <= 4:
            dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.3)
        else:
            dropout = trial.suggest_float(f'dropout_{i}', 0.2, 0.5)
        dropout_rates.append(dropout)
    
    # Gradient clipping
    gradient_clip = trial.suggest_float('gradient_clip', 0.5, 2.0)

    if feature_type == 'full' and input_dim > 10:
        l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 1e-3, log=True)
    else:
        l1_lambda = 0.0
    
    print(f"\nHyperparameters:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Dropout rates: {[f'{d:.3f}' for d in dropout_rates]}")
    print(f"  lr: {lr:.6f}")
    print(f"  batch_size: {batch_size}")
    print(f"  weight_decay: {weight_decay:.6f}")
    print(f"  gradient_clip: {gradient_clip:.2f}")
    print(f"  l1_lambda: {l1_lambda:.6f}")
    # =========================================================================
    # CREATE DATASETS - Filter out patients with NaN features
    # =========================================================================
    
    # Filter train_ids for NaN features
    train_ids_clean = []
    for pid in train_ids:
        if pid not in features_data:
            continue
        pdata = features_data[pid]
        
        # Check for NaN in features based on feature_type
        has_nan = False
        if feature_type in ['handcrafted', 'full']:
            from utilities import HAND_FEATURE_ORDER
            hand_feats = [pdata.get(f, np.nan) for f in HAND_FEATURE_ORDER]
            if any(np.isnan(hand_feats)):
                has_nan = True
        
        if feature_type in ['demographics', 'full']:
            if np.isnan(pdata.get('age', np.nan)):
                has_nan = True
        
        if not has_nan:
            train_ids_clean.append(pid)
    
    # Filter val_ids for NaN features
    val_ids_clean = []
    for pid in val_ids:
        if pid not in features_data:
            continue
        pdata = features_data[pid]
        
        has_nan = False
        if feature_type in ['handcrafted', 'full']:
            from utilities import HAND_FEATURE_ORDER
            hand_feats = [pdata.get(f, np.nan) for f in HAND_FEATURE_ORDER]
            if any(np.isnan(hand_feats)):
                has_nan = True
        
        if feature_type in ['demographics', 'full']:
            if np.isnan(pdata.get('age', np.nan)):
                has_nan = True
        
        if not has_nan:
            val_ids_clean.append(pid)
    
    n_train_removed = len(train_ids) - len(train_ids_clean)
    n_val_removed = len(val_ids) - len(val_ids_clean)
    
    if n_train_removed > 0 or n_val_removed > 0:
        print(f"\n⚠️  Removed patients with NaN features:")
        print(f"   Train: {n_train_removed} removed ({len(train_ids_clean)} remain)")
        print(f"   Val: {n_val_removed} removed ({len(val_ids_clean)} remain)")
    
    print(f"\nDatasets:")
    print(f"  Train: {len(train_ids_clean)} patients")
    print(f"  Val: {len(val_ids_clean)} patients")
    
    train_dataset = CorrectorDataset(
        train_ids_clean,
        patient_data,
        features_data,
        cnn_train_slopes,
        feature_type=feature_type,
        normalizer=None
    )
    
    # Fit normalizer on training data
    normalizer = train_dataset.normalizer
    
    val_dataset = CorrectorDataset(
        val_ids_clean,
        patient_data,
        features_data,
        cnn_val_slopes,
        feature_type=feature_type,
        normalizer=normalizer
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Corrector is fast, no need for workers
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # =========================================================================
    # MODEL & OPTIMIZER
    # =========================================================================

    model = AdaptiveSlopeCorrector(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates,
        use_batch_norm=use_batch_norm
    ).to(config['device'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5)
    
    criterion = nn.MSELoss()
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training...\n")
    
    for epoch in range(config['n_epochs']):
        print(f"Epoch {epoch + 1}/{config['n_epochs']}: ", end='', flush=True)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            config['device'], gradient_clip, l1_lambda
        )
        
        val_loss, val_metrics = validate(
            model, val_loader, criterion, config['device']
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"MAE: {val_metrics['mae']:.6f} | LR: {current_lr:.2e}", 
              end='', flush=True)
        
        scheduler.step(val_loss)
        
        # Report to Optuna
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            print(" → PRUNED")
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(" ✓ Best")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{config['patience']})")
            if patience_counter >= config['patience']:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
    
    print(f"\n  Final best val loss: {best_val_loss:.6f}\n")
    
    return best_val_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("MLP CORRECTOR HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    loader = IPFDataLoader(
        CONFIG['csv_path'],
        CONFIG['features_path'],
        CONFIG['npy_dir']
    )
    patient_data, features_data = loader.get_patient_data()
    
    print(f"✓ Loaded {len(patient_data)} patients")
    print(f"✓ Loaded {len(features_data)} feature records")
    
    # Load CNN predictions
    print(f"\n📊 Loading CNN predictions...")
    if not CONFIG['cnn_predictions_path'].exists():
        raise FileNotFoundError(
            f"CNN predictions not found at {CONFIG['cnn_predictions_path']}\n"
            f"Run CNN training first to generate predictions!"
        )
    
    import pickle
    with open(CONFIG['cnn_predictions_path'], 'rb') as f:
        cnn_predictions_all = pickle.load(f)
    
    # Use exact same train/val splits as CNN (no merging, no re-splitting)
    if isinstance(cnn_predictions_all, dict) and 'train' in cnn_predictions_all:
        # Structured as {'train': {...}, 'val': {...}, 'test': {...}}
        cnn_train_slopes = cnn_predictions_all['train']
        cnn_val_slopes = cnn_predictions_all['val']
        print(f"✓ Loaded CNN predictions: {len(cnn_train_slopes)} train, {len(cnn_val_slopes)} val")
    else:
        raise ValueError("CNN predictions must have 'train' and 'val' keys")
    
    # Get train/val patient IDs directly from CNN predictions (same split as CNN used)
    fold_idx = CONFIG['fold_for_optimization']
    train_ids = list(cnn_train_slopes.keys())
    val_ids = list(cnn_val_slopes.keys())
    
    # Filter to patients that have features available
    train_ids = [pid for pid in train_ids if pid in patient_data]
    val_ids = [pid for pid in val_ids if pid in patient_data]
    
    print(f"\nOptimization fold {fold_idx}:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    
    # Optimize for each feature type
    for feature_type in CONFIG['feature_types']:
        print("\n" + "="*80)
        print(f"OPTIMIZING: {feature_type.upper()}")
        print("="*80)
        
        # Storage for this feature type
        storage_path = CONFIG['results_dir'] / f'optuna_{feature_type}.db'
        storage_url = f"sqlite:///{storage_path}"
        
        study = optuna.create_study(
            study_name=f'corrector_{feature_type}',
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            storage=storage_url,
            load_if_exists=True
        )
        
        # Check if resuming
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_completed > 0:
            print(f"\n🔄 RESUMING: Found {n_completed} completed trials")
            print(f"   Will run {CONFIG['n_trials'] - n_completed} more trials")
        else:
            print(f"\n🆕 NEW STUDY: Starting fresh with {CONFIG['n_trials']} trials")
        
        # Optimize
        study.optimize(
            lambda trial: objective(
                trial, CONFIG, feature_type, train_ids, val_ids,
                patient_data, features_data, cnn_train_slopes, cnn_val_slopes
            ),
            n_trials=CONFIG['n_trials'] - n_completed,
            show_progress_bar=True
        )
        
        # Save best parameters
        print("\n" + "="*80)
        print(f"OPTIMIZATION COMPLETE: {feature_type}")
        print("="*80)
        
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.6f}")
        
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save
        best_params_path = CONFIG['best_params_dir'] / f'best_params_{feature_type}.yaml'
        with open(best_params_path, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        
        print(f"\n✓ Saved best parameters to {best_params_path}")
    
    print("\n" + "="*80)
    print("✅ ALL OPTIMIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {CONFIG['results_dir']}")
    print(f"Best params saved in: {CONFIG['best_params_dir']}")


if __name__ == '__main__':
    main()
