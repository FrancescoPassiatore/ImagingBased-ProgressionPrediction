"""
CNN Hyperparameter Optimization with Optuna
============================================

Optimizes CNN hyperparameters using Optuna on fold 0.
Best hyperparameters will be used for full 5-fold training.

Hyperparameters to optimize:
- Learning rate
- Batch size (patients per batch)
- Weight decay
- Dropout rate
- Optimizer (AdamW vs SGD)
- Scheduler type
- Architecture depth
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
    IPFSliceDataset,
    PatientBatchSampler,
    patient_group_collate,
    ImprovedSliceLevelCNN,
    AttentionGuidedLoss,
    compute_metrics,
    save_batch_images
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    
    # Optuna
    'n_trials': 100,
    'fold_for_optimization': 0,  # Use fold 0 for hyperparameter search
    
    # Fixed parameters
    'n_epochs': 50,
    'patience': 10,
    'image_size': (224, 224),
    'backbone': 'efficientnet_b0',
    'pretrained': True,
    'normalize_slope': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Output
    'checkpoint_dir': Path('Training_2/CNN_Training/optuna_checkpoints'),
    'results_dir': Path('Training_2/CNN_Training/optuna_results'),
    'best_params_path': Path('Training_2/CNN_Training/best_params.yaml'),
    'visualization_dir': Path('Training_2/CNN_Training/training_images'),
    'save_images_every': 50  # Salva immagini ogni N batch
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, 
                gradient_clip: float = 1.0, accumulation_steps: int = 1,
                use_attention_loss: bool = False, save_dir: Path = None,
                epoch: int = 0, save_every: int = 50):
    """Train for one epoch - matching notebook implementation"""
    model.train()
    total_loss = 0.0
    n_patients = 0
    
    optimizer.zero_grad()
    batch_idx = -1
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        
        images = batch['images'].to(device)
        slopes = batch['slopes'].to(device)
        lengths = batch['lengths'].tolist()
        
        # Forward pass (with or without attention - but don't use attention in loss)
        # This matches the notebook: attention is computed but NOT used in training loss
        if use_attention_loss:
            preds_per_slice, _ = model(images, return_attention=True)
        else:
            preds_per_slice = model(images)
        
        preds_per_slice = preds_per_slice.view(-1)
        
        # Aggregate per patient (same as notebook)
        pred_blocks = torch.split(preds_per_slice, lengths)
        slopes_blocks = torch.split(slopes, lengths)
        
        patient_preds = torch.stack([block.mean() for block in pred_blocks])
        patient_slopes = torch.stack([block[0] for block in slopes_blocks])
        
        # Loss - ALWAYS use standard MSE, like in notebook
        loss = criterion(patient_preds, patient_slopes)
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        n_patients_in_batch = len(lengths)
        total_loss += loss.item() * n_patients_in_batch * accumulation_steps
        n_patients += n_patients_in_batch
    
    # Final update if there are remaining gradients
    if batch_idx >= 0 and (batch_idx + 1) % accumulation_steps != 0:  # ✓ Also check batch_idx >= 0
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / n_patients if n_patients > 0 else 0.0
        


def validate(model, dataloader, criterion, device):
    """
    Validate with patient-level aggregation
    
    Returns both patient-level loss and metrics
    """
    model.eval()
    total_loss = 0.0
    n_patients = 0
    
    all_patient_preds = []
    all_patient_slopes = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            slopes = batch['slopes'].to(device)
            lengths = batch['lengths'].tolist()
            
            # Predict per-slice
            preds_per_slice = model(images)
            
            # Aggregate per-patient
            pred_blocks = torch.split(preds_per_slice, lengths)
            slopes_blocks = torch.split(slopes, lengths)
            
            patient_preds = torch.stack([block.mean() for block in pred_blocks])
            patient_slopes = torch.stack([block[0] for block in slopes_blocks])
            
            # Patient-level loss
            loss = criterion(patient_preds, patient_slopes)
            
            n_patients_in_batch = len(lengths)
            total_loss += loss.item() * n_patients_in_batch
            n_patients += n_patients_in_batch
            
            # Store for metrics
            all_patient_preds.extend(patient_preds.cpu().numpy())
            all_patient_slopes.extend(patient_slopes.cpu().numpy())
    
    avg_loss = total_loss / n_patients if n_patients > 0 else 0.0
    
    # Compute patient-level metrics
    all_patient_preds = np.array(all_patient_preds)
    all_patient_slopes = np.array(all_patient_slopes)
    metrics = compute_metrics(all_patient_slopes, all_patient_preds)
    
    return avg_loss, metrics


def objective(trial: Trial, config: dict, train_ids: list, val_ids: list,
              patient_data: dict, features_data: dict):
    """Optuna objective function"""
    
    # Clear GPU cache before trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*80}")
    
    # =========================================================================
    # HYPERPARAMETERS TO OPTIMIZE
    # =========================================================================
    
    # Learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Batch size (patients per batch) - REDUCED for 8GB GPU
    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    
    # Weight decay
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Dropout
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Gradient clipping
    gradient_clip = trial.suggest_float('gradient_clip', 0.5, 2.0)
    
    # Optimizer type
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    
    # Scheduler type
    scheduler_name = trial.suggest_categorical('scheduler', 
                                                ['ReduceLROnPlateau', 'CosineAnnealing', 'StepLR'])
    
    # Gradient accumulation steps
    accumulation_steps = trial.suggest_categorical('accumulation_steps', [1, 2, 4])
    
    # Use attention mechanism (for feature extraction, not loss)
    # This matches notebook: attention is used but NOT in the loss function
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    
    # Print selected hyperparameters
    print(f"\nHyperparameters:")
    print(f"  lr: {lr:.6f}")
    print(f"  batch_size: {batch_size}")
    print(f"  weight_decay: {weight_decay:.6f}")
    print(f"  dropout: {dropout:.3f}")
    print(f"  gradient_clip: {gradient_clip:.2f}")
    print(f"  optimizer: {optimizer_name}")
    print(f"  scheduler: {scheduler_name}")
    print(f"  accumulation_steps: {accumulation_steps}")
    print(f"  use_attention: {use_attention}")
    
    # =========================================================================
    # CREATE DATASETS
    # =========================================================================
    
    train_dataset = IPFSliceDataset(
        train_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        augment=False
    )
    
    val_dataset = IPFSliceDataset(
        val_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        slope_scaler=train_dataset.slope_scaler,
        augment=False
    )
    
    print(f"\nDatasets:")
    print(f"  Train slices: {len(train_dataset)}")
    print(f"  Val slices: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PatientBatchSampler(
            train_dataset,
            patients_per_batch=batch_size,
            shuffle=True
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=PatientBatchSampler(
            val_dataset,
            patients_per_batch=batch_size,
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    
    model = ImprovedSliceLevelCNN(
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=dropout
    ).to(config['device'])
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:  # SGD
        momentum = trial.suggest_float('sgd_momentum', 0.85, 0.95)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
    
    # =========================================================================
    # SCHEDULER
    # =========================================================================
    
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 7)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience
        )
    elif scheduler_name == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
    else:  # StepLR
        step_size = trial.suggest_int('step_size', 5, 15)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=0.5
        )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    # Use standard MSE loss (like notebook - attention NOT used in loss)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training...\n")
    
    try:
        for epoch in range(config['n_epochs']):
            print(f"Epoch {epoch + 1}/{config['n_epochs']}: ", end='', flush=True)
        
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, config['device'],
                gradient_clip=gradient_clip,
                accumulation_steps=accumulation_steps,
                use_attention_loss=use_attention,  # Compute attention but don't use in loss
                save_dir=config['visualization_dir'],
                epoch=epoch,
                save_every=config['save_images_every']
            )
            
            # Validate
            val_loss, val_metrics = validate(model, val_loader, criterion, config['device'])
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | MAE: {val_metrics['mae']:.6f} | LR: {current_lr:.2e}", end='', flush=True)
            
            # Update scheduler
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)
            
            # Check if trial should be pruned
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
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n  ⚠️  OOM Error: {e}")
            # Clear cache and return a penalty value
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('inf')  # Return worst possible loss
        else:
            raise e
    
    finally:
        # Always clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n  Final best val loss: {best_val_loss:.6f}\n")
    
    return best_val_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("CNN HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    print(f"\nDevice: {CONFIG['device']}")
    print(f"Trials: {CONFIG['n_trials']}")
    print(f"Optimization fold: {CONFIG['fold_for_optimization']}")
    
    # Create output directories
    CONFIG['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['best_params_path'].parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    dl = IPFDataLoader(CONFIG['csv_path'], CONFIG['features_path'], CONFIG['npy_dir'])
    patient_data, features_data = dl.get_patient_data()
    
    print(f"✓ Loaded {len(patient_data)} patients")
    
    # Create splits
    splits_path = CONFIG['checkpoint_dir'] / 'kfold_splits.pkl'
    if splits_path.exists():
        import pickle
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        print(f"✓ Loaded existing splits from {splits_path}")
    else:
        splits = create_kfold_splits(
            list(patient_data.keys()),
            n_folds=5,
            random_state=42,
            save_path=splits_path
        )
    
    # Get train/val for optimization fold
    fold_idx = CONFIG['fold_for_optimization']
    train_ids = splits[fold_idx]['train']
    val_ids = splits[fold_idx]['val']
    
    print(f"\nOptimization fold {fold_idx}:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    
    # Create Optuna study
    print("\n" + "="*80)
    print("STARTING OPTUNA OPTIMIZATION")
    print("="*80)
    
    # Use SQLite storage for persistence (allows resume)
    storage_path = CONFIG['results_dir'] / 'optuna_study.db'
    storage_url = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name='cnn_hyperparam_optimization',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=storage_url,
        load_if_exists=True  # Resume if study exists
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
        lambda trial: objective(trial, CONFIG, train_ids, val_ids, 
                               patient_data, features_data),
        n_trials=CONFIG['n_trials'] - n_completed,  # Only run remaining trials
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    with open(CONFIG['best_params_path'], 'w') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    
    print(f"\n✓ Saved best parameters to {CONFIG['best_params_path']}")
    
    # Save study
    study_path = CONFIG['results_dir'] / 'optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"✓ Saved study to {study_path}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig1.savefig(CONFIG['results_dir'] / 'optimization_history.png', dpi=300, bbox_inches='tight')
        
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        fig2.savefig(CONFIG['results_dir'] / 'param_importances.png', dpi=300, bbox_inches='tight')
        
        print(f"✓ Saved plots to {CONFIG['results_dir']}")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nNext step: Run train_cnn_best.py with these hyperparameters")


if __name__ == "__main__":
    main()
