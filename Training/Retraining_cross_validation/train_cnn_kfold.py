"""
Train CNN Slope Predictor with 5-Fold Cross-Validation
=======================================================

This script trains the CNN backbone to predict per-slice slopes using 5-fold CV.
The trained CNN is used by all 4 approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.filterwarnings('ignore')

from utilities import (
    IPFDataLoader,
    create_kfold_splits,
    IPFSliceDataset,
    PatientBatchSampler,
    patient_group_collate,
    ImprovedSliceLevelCNN,
    save_fold_results
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': 'Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'features_path': 'Training/CNN_Slope_Prediction/patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    
    # Model
    'backbone': 'efficientnet_b0',
    'pretrained': True,
    
    # Training
    'n_folds': 5,
    'n_epochs': 50,
    'patience': 10,
    'batch_size': 4,  # patients per batch
    'lr': 1e-4,
    'weight_decay': 1e-4,
    
    # Data
    'image_size': (224, 224),
    'normalize_slope': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Output
    'checkpoint_dir': Path('Training/Retraining_cross_validation/checkpoints'),
    'results_dir': Path('Training/Retraining_cross_validation/results'),
    'plots_dir': Path('Training/Retraining_cross_validation/plots')
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train CNN for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue
        
        images = batch['images'].to(device)
        slopes = batch['slopes'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, slopes)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / n_samples if n_samples > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate CNN"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    all_preds = []
    all_true = []
    all_patients = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            slopes = batch['slopes'].to(device)
            patient_ids = batch['patient_ids']
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, slopes)
            
            # Track metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            # Store predictions
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(slopes.cpu().numpy())
            all_patients.extend(patient_ids)
    
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    
    # Compute per-patient metrics
    patient_preds = {}
    patient_true = {}
    
    for pred, true, pid in zip(all_preds, all_true, all_patients):
        if pid not in patient_preds:
            patient_preds[pid] = []
            patient_true[pid] = []
        patient_preds[pid].append(pred)
        patient_true[pid].append(true)
    
    # Average per patient
    patient_mean_preds = {pid: np.mean(preds) for pid, preds in patient_preds.items()}
    patient_mean_true = {pid: np.mean(true) for pid, true in patient_true.items()}
    
    return {
        'loss': avg_loss,
        'slice_preds': all_preds,
        'slice_true': all_true,
        'patient_preds': patient_mean_preds,
        'patient_true': patient_mean_true
    }


def train_fold(fold_idx, train_ids, val_ids, patient_data, features_data, config):
    """Train CNN for one fold"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{config['n_folds']}")
    print(f"{'='*80}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Val patients: {len(val_ids)}")
    
    # Create datasets
    train_dataset = IPFSliceDataset(
        train_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope']
    )
    
    val_dataset = IPFSliceDataset(
        val_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        slope_scaler=train_dataset.slope_scaler  # Use same scaler
    )
    
    print(f"Train slices: {len(train_dataset)}")
    print(f"Val slices: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PatientBatchSampler(
            train_dataset,
            patients_per_batch=config['batch_size'],
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
            patients_per_batch=config['batch_size'],
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = ImprovedSliceLevelCNN(
        backbone_name=config['backbone'],
        pretrained=config['pretrained']
    ).to(config['device'])
    
    # Optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(config['n_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['n_epochs']}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config['device'])
        history['train_loss'].append(train_loss)
        
        # Validate
        val_results = validate(model, val_loader, criterion, config['device'])
        val_loss = val_results['loss']
        history['val_loss'].append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = config['checkpoint_dir'] / f'cnn_fold{fold_idx}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping triggered (patience={config['patience']})")
                break
    
    # Load best model
    model.load_state_dict(torch.load(config['checkpoint_dir'] / f'cnn_fold{fold_idx}.pth'))
    
    # Final validation
    final_val_results = validate(model, val_loader, criterion, config['device'])
    
    # Save slope scaler
    scaler_path = config['checkpoint_dir'] / f'slope_scaler_fold{fold_idx}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.slope_scaler, f)
    
    return {
        'model_state': model.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'final_val_results': final_val_results,
        'slope_scaler': train_dataset.slope_scaler,
        'train_ids': train_ids,
        'val_ids': val_ids
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("CNN SLOPE PREDICTOR - 5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Device info
    print(f"\nDevice: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_loader = IPFDataLoader(
        CONFIG['csv_path'],
        CONFIG['features_path'],
        CONFIG['npy_dir']
    )
    
    patient_data, features_data = data_loader.get_patient_data()
    
    print(f"✓ Loaded {len(patient_data)} patients")
    print(f"✓ Loaded {len(features_data)} feature sets")
    
    # Get all patient IDs
    all_patient_ids = list(patient_data.keys())
    print(f"\nTotal patients: {len(all_patient_ids)}")
    
    # Create K-Fold splits
    print("\n" + "="*80)
    print("CREATING K-FOLD SPLITS")
    print("="*80)
    
    splits = create_kfold_splits(all_patient_ids, n_splits=CONFIG['n_folds'], random_state=42)
    
    for i, (train_ids, val_ids) in enumerate(splits):
        print(f"Fold {i+1}: Train={len(train_ids)}, Val={len(val_ids)}")
    
    # Save splits
    splits_path = CONFIG['results_dir'] / 'kfold_splits.pkl'
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\n✓ Saved splits: {splits_path}")
    
    # Train each fold
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    all_fold_results = []
    
    for fold_idx, (train_ids, val_ids) in enumerate(splits):
        fold_results = train_fold(
            fold_idx,
            train_ids,
            val_ids,
            patient_data,
            features_data,
            CONFIG
        )
        
        all_fold_results.append(fold_results)
        
        # Save fold results
        save_fold_results(
            fold_idx,
            'cnn',
            fold_results,
            CONFIG['results_dir']
        )
    
    # Aggregate results
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    
    val_losses = [r['best_val_loss'] for r in all_fold_results]
    
    print(f"\nValidation Loss (MSE):")
    for i, loss in enumerate(val_losses):
        print(f"  Fold {i+1}: {loss:.6f}")
    print(f"\n  Mean: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
    
    # Plot training curves
    print("\n" + "="*80)
    print("PLOTTING TRAINING CURVES")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    for i, results in enumerate(all_fold_results):
        history = results['history']
        ax.plot(history['train_loss'], label=f'Fold {i+1} Train', alpha=0.7)
        ax.plot(history['val_loss'], label=f'Fold {i+1} Val', alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Curves (All Folds)', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Box plot of final losses
    ax = axes[1]
    ax.bar(range(1, CONFIG['n_folds'] + 1), val_losses, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(np.mean(val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_losses):.6f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Best Val Loss (MSE)', fontsize=12)
    ax.set_title('Best Validation Loss per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, CONFIG['n_folds'] + 1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = CONFIG['plots_dir'] / 'cnn_training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.show()
    
    print("\n" + "="*80)
    print("✅ CNN TRAINING COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved in: {CONFIG['checkpoint_dir']}")
    print(f"Results saved in: {CONFIG['results_dir']}")
    print(f"Plots saved in: {CONFIG['plots_dir']}")


if __name__ == "__main__":
    main()
