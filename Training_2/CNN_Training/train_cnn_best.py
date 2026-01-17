"""
CNN Training with Best Hyperparameters (5-Fold Cross-Validation)
=================================================================

Trains the CNN model using best hyperparameters from Optuna on all 5 folds.
Saves models and predictions for MLP corrector training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import pickle
from tqdm import tqdm
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
    
    # Best params from Optuna
    'best_params_path': Path('Training_2/CNN_Training/best_params.yaml'),
    
    # Training
    'n_folds': 5,
    'n_epochs': 100,
    'patience': 20,
    'image_size': (224, 224),
    'backbone': 'efficientnet_b0',
    'pretrained': True,
    'normalize_slope': False,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Output
    'checkpoint_dir': Path('Training_2/CNN_Training/final_checkpoints_no_norm'),
    'predictions_dir': Path('Training_2/CNN_Training/predictions_no_norm'),
    'results_dir': Path('Training_2/CNN_Training/final_results_no_norm')
}

# Create directories
for dir_path in [CONFIG['checkpoint_dir'], CONFIG['predictions_dir'], 
                 CONFIG['results_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, 
                gradient_clip=1.0, accumulation_steps=1, use_attention=False):
    """Train for one epoch"""
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
        
        # Forward pass
        if use_attention:
            preds_per_slice, _ = model(images, return_attention=True)
        else:
            preds_per_slice = model(images)
        
        preds_per_slice = preds_per_slice.view(-1)
        
        # Aggregate per patient
        pred_blocks = torch.split(preds_per_slice, lengths)
        slopes_blocks = torch.split(slopes, lengths)
        
        patient_preds = torch.stack([block.mean() for block in pred_blocks])
        patient_slopes = torch.stack([block[0] for block in slopes_blocks])
        
        # Loss
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
    
    # Final update
    if batch_idx >= 0 and (batch_idx + 1) % accumulation_steps != 0:
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / n_patients if n_patients > 0 else 0.0


def validate(model, dataloader, criterion, device, return_predictions=False):
    """Validate"""
    model.eval()
    total_loss = 0.0
    n_patients = 0
    
    all_patient_preds = []
    all_patient_slopes = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            slopes = batch['slopes'].to(device)
            lengths = batch['lengths'].tolist()
            patient_ids = batch['patient_ids']
            
            preds_per_slice = model(images)
            
            pred_blocks = torch.split(preds_per_slice, lengths)
            slopes_blocks = torch.split(slopes, lengths)
            
            patient_preds = torch.stack([block.mean() for block in pred_blocks])
            patient_slopes = torch.stack([block[0] for block in slopes_blocks])
            
            loss = criterion(patient_preds, patient_slopes)
            
            n_patients_in_batch = len(lengths)
            total_loss += loss.item() * n_patients_in_batch
            n_patients += n_patients_in_batch
            
            all_patient_preds.extend(patient_preds.cpu().numpy())
            all_patient_slopes.extend(patient_slopes.cpu().numpy())
            all_patient_ids.extend(patient_ids)
    
    avg_loss = total_loss / n_patients if n_patients > 0 else 0.0
    
    all_patient_preds = np.array(all_patient_preds)
    all_patient_slopes = np.array(all_patient_slopes)
    metrics = compute_metrics(all_patient_slopes, all_patient_preds)
    
    if return_predictions:
        return avg_loss, metrics, all_patient_preds, all_patient_slopes, all_patient_ids
    
    return avg_loss, metrics


def predict_fold(model, dataloader, device, slope_scaler):
    """Generate predictions for all patients in a fold"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            lengths = batch['lengths'].tolist()
            patient_ids = batch['patient_ids']
            
            preds_per_slice = model(images)
            
            pred_blocks = torch.split(preds_per_slice, lengths)
            
            for patient_id, pred_block in zip(patient_ids, pred_blocks):
                # Mean of slices (normalized)
                pred_mean_norm = pred_block.mean().cpu().item()
                
                # Denormalize
                if slope_scaler is not None:
                    pred_mean = slope_scaler.inverse_transform([[pred_mean_norm]])[0][0]
                else:
                    pred_mean = pred_mean_norm
                
                predictions[patient_id] = pred_mean
    
    return predictions


def train_fold(fold_idx, train_ids, val_ids, test_ids, patient_data, features_data, 
               best_params, config):
    """Train one fold"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*80}")
    print(f"Train: {len(train_ids)} patients | Val: {len(val_ids)} patients | Test: {len(test_ids)} patients")
    
    # Create datasets
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
    
    print(f"Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PatientBatchSampler(
            train_dataset,
            patients_per_batch=best_params['batch_size'],
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
            patients_per_batch=best_params['batch_size'],
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = ImprovedSliceLevelCNN(
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=best_params['dropout']
    ).to(config['device'])
    
    # Optimizer
    if best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            momentum=best_params.get('sgd_momentum', 0.9),
            nesterov=True
        )
    
    # Scheduler
    if best_params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=best_params.get('scheduler_patience', 5)
        )
    elif best_params['scheduler'] == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=best_params.get('step_size', 10),
            gamma=0.5
        )
    
    criterion = nn.HuberLoss(delta=1.0)  # ✓ Instead of MSELoss
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    print(f"\n🚀 Starting training...\n")
    
    for epoch in range(config['n_epochs']):
        print(f"Epoch {epoch + 1}/{config['n_epochs']}: ", end='', flush=True)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config['device'],
            gradient_clip=best_params['gradient_clip'],
            accumulation_steps=best_params['accumulation_steps'],
            use_attention=best_params['use_attention']
        )
        
        # Get predictions every 5 epochs for detailed comparison
        return_preds = (epoch + 1) % 5 == 0
        if return_preds:
            val_loss, val_metrics, preds, trues, patient_ids = validate(
                model, val_loader, criterion, config['device'], return_predictions=True
            )
        else:
            val_loss, val_metrics = validate(model, val_loader, criterion, config['device'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"MAE: {val_metrics['mae']:.6f} | RMSE: {val_metrics['rmse']:.6f} | "
              f"R²: {val_metrics['r2']:.4f} | LR: {current_lr:.2e}", end='', flush=True)
        
        # Update scheduler
        if best_params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(" ✓ Best")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{config['patience']})")
            if patience_counter >= config['patience']:
                print(f"\n  ⏹️  Early stopping at epoch {epoch + 1}")
                break
        
        # Print detailed comparison every 5 epochs
        if return_preds:
            print(f"\n  {'─'*76}")
            print(f"  📊 Sample predictions (first 5 patients):")
            print(f"  {'─'*76}")
            print(f"  {'Patient ID':<20} {'Predicted':<15} {'True':<15} {'Error':<15}")
            print(f"  {'─'*76}")
            for i in range(min(5, len(patient_ids))):
                pred_val = float(preds[i])
                true_val = float(trues[i])
                error = pred_val - true_val
                print(f"  {patient_ids[i]:<20} {pred_val:>12.4f}   {true_val:>12.4f}   {error:>12.4f}")
            print(f"  {'─'*76}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save checkpoint
    checkpoint_path = config['checkpoint_dir'] / f'cnn_fold{fold_idx}.pt'
    torch.save({
        'fold': fold_idx,
        'model_state_dict': best_model_state,
        'best_val_loss': best_val_loss,
        'hyperparameters': best_params,
        'slope_scaler': train_dataset.slope_scaler,
        'history': history
    }, checkpoint_path)
    
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")
    
    # Generate predictions for corrector
    print(f"\n📊 Generating predictions for corrector...")
    
    # Create test dataset and loader
    test_dataset = IPFSliceDataset(
        test_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        slope_scaler=train_dataset.slope_scaler,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=PatientBatchSampler(
            test_dataset,
            patients_per_batch=best_params['batch_size'],
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    train_preds = predict_fold(model, train_loader, config['device'], train_dataset.slope_scaler)
    val_preds = predict_fold(model, val_loader, config['device'], train_dataset.slope_scaler)
    test_preds = predict_fold(model, test_loader, config['device'], train_dataset.slope_scaler)
    
    predictions = {'train': train_preds, 'val': val_preds, 'test': test_preds}
    
    pred_path = config['predictions_dir'] / f'cnn_predictions_fold{fold_idx}.pkl'
    with open(pred_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    print(f"✓ Saved predictions to {pred_path}")
    print(f"  Train: {len(train_preds)} patients | Val: {len(val_preds)} patients | Test: {len(test_preds)} patients")
    
    return {
        'fold': fold_idx,
        'best_val_loss': best_val_loss,
        'history': history,
        'final_metrics': val_metrics
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("CNN TRAINING - 5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Load best hyperparameters
    print(f"\n📋 Loading best hyperparameters from {CONFIG['best_params_path']}")
    
    if not CONFIG['best_params_path'].exists():
        raise FileNotFoundError(
            f"Best parameters not found at {CONFIG['best_params_path']}\n"
            f"Run train_cnn_optuna.py first to find optimal hyperparameters!"
        )
    
    with open(CONFIG['best_params_path'], 'r') as f:
        best_params = yaml.safe_load(f)
    
    print("\n📊 Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
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
    
    # Load K-fold splits
    splits_path = Path('Training_2/kfold_splits.pkl')
    if splits_path.exists():
        print(f"\n📁 Loading K-fold splits from {splits_path}")
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
    else:
        print("\n🔄 Creating K-fold splits...")
        splits = create_kfold_splits(
            list(patient_data.keys()),
            n_folds=CONFIG['n_folds'],
            random_state=42,
            save_path=splits_path
        )
    
    # Train all folds
    all_results = []
    
    for fold_idx in range(CONFIG['n_folds']):
        train_ids = splits[fold_idx]['train']
        val_ids = splits[fold_idx]['val']
        test_ids = splits[fold_idx]['test']
        
        fold_result = train_fold(
            fold_idx, train_ids, val_ids, test_ids,
            patient_data, features_data,
            best_params, CONFIG
        )
        
        all_results.append(fold_result)
    
    # Aggregate results
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL FOLDS")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            'Fold': r['fold'],
            'Val Loss': r['best_val_loss'],
            'MAE': r['final_metrics']['mae'],
            'RMSE': r['final_metrics']['rmse'],
            'R²': r['final_metrics']['r2']
        }
        for r in all_results
    ])
    
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("AVERAGE METRICS:")
    print("-"*80)
    print(f"Val Loss: {results_df['Val Loss'].mean():.6f} ± {results_df['Val Loss'].std():.6f}")
    print(f"MAE:      {results_df['MAE'].mean():.6f} ± {results_df['MAE'].std():.6f}")
    print(f"RMSE:     {results_df['RMSE'].mean():.6f} ± {results_df['RMSE'].std():.6f}")
    print(f"R²:       {results_df['R²'].mean():.4f} ± {results_df['R²'].std():.4f}")
    
    # Save results
    results_path = CONFIG['results_dir'] / 'final_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")
    
    # Save summary
    summary = {
        'hyperparameters': best_params,
        'mean_metrics': {
            'val_loss': float(results_df['Val Loss'].mean()),
            'mae': float(results_df['MAE'].mean()),
            'rmse': float(results_df['RMSE'].mean()),
            'r2': float(results_df['R²'].mean())
        },
        'std_metrics': {
            'val_loss': float(results_df['Val Loss'].std()),
            'mae': float(results_df['MAE'].std()),
            'rmse': float(results_df['RMSE'].std()),
            'r2': float(results_df['R²'].std())
        },
        'per_fold': [
            {
                'fold': r['fold'],
                'best_val_loss': r['best_val_loss'],
                'final_metrics': r['final_metrics']
            }
            for r in all_results
        ]
    }
    
    summary_path = CONFIG['results_dir'] / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"📁 Predictions: {CONFIG['predictions_dir']}")
    print(f"📁 Results: {CONFIG['results_dir']}")


if __name__ == '__main__':
    main()
