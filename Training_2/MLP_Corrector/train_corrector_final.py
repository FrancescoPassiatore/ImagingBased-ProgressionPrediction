"""
MLP Corrector Training with Optimized Hyperparameters
======================================================

Trains MLP corrector on all 5 folds using the best hyperparameters 
found by Optuna for each feature type.

Feature types:
- demographics: CNN + age, sex, smoking
- handcrafted: CNN + 9 handcrafted features
- full: CNN + demographics + handcrafted (13 features total)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader,
    CorrectorDataset,
    FeatureNormalizer,
    ImprovedSlopeCorrector,
    compute_metrics
)

# Configuration
CONFIG = {
    'npy_dir': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Dataset/extracted_npy/extracted_npy',
    'train_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'features_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/patient_features.csv',
    'cnn_predictions_dir': 'Training_2/CNN_Training/predictions_robust',
    'best_params_dir': Path('Training_2/MLP_Corrector/best_params'),
    'results_dir': Path('Training_2/MLP_Corrector/results'),
    'models_dir': Path('Training_2/MLP_Corrector/models'),
    'predictions_dir': Path('Training_2/MLP_Corrector/predictions'),
    'feature_types': ['demographics', 'handcrafted', 'full'],
    'n_folds': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_epochs': 1000,
}

# Create output directories
CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['models_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['predictions_dir'].mkdir(parents=True, exist_ok=True)


def train_epoch(model, dataloader, criterion, optimizer, device, l1_lambda, gradient_clip):
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
        
        # L1 regularization on first layer
        l1_loss = 0.0
        if l1_lambda > 0:
            first_layer = model.network[0]
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
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []
    all_patient_ids = []
    
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
            all_patient_ids.extend(batch['patient_id'])
    
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, metrics, all_patient_ids, all_preds, all_targets


def train_one_fold(fold_idx, feature_type, params, patient_data, features_data, 
                   cnn_train_slopes, cnn_val_slopes, device):
    """Train MLP corrector for one fold"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} - {feature_type.upper()}")
    print(f"{'='*80}")
    
    # Get patient IDs from CNN predictions
    train_ids = list(cnn_train_slopes.keys())
    val_ids = list(cnn_val_slopes.keys())
    
    # Filter for NaN in features
    train_ids_clean = []
    for pid in train_ids:
        if pid not in features_data or pid not in patient_data:
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
            train_ids_clean.append(pid)
    
    val_ids_clean = []
    for pid in val_ids:
        if pid not in features_data or pid not in patient_data:
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
    
    print(f"Patients: Train={len(train_ids_clean)}, Val={len(val_ids_clean)}")
    
    # Create datasets
    train_dataset = CorrectorDataset(
        train_ids_clean,
        patient_data,
        features_data,
        cnn_train_slopes,
        feature_type=feature_type,
        normalizer=None
    )
    
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
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Get input dimension
    if feature_type == 'demographics':
        input_dim = 1 + 3
    elif feature_type == 'handcrafted':
        input_dim = 1 + 9
    else:  # full
        input_dim = 1 + 9 + 3
    
    # Build hidden sizes from log params
    hidden_sizes = []
    for i in range(params['n_layers']):
        log_size = params[f'hidden_{i}_log']
        hidden_sizes.append(2 ** log_size)
    
    # Build dropout rates
    dropout_rates = []
    for i in range(params['n_layers']):
        dropout_rates.append(params[f'dropout_{i}'])
    
    print(f"\nModel architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {hidden_sizes}")
    print(f"  Dropout: {[f'{d:.3f}' for d in dropout_rates]}")
    print(f"  Batch norm: {params['use_batch_norm']}")
    print(f"  LR: {params['lr']:.6f}")
    print(f"  Weight decay: {params['weight_decay']:.6f}")
    print(f"  Gradient clip: {params['gradient_clip']:.2f}")
    
    # Create model (ImprovedSlopeCorrector uses 'hidden_dims' not 'hidden_sizes')
    model = ImprovedSlopeCorrector(
        input_dim=input_dim,
        hidden_dims=hidden_sizes,
        dropout_rates=dropout_rates
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    # Get L1 lambda
    l1_lambda = params.get('l1_lambda', 0.0)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    
    print("\nStarting training...")
    
    for epoch in range(CONFIG['max_epochs']):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            l1_lambda, params['gradient_clip']
        )
        
        val_loss, val_metrics, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            
            model_path = CONFIG['models_dir'] / f'{feature_type}_fold{fold_idx}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics,
                'params': params,
                'normalizer': normalizer
            }, model_path)
        
        # Print progress every 5 epochs or when best model found
        if (epoch + 1) % 5 == 0 or val_loss == best_val_loss:
            print(f"Epoch {epoch+1:3d}/{CONFIG['max_epochs']}: "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"MAE: {val_metrics['mae']:.6f} | "
                  f"R²: {val_metrics['r2']:.4f}{' *' if val_loss == best_val_loss else ''}")
    print(f"  MAE: {best_metrics['mae']:.6f}")
    print(f"  R²: {best_metrics['r2']:.4f}")
    print(f"  RMSE: {best_metrics['rmse']:.4f}")
    
    # Load best model and get predictions on train and val
    print(f"\n📊 Generating predictions...")
    model_path = CONFIG['models_dir'] / f'{feature_type}_fold{fold_idx}_best.pt'
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get train predictions
    train_loader_eval = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=0
    )
    _, _, train_patient_ids, train_preds, train_targets = validate(
        model, train_loader_eval, criterion, device
    )
    
    # Get val predictions
    _, _, val_patient_ids, val_preds, val_targets = validate(
        model, val_loader, criterion, device
    )
    
    # Create predictions dict (patient_id -> prediction)
    train_predictions_dict = {pid: pred for pid, pred in zip(train_patient_ids, train_preds)}
    val_predictions_dict = {pid: pred for pid, pred in zip(val_patient_ids, val_preds)}
    
    # Save predictions
    predictions = {
        'train': train_predictions_dict,
        'val': val_predictions_dict,
    }
    
    pred_path = CONFIG['predictions_dir'] / f'{feature_type}_predictions_fold{fold_idx}.pkl'
    import pickle
    with open(pred_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    print(f"✓ Saved predictions to {pred_path}")
    
    return best_val_loss, best_metrics


def main():
    print("="*80)
    print("MLP CORRECTOR TRAINING - ALL FOLDS")
    print("="*80)
    
    device = torch.device(CONFIG['device'])
    print(f"\nDevice: {device}")
    
    # Load patient data
    print("\n📁 Loading patient data...")
    data_loader = IPFDataLoader(
        csv_path=CONFIG['train_csv'],
        features_path=CONFIG['features_csv'],
        npy_dir=CONFIG['npy_dir']
    )
    patient_data, features_data = data_loader.get_patient_data()
    print(f"✓ Loaded {len(patient_data)} patients")
    
    # Results storage
    all_results = []
    
    # Train each feature type on all folds
    for feature_type in CONFIG['feature_types']:
        print(f"\n{'#'*80}")
        print(f"# TRAINING: {feature_type.upper()}")
        print(f"{'#'*80}")
        
        # Load best parameters
        params_path = CONFIG['best_params_dir'] / f'best_params_{feature_type}.yaml'
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        print(f"\n✓ Loaded best parameters from {params_path}")
        
        # Train on all folds
        for fold_idx in range(CONFIG['n_folds']):
            # Load CNN predictions for this fold
            cnn_pred_path = Path(CONFIG['cnn_predictions_dir']) / f'cnn_predictions_fold{fold_idx}.pkl'
            
            if not cnn_pred_path.exists():
                print(f"\n⚠️  CNN predictions not found for fold {fold_idx}: {cnn_pred_path}")
                continue
            
            import pickle
            with open(cnn_pred_path, 'rb') as f:
                cnn_predictions_all = pickle.load(f)
            
            cnn_train_slopes = cnn_predictions_all['train']
            cnn_val_slopes = cnn_predictions_all['val']
            
            # Train this fold
            val_loss, val_metrics = train_one_fold(
                fold_idx, feature_type, params,
                patient_data, features_data,
                cnn_train_slopes, cnn_val_slopes,
                device
            )
            
            # Store results
            all_results.append({
                'feature_type': feature_type,
                'fold': fold_idx,
                'val_loss': val_loss,
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'val_rmse': val_metrics['rmse'],
                'val_mse': val_metrics['mse'],
            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = CONFIG['results_dir'] / 'mlp_corrector_all_folds_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✓ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Mean ± Std across 5 folds")
    print("="*80)
    
    for feature_type in CONFIG['feature_types']:
        fold_results = results_df[results_df['feature_type'] == feature_type]
        print(f"\n{feature_type.upper()}:")
        print(f"  R²:    {fold_results['val_r2'].mean():.4f} ± {fold_results['val_r2'].std():.4f}")
        print(f"  MAE:   {fold_results['val_mae'].mean():.4f} ± {fold_results['val_mae'].std():.4f}")
        print(f"  RMSE:  {fold_results['val_rmse'].mean():.4f} ± {fold_results['val_rmse'].std():.4f}")


if __name__ == '__main__':
    main()
