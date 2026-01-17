"""
Train Slope Correctors with 5-Fold Cross-Validation
====================================================

This script trains 4 different slope correction approaches:
1. CNN-only (no correction)
2. CNN + Handcrafted features
3. CNN + Demographics
4. CNN + Handcrafted + Demographics
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
from collections import defaultdict

warnings.filterwarnings('ignore')

from utilities import (
    IPFDataLoader,
    IPFSliceDataset,
    PatientBatchSampler,
    patient_group_collate,
    ImprovedSliceLevelCNN,
    SlopeCorrectorCNNOnly,
    SlopeCorrectorCNNHandcrafted,
    SlopeCorrectorCNNDemographics,
    SlopeCorrectorFull,
    CorrectorDataset,
    extract_patient_features,
    HAND_FEATURE_ORDER,
    DEMOGRAPHIC_FEATURES,
    save_fold_results,
    load_fold_results
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    
    # Training
    'n_folds': 5,
    'n_epochs': 200,
    'patience': 15,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    
    # Data
    'image_size': (224, 224),
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Input/Output
    'checkpoint_dir': Path('Training/Retraining_cross_validation/checkpoints'),
    'results_dir': Path('Training/Retraining_cross_validation/results'),
    'plots_dir': Path('Training/Retraining_cross_validation/plots')
}

# Define approaches
APPROACHES = {
    'cnn_only': {
        'name': 'CNN Only',
        'feature_type': 'none',
        'model_class': SlopeCorrectorCNNOnly,
        'model_kwargs': {}
    },
    'cnn_handcrafted': {
        'name': 'CNN + Handcrafted',
        'feature_type': 'handcrafted',
        'model_class': SlopeCorrectorCNNHandcrafted,
        'model_kwargs': {'n_handcrafted': len(HAND_FEATURE_ORDER)}
    },
    'cnn_demographics': {
        'name': 'CNN + Demographics',
        'feature_type': 'demographics',
        'model_class': SlopeCorrectorCNNDemographics,
        'model_kwargs': {'n_demographics': len(DEMOGRAPHIC_FEATURES)}
    },
    'cnn_full': {
        'name': 'CNN + Handcrafted + Demographics',
        'feature_type': 'full',
        'model_class': SlopeCorrectorFull,
        'model_kwargs': {
            'n_handcrafted': len(HAND_FEATURE_ORDER),
            'n_demographics': len(DEMOGRAPHIC_FEATURES)
        }
    }
}

# =============================================================================
# EXTRACT CNN SLOPES
# =============================================================================

def extract_cnn_slopes_for_patients(model, patient_ids, patient_data, features_data, 
                                    slope_scaler, device, image_size=(224, 224)):
    """Extract mean CNN slopes for each patient"""
    
    print(f"\nExtracting CNN slopes for {len(patient_ids)} patients...")
    
    # Create dataset
    dataset = IPFSliceDataset(
        patient_ids,
        patient_data,
        features_data,
        image_size=image_size,
        normalize_slope=True,
        slope_scaler=slope_scaler
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_sampler=PatientBatchSampler(
            dataset,
            patients_per_batch=4,
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract slopes
    model.eval()
    patient_slopes = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting slopes"):
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            patient_ids_batch = batch['patient_ids']
            
            predictions = model(images).cpu().numpy()
            
            for pred, pid in zip(predictions, patient_ids_batch):
                patient_slopes[pid].append(pred)
    
    # Average slopes per patient
    mean_slopes = {pid: np.mean(slopes) for pid, slopes in patient_slopes.items()}
    
    return mean_slopes


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_corrector_epoch(model, dataloader, optimizer, criterion, device):
    """Train corrector for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        slopes = batch['slope'].to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, slopes)
        loss.backward()
        optimizer.step()
        
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    
    return total_loss / n_samples if n_samples > 0 else 0.0


def validate_corrector(model, dataloader, device):
    """Validate corrector"""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            slopes = batch['slope'].to(device)
            
            predictions = model(features)
            
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(slopes.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    mse = np.mean((all_preds - all_true) ** 2)
    mae = np.mean(np.abs(all_preds - all_true))
    
    return {
        'mse': mse,
        'mae': mae,
        'predictions': all_preds,
        'true': all_true
    }


def train_corrector_fold(fold_idx, approach_key, approach_config, train_ids, val_ids,
                         patient_data, features_data, train_cnn_slopes, val_cnn_slopes, config):
    """Train one corrector approach for one fold"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{config['n_folds']} - {approach_config['name']}")
    print(f"{'='*80}")
    
    # Create datasets
    train_dataset = CorrectorDataset(
        train_ids,
        patient_data,
        features_data,
        train_cnn_slopes,
        feature_type=approach_config['feature_type']
    )
    
    val_dataset = CorrectorDataset(
        val_ids,
        patient_data,
        features_data,
        val_cnn_slopes,
        feature_type=approach_config['feature_type'],
        scaler=train_dataset.scaler  # Use same scaler
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = approach_config['model_class'](**approach_config['model_kwargs']).to(config['device'])
    
    # CNN-only has no trainable parameters - skip training
    if approach_key == 'cnn_only':
        print("⏭️  CNN-only has no trainable parameters - skipping training")
        
        # Just validate to get metrics
        val_results = validate_corrector(model, val_loader, config['device'])
        
        # Save model (empty state dict) and scaler
        checkpoint_path = config['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        
        scaler_path = config['checkpoint_dir'] / f'{approach_key}_scaler_fold{fold_idx}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.scaler, f)
        
        return {
            'model_state': model.state_dict(),
            'history': {'train_loss': [], 'val_loss': []},
            'best_val_loss': val_results['mse'],
            'final_val_results': val_results,
            'scaler': train_dataset.scaler
        }
    
    # Optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
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
        # Train
        train_loss = train_corrector_epoch(model, train_loader, optimizer, criterion, config['device'])
        history['train_loss'].append(train_loss)
        
        # Validate
        val_results = validate_corrector(model, val_loader, config['device'])
        val_loss = val_results['mse']
        history['val_loss'].append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config['n_epochs']}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = config['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(config['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'))
    
    # Final validation
    final_val_results = validate_corrector(model, val_loader, config['device'])
    
    # Save scaler
    scaler_path = config['checkpoint_dir'] / f'{approach_key}_scaler_fold{fold_idx}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    
    return {
        'model_state': model.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'final_val_results': final_val_results,
        'scaler': train_dataset.scaler
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("SLOPE CORRECTOR TRAINING - 5-FOLD CROSS-VALIDATION")
    print("4 Approaches: CNN-only, CNN+HF, CNN+Demo, CNN+HF+Demo")
    print("="*80)
    
    # Device info
    print(f"\nDevice: {CONFIG['device']}")
    
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
    
    # Load K-Fold splits
    splits_path = CONFIG['results_dir'] / 'kfold_splits.pkl'
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    print(f"✓ Loaded {len(splits)} folds")
    
    # Train each approach for each fold
    all_results = {key: [] for key in APPROACHES.keys()}
    
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold_idx + 1}/{CONFIG['n_folds']}")
        print(f"Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        print(f"{'='*80}")
        
        # Check if all approaches for this fold are already trained
        all_trained = True
        for approach_key in APPROACHES.keys():
            checkpoint_path = CONFIG['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'
            results_path = CONFIG['results_dir'] / f'fold{fold_idx}_{approach_key}_results.pkl'
            if not (checkpoint_path.exists() and results_path.exists()):
                all_trained = False
                break
        
        if all_trained:
            print(f"⏭️  All approaches already trained for this fold - loading results")
            for approach_key in APPROACHES.keys():
                results = load_fold_results(fold_idx, approach_key, CONFIG['results_dir'])
                all_results[approach_key].append(results)
            continue
        
        # Load CNN model for this fold
        cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
        cnn_checkpoint = CONFIG['checkpoint_dir'] / f'cnn_fold{fold_idx}.pth'
        cnn_model.load_state_dict(torch.load(cnn_checkpoint))
        cnn_model = cnn_model.to(CONFIG['device'])
        print(f"✓ Loaded CNN model: {cnn_checkpoint}")
        
        # Load slope scaler
        scaler_path = CONFIG['checkpoint_dir'] / f'slope_scaler_fold{fold_idx}.pkl'
        with open(scaler_path, 'rb') as f:
            slope_scaler = pickle.load(f)
        print(f"✓ Loaded slope scaler")
        
        # Extract CNN slopes for train and val
        train_cnn_slopes = extract_cnn_slopes_for_patients(
            cnn_model, train_ids, patient_data, features_data,
            slope_scaler, CONFIG['device'], CONFIG['image_size']
        )
        
        val_cnn_slopes = extract_cnn_slopes_for_patients(
            cnn_model, val_ids, patient_data, features_data,
            slope_scaler, CONFIG['device'], CONFIG['image_size']
        )
        
        print(f"✓ Extracted CNN slopes: Train={len(train_cnn_slopes)}, Val={len(val_cnn_slopes)}")
        
        # Train each approach
        for approach_key, approach_config in APPROACHES.items():
            # Check if this approach is already trained
            checkpoint_path = CONFIG['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'
            results_path = CONFIG['results_dir'] / f'fold{fold_idx}_{approach_key}_results.pkl'
            
            if checkpoint_path.exists() and results_path.exists():
                print(f"\n⏭️  SKIPPING {approach_config['name']} - Already trained")
                
                # Load existing results
                results = load_fold_results(fold_idx, approach_key, CONFIG['results_dir'])
                all_results[approach_key].append(results)
                continue
            
            results = train_corrector_fold(
                fold_idx,
                approach_key,
                approach_config,
                train_ids,
                val_ids,
                patient_data,
                features_data,
                train_cnn_slopes,
                val_cnn_slopes,
                CONFIG
            )
            
            # Add metadata
            results['train_ids'] = train_ids
            results['val_ids'] = val_ids
            results['train_cnn_slopes'] = train_cnn_slopes
            results['val_cnn_slopes'] = val_cnn_slopes
            
            all_results[approach_key].append(results)
            
            # Save results
            save_fold_results(
                fold_idx,
                approach_key,
                results,
                CONFIG['results_dir']
            )
    
    # Aggregate and compare results
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS - ALL APPROACHES")
    print("="*80)
    
    summary_data = []
    
    for approach_key, approach_config in APPROACHES.items():
        fold_results = all_results[approach_key]
        val_losses = [r['best_val_loss'] for r in fold_results]
        
        print(f"\n{approach_config['name']}:")
        for i, loss in enumerate(val_losses):
            print(f"  Fold {i+1}: MSE = {loss:.6f}")
        print(f"  Mean: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
        
        summary_data.append({
            'Approach': approach_config['name'],
            'Mean_MSE': np.mean(val_losses),
            'Std_MSE': np.std(val_losses),
            'Min_MSE': np.min(val_losses),
            'Max_MSE': np.max(val_losses)
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = CONFIG['results_dir'] / 'corrector_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Plot comparison
    print("\n" + "="*80)
    print("PLOTTING RESULTS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of mean MSE
    ax = axes[0]
    x_pos = np.arange(len(APPROACHES))
    means = [summary_df[summary_df['Approach'] == APPROACHES[k]['name']]['Mean_MSE'].values[0] 
             for k in APPROACHES.keys()]
    stds = [summary_df[summary_df['Approach'] == APPROACHES[k]['name']]['Std_MSE'].values[0] 
            for k in APPROACHES.keys()]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('MSE (Mean ± Std)', fontsize=12)
    ax.set_title('Corrector Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([APPROACHES[k]['name'] for k in APPROACHES.keys()], 
                       rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds) * 0.1,
               f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Box plot of all folds
    ax = axes[1]
    data_for_box = []
    labels_for_box = []
    
    for approach_key, approach_config in APPROACHES.items():
        fold_results = all_results[approach_key]
        val_losses = [r['best_val_loss'] for r in fold_results]
        data_for_box.append(val_losses)
        labels_for_box.append(approach_config['name'])
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE Distribution Across Folds', fontsize=13, fontweight='bold')
    ax.set_xticklabels(labels_for_box, rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = CONFIG['plots_dir'] / 'corrector_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.show()
    
    print("\n" + "="*80)
    print("✅ CORRECTOR TRAINING COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved in: {CONFIG['checkpoint_dir']}")
    print(f"Results saved in: {CONFIG['results_dir']}")
    print(f"Plots saved in: {CONFIG['plots_dir']}")
    
    print("\n📊 SUMMARY:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
