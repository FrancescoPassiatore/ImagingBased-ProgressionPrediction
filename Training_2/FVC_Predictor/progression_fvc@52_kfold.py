"""
K-Fold Cross-Validation for FVC@52 Prediction
Combines train+val splits and performs k-fold on them
Uses test set separately for final evaluation
"""

import json
import warnings
import pickle
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    CombinedRegressionLoss, 
    IPFDataLoaderPredictorProgression, 
    PatientMLPDataset, 
    SimpleFusionMLP, 
    SliceFeatureDataset, 
    compute_feature_stats, 
    train_model,
    validate,
    ImprovedSliceLevelCNNExtractor,
    print_top_errors
)

HAND_FEATURE_ORDER = [
    'approx_vol',
    'avg_num_tissue_pixel',
    'avg_tissue',
    'avg_tissue_thickness',
    'avg_tissue_by_total',
    'avg_tissue_by_lung',
    'mean',
    'skew',
    'kurtosis',
    'age',
    'sex',
    'smoking_status'
]

NORMALIZE_FEATURES = [
    'approx_vol',
    'avg_num_tissue_pixel',
    'avg_tissue',
    'avg_tissue_thickness',
    'avg_tissue_by_total',
    'avg_tissue_by_lung',
    'mean',
    'skew',
    'kurtosis',
    'age'
]


def collate_fn(batch):
    """Custom collate function to handle dictionary outputs"""
    images = torch.stack([item['image'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return images, patient_ids

def  extract_features_with_cnn(cnn_model, patient_ids, patient_data, device, batch_size=32):
    """Extract features using CNN model"""
    loader = DataLoader(
        SliceFeatureDataset(patient_ids, patient_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    patient_feats = defaultdict(list)
    cnn_model.eval()
    with torch.no_grad():
        for images, pids in tqdm(loader, desc="Extracting features", leave=False):
            images = images.to(device)
            z = cnn_model.extract_features(images)
            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    return patient_embeddings


if __name__ == "__main__":
    
    # Disable warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    print("\n" + "="*80)
    print("FVC@52 PREDICTION - CYCLIC K-FOLD CROSS-VALIDATION")
    print("="*80)

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    N_FOLDS = 5
    EPOCHS = 300
    LR = 1e-4
    WEIGHT_DECAY = 5e-2
    GRAD_CLIP = 1.0
    BATCH_SIZE = 32
    
    # Paths
    KFOLD_SPLITS_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\Kfold_cyclic\kfold_cyclic_splits.pkl'
    CNN_MODELS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\checkpoints_trainings\checkpoints_mse')
    
    RESULTS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\results_cyclic_kfold')
    MODELS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\models_cyclic_kfold')
    PREDICTIONS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\predictions_cyclic_kfold')
    
    # Create directories
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    # =============================================================================
    # STEP 1: LOAD CYCLIC K-FOLD SPLITS
    # =============================================================================
    print("\n[1/5] LOADING CYCLIC K-FOLD SPLITS")
    
    with open(KFOLD_SPLITS_PATH, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"✓ Loaded {len(kfold_splits)} folds")

    # =============================================================================
    # PATHS AND CONFIGURATION
    # =============================================================================
    CSV_PATH = r'Training\CNN_Slope_Prediction\train_with_coefs.csv'
    CSV_PATH_LABEL_52 = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv'
    CSV_FEATURES_PATH = r'Training\CNN_Slope_Prediction\patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")

    # =============================================================================
    # STEP 2: LOAD DATA
    # =============================================================================
    print("\n[2/5] LOADING DATA")
    
    dl = IPFDataLoaderPredictorProgression(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()
    print(f"✓ Loaded {len(patient_data)} patients")


    # =============================================================================
    # STEP 3: CYCLIC K-FOLD TRAINING
    # =============================================================================
    print(f"\n[3/5] RUNNING CYCLIC {N_FOLDS}-FOLD CROSS-VALIDATION")
    
    all_fold_results = []
    all_predictions = {}
    
    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{N_FOLDS - 1}")
        print(f"{'='*80}")
        
        # Get splits for this fold
        split = kfold_splits[fold_idx]
        train_ids = split['train']
        val_ids = split['val']
        test_ids = split['test']
        
        print(f"Train: {len(train_ids)} patients")
        print(f"Val:   {len(val_ids)} patients")
        print(f"Test:  {len(test_ids)} patients")
        
        # =============================================================================
        # LOAD CNN MODEL FOR THIS FOLD
        # =============================================================================
        print(f"\n📦 Loading CNN model for fold {fold_idx}...")
        
        cnn_model_path = CNN_MODELS_DIR / f'cnn_fold{fold_idx}.pt'
        
        if not cnn_model_path.exists():
            print(f"⚠️  CNN model not found at {cnn_model_path}")
            print(f"   Skipping fold {fold_idx}")
            continue
        
        # Load CNN model
        cnn_model = ImprovedSliceLevelCNNExtractor(backbone_name='efficientnet_b1', pretrained=False)
        checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model = cnn_model.to(device)
        print(f"✓ CNN model loaded from {cnn_model_path}")
        
        # =============================================================================
        # EXTRACT FEATURES
        # =============================================================================
        print(f"\n🔍 Extracting features for fold {fold_idx}...")
        
        # Get all unique patient IDs for this fold
        all_fold_ids = list(set(train_ids + val_ids + test_ids))
        
        # Extract features using the fold-specific CNN
        patient_embeddings = extract_features_with_cnn(
            cnn_model, all_fold_ids, patient_data, device, BATCH_SIZE
        )
        print(f"✓ Extracted {len(patient_embeddings)} embeddings")
        
        # Free up memory
        del cnn_model
        torch.cuda.empty_cache()
        
        # =============================================================================
        # COMPUTE NORMALIZATION STATS
        # =============================================================================
        print(f"\n📊 Computing normalization stats...")
        
        # Compute feature stats from TRAIN set only
        feature_stats = compute_feature_stats(
            handcrafted_dict=features_data,
            patient_ids=train_ids,
            feature_names=NORMALIZE_FEATURES
        )
        
        # Compute FVC normalization from TRAIN set only
        label_df = pd.read_csv(CSV_PATH_LABEL_52)
        train_fvc_values = label_df[label_df['Patient'].isin(train_ids)]['fvc_52'].values
        fvc_mean = float(np.mean(train_fvc_values))
        fvc_std = float(np.std(train_fvc_values))
        if fvc_std < 1e-6:
            fvc_std = 1.0
        fvc_norm_stats = {'mean': fvc_mean, 'std': fvc_std}
        print(f"✓ FVC norm stats: mean={fvc_mean:.2f}, std={fvc_std:.2f}")
        
        # =============================================================================
        # CREATE DATASETS
        # =============================================================================
        print(f"\n📁 Creating datasets...")
        
        train_ds = PatientMLPDataset(
            label_csv=CSV_PATH_LABEL_52,
            embeddings_dict=patient_embeddings,
            handcrafted_dict=features_data,
            patient_list=train_ids,
            feature_stats=feature_stats,
            fvc_norm_stats=fvc_norm_stats
        )
        
        val_ds = PatientMLPDataset(
            label_csv=CSV_PATH_LABEL_52,
            embeddings_dict=patient_embeddings,
            handcrafted_dict=features_data,
            patient_list=val_ids,
            feature_stats=feature_stats,
            fvc_norm_stats=fvc_norm_stats
        )
        
        test_ds = PatientMLPDataset(
            label_csv=CSV_PATH_LABEL_52,
            embeddings_dict=patient_embeddings,
            handcrafted_dict=features_data,
            patient_list=test_ids,
            feature_stats=feature_stats,
            fvc_norm_stats=fvc_norm_stats
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        print(f"✓ Datasets created")
        
        # =============================================================================
        # CREATE AND TRAIN MODEL
        # =============================================================================
        print(f"\n Creating fusion model...")

        #Determine actual feature dimension from dataset
        sample_batch = next(iter(train_loader))
        actual_hand_dim = sample_batch['x_hand'].shape[1]
        print(f"✓ Detected hand features dimension: {actual_hand_dim}")
        
        model = SimpleFusionMLP(
            img_dim=320,  # CNN embedding dimension
            hand_dim=actual_hand_dim,  
            hidden=32,
            dropout=0.5
        ).to(device)
        
        # Train model
        print(f"\n Training model for fold {fold_idx}...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            grad_clip=GRAD_CLIP,
            use_class_weights=False,
            device=device,
            fvc_norm_stats=fvc_norm_stats
        )
        
        # Save best model for this fold
        model_path = MODELS_DIR / f'model_fold{fold_idx}_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f"✓ Saved model to {model_path}")
        
        # Load best model
        model.load_state_dict(torch.load('best_fusion_mlp.pth'))
        
        # =============================================================================
        # EVALUATE ON ALL SPLITS
        # =============================================================================
        print(f"\n📊 Evaluating on all splits...")
        
        criterion = CombinedRegressionLoss(alpha=0.7)
        
        # Train predictions
        train_metrics, train_preds, train_labels, train_pids = validate(
            model, train_loader, criterion, device
        )
        train_preds_denorm = np.array(train_preds) * fvc_std + fvc_mean
        train_labels_denorm = np.array(train_labels) * fvc_std + fvc_mean
        
        # Val predictions
        val_metrics, val_preds, val_labels, val_pids = validate(
            model, val_loader, criterion, device
        )
        val_preds_denorm = np.array(val_preds) * fvc_std + fvc_mean
        val_labels_denorm = np.array(val_labels) * fvc_std + fvc_mean
        
        # Test predictions
        test_metrics, test_preds, test_labels, test_pids = validate(
            model, test_loader, criterion, device
        )
        test_preds_denorm = np.array(test_preds) * fvc_std + fvc_mean
        test_labels_denorm = np.array(test_labels) * fvc_std + fvc_mean
        
        # Compute metrics
        train_mae = mean_absolute_error(train_labels_denorm, train_preds_denorm)
        train_rmse = np.sqrt(mean_squared_error(train_labels_denorm, train_preds_denorm))
        train_r2 = r2_score(train_labels_denorm, train_preds_denorm)
        
        val_mae = mean_absolute_error(val_labels_denorm, val_preds_denorm)
        val_rmse = np.sqrt(mean_squared_error(val_labels_denorm, val_preds_denorm))
        val_r2 = r2_score(val_labels_denorm, val_preds_denorm)
        
        test_mae = mean_absolute_error(test_labels_denorm, test_preds_denorm)
        test_rmse = np.sqrt(mean_squared_error(test_labels_denorm, test_preds_denorm))
        test_r2 = r2_score(test_labels_denorm, test_preds_denorm)
        
        # Store results
        fold_result = {
            'fold': fold_idx,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
        }
        all_fold_results.append(fold_result)
        
        # Store predictions
        all_predictions[f'fold_{fold_idx}'] = {
            'train': {pid: pred for pid, pred in zip(train_pids, train_preds_denorm)},
            'val': {pid: pred for pid, pred in zip(val_pids, val_preds_denorm)},
            'test': {pid: pred for pid, pred in zip(test_pids, test_preds_denorm)}
        }
        
        # Print fold results
        print(f"\n✓ FOLD {fold_idx} RESULTS:")
        print(f"   Train - MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
        print(f"   Val   - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}")
        print(f"   Test  - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    # =============================================================================
    # STEP 5: SAVE RESULTS
    # =============================================================================
    print("\n" + "="*80)
    print("[5/5] SAVING RESULTS")
    print("="*80)
    
    # Save fold results
    results_df = pd.DataFrame(all_fold_results)
    results_path = RESULTS_DIR / 'cyclic_kfold_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved results to {results_path}")
    
    # Save predictions
    predictions_path = PREDICTIONS_DIR / 'all_predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump(all_predictions, f)
    print(f"✓ Saved predictions to {predictions_path}")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY - MEAN ± STD ACROSS FOLDS")
    print("="*80)
    
    print(f"\n📊 TRAIN SET:")
    print(f"   MAE:  {results_df['train_mae'].mean():.2f} ± {results_df['train_mae'].std():.2f} mL")
    print(f"   RMSE: {results_df['train_rmse'].mean():.2f} ± {results_df['train_rmse'].std():.2f} mL")
    print(f"   R²:   {results_df['train_r2'].mean():.4f} ± {results_df['train_r2'].std():.4f}")
    
    print(f"\n📊 VALIDATION SET:")
    print(f"   MAE:  {results_df['val_mae'].mean():.2f} ± {results_df['val_mae'].std():.2f} mL")
    print(f"   RMSE: {results_df['val_rmse'].mean():.2f} ± {results_df['val_rmse'].std():.2f} mL")
    print(f"   R²:   {results_df['val_r2'].mean():.4f} ± {results_df['val_r2'].std():.4f}")
    
    print(f"\n📊 TEST SET:")
    print(f"   MAE:  {results_df['test_mae'].mean():.2f} ± {results_df['test_mae'].std():.2f} mL")
    print(f"   RMSE: {results_df['test_rmse'].mean():.2f} ± {results_df['test_rmse'].std():.2f} mL")
    print(f"   R²:   {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
    
    print("\n✅ CYCLIC K-FOLD CROSS-VALIDATION COMPLETE!")
