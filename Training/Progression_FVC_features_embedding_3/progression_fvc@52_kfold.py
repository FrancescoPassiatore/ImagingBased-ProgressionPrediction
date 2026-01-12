"""
K-Fold Cross-Validation for FVC@52 Prediction
Combines train+val splits and performs k-fold on them
Uses test set separately for final evaluation
"""

import json
from utilities import *
import warnings
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
    'smoking_status',
    'fvc_baseline'
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
    'age',
    'fvc_baseline'
]


def collate_fn(batch):
    """Custom collate function to handle dictionary outputs"""
    images = torch.stack([item['image'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return images, patient_ids


if __name__ == "__main__":
    
    # Disable warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    print("\n" + "="*80)
    print("FVC@52 PREDICTION - K-FOLD CROSS-VALIDATION")
    print("="*80)

    # =============================================================================
    # STEP 1: LOAD CNN MODEL
    # =============================================================================
    print("\n[1/7] LOADING CNN MODEL")
    
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(
        r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', 
        map_location=torch.device('cpu')
    ))
    print("✓ CNN model loaded")

    # =============================================================================
    # PATHS AND HYPERPARAMETERS
    # =============================================================================
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_PATH_LABEL_52 = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    cnn_model = cnn_model.to(device)

    # =============================================================================
    # STEP 2: LOAD SPLITS
    # =============================================================================
    print("\n[2/7] LOADING DATA SPLITS")
    
    train_ids = pd.read_csv("Training/Progression_prediction_risk_2/data/train_patients_52w.csv")['Patient'].tolist()
    val_ids = pd.read_csv("Training/Progression_prediction_risk_2/data/val_patients_52w.csv")['Patient'].tolist()
    test_ids = pd.read_csv("Training/Progression_prediction_risk_2/data/test_patients_52w.csv")['Patient'].tolist()
    
    # Combine train + val for k-fold
    kfold_ids = train_ids + val_ids
    
    print(f"✓ Train+Val (for k-fold): {len(kfold_ids)} patients")
    print(f"✓ Test (holdout): {len(test_ids)} patients")

    # =============================================================================
    # STEP 3: LOAD DATA
    # =============================================================================
    print("\n[3/7] LOADING DATA")
    
    dl = IPFDataLoaderPredictorProgression(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()
    print(f"✓ Loaded {len(patient_data)} patients")

    # =============================================================================
    # STEP 4: LOAD FVC BASELINE
    # =============================================================================
    print("\n[4/7] LOADING FVC BASELINE")
    
    train_csv_path = 'Training/CNN_Slope_Prediction/train.csv'
    train_df = pd.read_csv(train_csv_path)
    
    baseline_fvc = {}
    for patient_id in patient_data.keys():
        patient_weeks = train_df[train_df['Patient'] == patient_id]['Weeks'].values
        if len(patient_weeks) > 0:
            earliest_week_idx = np.argmin(patient_weeks)
            baseline_fvc_val = train_df[train_df['Patient'] == patient_id].iloc[earliest_week_idx]['FVC']
            baseline_fvc[patient_id] = baseline_fvc_val
        else:
            baseline_fvc[patient_id] = 0.0
    
    for patient_id in features_data.keys():
        features_data[patient_id]['fvc_baseline'] = baseline_fvc.get(patient_id, 0.0)
    
    print(f"✓ FVC Baseline loaded for {len(baseline_fvc)} patients")

    # =============================================================================
    # STEP 5: EXTRACT FEATURES FROM SLICES
    # =============================================================================
    print("\n[5/7] EXTRACTING CNN FEATURES")
    
    all_patient_ids = list(patient_data.keys())
    loader = DataLoader(
        SliceFeatureDataset(all_patient_ids, patient_data),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    patient_feats = defaultdict(list)
    cnn_model.eval()
    with torch.no_grad():
        for images, pids in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            z = cnn_model.extract_features(images)
            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    print(f"✓ Extracted {len(patient_embeddings)} embeddings")

    # =============================================================================
    # STEP 6: COMPUTE NORMALIZATION STATS
    # =============================================================================
    print("\n[6/7] COMPUTING NORMALIZATION STATS")
    
    # For k-fold: compute stats from KFOLD_IDS only (not test)
    feature_stats = compute_feature_stats(
        handcrafted_dict=features_data,
        patient_ids=kfold_ids,
        feature_names=NORMALIZE_FEATURES
    )
    print(f"✓ Feature stats computed from {len(kfold_ids)} kfold patients")

    # =============================================================================
    # STEP 7: K-FOLD CROSS-VALIDATION
    # =============================================================================
    print("\n[7/7] RUNNING K-FOLD CROSS-VALIDATION")
    
    kfold_results = train_kfold(
        model_class=SimpleFusionMLP,
        patient_ids=kfold_ids,
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        feature_stats=feature_stats,
        n_splits=5,
        epochs=100,
        lr=1e-4,
        weight_decay=5e-2,
        grad_clip=1.0,
        device=device
    )

    # =============================================================================
    # FINAL EVALUATION ON HOLDOUT TEST SET
    # =============================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON HOLDOUT TEST SET")
    print("="*80)
    
    # Retrain on full kfold data, then evaluate on test
    print("\nRetraining on full k-fold set for test evaluation...")
    
    label_df = pd.read_csv(CSV_PATH_LABEL_52)
    all_fvc_values = label_df[label_df['Patient'].isin(kfold_ids)]['fvc_52'].values
    fvc_mean = float(np.mean(all_fvc_values))
    fvc_std = float(np.std(all_fvc_values))
    if fvc_std < 1e-6:
        fvc_std = 1.0
    fvc_norm_stats = {'mean': fvc_mean, 'std': fvc_std}
    
    # Create datasets
    final_train_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=kfold_ids,
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
    
    final_train_loader = DataLoader(final_train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # Train final model
    final_model = SimpleFusionMLP(
        img_dim=320,
        hand_dim=13,
        hidden=32,
        dropout=0.5
    ).to(device)
    
    # Create a dummy val_loader (won't be used for saving, just for training)
    history = train_model(
        model=final_model,
        train_loader=final_train_loader,
        val_loader=test_loader,  # Use test loader for validation during training
        epochs=100,
        lr=1e-4,
        weight_decay=5e-2,
        grad_clip=1.0,
        use_class_weights=False,
        device=device,
        fvc_norm_stats=fvc_norm_stats
    )
    
    # Load best model
    final_model.load_state_dict(torch.load('best_fusion_mlp.pth'))
    
    # Evaluate on test set
    criterion = CombinedRegressionLoss(alpha=0.7)
    test_metrics, test_preds, test_labels, test_pids = validate(
        final_model, test_loader, criterion, device
    )
    
    # Denormalize
    test_preds_denorm = np.array(test_preds) * fvc_std + fvc_mean
    test_labels_denorm = np.array(test_labels) * fvc_std + fvc_mean
    
    # Compute metrics
    test_mae = mean_absolute_error(test_labels_denorm, test_preds_denorm)
    test_rmse = np.sqrt(mean_squared_error(test_labels_denorm, test_preds_denorm))
    test_r2 = r2_score(test_labels_denorm, test_preds_denorm)
    
    print(f"\n📊 TEST SET RESULTS [DENORMALIZED]:")
    print(f"   MAE:  {test_mae:.2f} mL")
    print(f"   RMSE: {test_rmse:.2f} mL")
    print(f"   R²:   {test_r2:.4f}")
    
    # Save test predictions
    test_df = pd.DataFrame({
        'patient_id': test_pids,
        'true_fvc': test_labels_denorm,
        'pred_fvc': test_preds_denorm,
        'error': test_labels_denorm - test_preds_denorm,
        'error_percent': ((test_labels_denorm - test_preds_denorm) / test_labels_denorm) * 100
    })
    test_df.to_csv('test_predictions_final.csv', index=False)
    print(f"\n✅ Test predictions saved to 'test_predictions_final.csv'")
    
    # Print top errors
    print_top_errors(test_preds_denorm, test_labels_denorm, test_pids, n=5,
                    title="Test Set - Top 5 Worst Predictions")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n📊 K-Fold Cross-Validation ({5} folds):")
    print(f"   MAE:  {kfold_results['mean_mae']:.2f} ± {kfold_results['std_mae']:.2f} mL")
    print(f"   RMSE: {kfold_results['mean_rmse']:.2f} ± {kfold_results['std_rmse']:.2f} mL")
    print(f"   R²:   {kfold_results['mean_r2']:.4f} ± {kfold_results['std_r2']:.4f}")
    
    print(f"\n📊 Final Test Set Evaluation:")
    print(f"   MAE:  {test_mae:.2f} mL")
    print(f"   RMSE: {test_rmse:.2f} mL")
    print(f"   R²:   {test_r2:.4f}")
