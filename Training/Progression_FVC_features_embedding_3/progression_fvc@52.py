import json
from utilities import *
import warnings
import pickle
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    """Custom collate function to handle dictionary outputs"""
    images = torch.stack([item['image'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    return images, patient_ids

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
    'fvc_baseline'  # ⭐ CRITICAL: Add baseline FVC as 13th feature
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
    'fvc_baseline'  # ⭐ CRITICAL: Normalize baseline FVC too
]



if __name__ == "__main__":
    
    # Disable warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    # =============================================================================
    # STEP 1: LOAD CNN MODEL
    # =============================================================================
    print("\n" + "="*80)
    print("[1/10] LOADING CNN MODEL")
    print("="*80)
    
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(
        r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', 
        map_location=torch.device('cpu')
    ))

    # =============================================================================
    # PATHS AND HYPERPARAMETERS
    # =============================================================================
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_PATH_LABEL_52 = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    IMAGE_SIZE = (224, 224)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    # Move model to device
    cnn_model = cnn_model.to(device)

    # =============================================================================
    # STEP 2: TRAIN/VAL/TEST SPLIT
    # =============================================================================
    print("\n" + "="*80)
    print("[3/10] LOADING TRAIN/VAL/TEST SPLITS")
    print("="*80)

    train_ids = pd.read_csv("Training/Progression_prediction_risk_2/data/train_patients_52w.csv")['Patient'].tolist()
    val_ids   = pd.read_csv("Training/Progression_prediction_risk_2/data/val_patients_52w.csv")['Patient'].tolist()
    test_ids  = pd.read_csv("Training/Progression_prediction_risk_2/data/test_patients_52w.csv")['Patient'].tolist()

    print(f"✓ Train patients: {len(train_ids)}")
    print(f"✓ Val patients: {len(val_ids)}")
    print(f"✓ Test patients: {len(test_ids)}")

    # =============================================================================
    # STEP 3: LOAD DATA
    # =============================================================================
    print("\n" + "="*80)
    print("[2/10] LOADING DATA")
    print("="*80)

    dl = IPFDataLoaderPredictorProgression(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()

    print(f"\n✓ Loaded patient_data for {len(patient_data)} patients")
    print(f"✓ Loaded features_data for {len(features_data)} patients")

    # Verify data structure
    sample_patient = list(patient_data.keys())[0]
    print(f"\n📋 Sample patient data structure (ID: {sample_patient}):")
    for key, value in patient_data[sample_patient].items():
        if isinstance(value, list):
            print(f"   {key}: list with {len(value)} items")
        else:
            print(f"   {key}: {type(value).__name__} = {value}")

    print(f"\n📋 Sample feature data structure:")
    for key, value in features_data[sample_patient].items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    # =============================================================================
    # LOAD FVC BASELINE (CRITICAL FEATURE!)
    # =============================================================================
    print("\n" + "="*80)
    print("LOADING FVC BASELINE FROM TRAIN.CSV")
    print("="*80)
    
    # Load train.csv to get baseline FVC
    train_csv_path = 'Training/CNN_Slope_Prediction/train.csv'
    train_df = pd.read_csv(train_csv_path)
    
    # Get baseline FVC (earliest measurement) for each patient
    baseline_fvc = {}
    for patient_id in patient_data.keys():
        patient_weeks = train_df[train_df['Patient'] == patient_id]['Weeks'].values
        if len(patient_weeks) > 0:
            earliest_week_idx = np.argmin(patient_weeks)
            baseline_fvc_val = train_df[train_df['Patient'] == patient_id].iloc[earliest_week_idx]['FVC']
            baseline_fvc[patient_id] = baseline_fvc_val
        else:
            baseline_fvc[patient_id] = 0.0  # Fallback
    
    print(f"✓ Loaded baseline FVC for {len(baseline_fvc)} patients")
    print(f"  Baseline FVC range: {np.min(list(baseline_fvc.values())):.0f} - {np.max(list(baseline_fvc.values())):.0f} mL")
    
    # Add baseline FVC to features_data
    for patient_id in features_data.keys():
        features_data[patient_id]['fvc_baseline'] = baseline_fvc.get(patient_id, 0.0)
    
    print("✓ Added fvc_baseline to all patient features")

    # =============================================================================
    # STEP 4: EXTRACT FEATURES FROM SLICES
    # =============================================================================
    print("\n" + "="*80)
    print("[4/10] EXTRACTING FEATURES FROM SLICES")
    print("="*80)
    

    # Get all patient IDs (you might want to filter to train+val+test only)
    all_patient_ids = list(patient_data.keys())
    
    loader = DataLoader(
        SliceFeatureDataset(all_patient_ids, patient_data),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn  # ADD CUSTOM COLLATE FUNCTION
    )

    patient_feats = defaultdict(list)

    cnn_model.eval()
    with torch.no_grad():
        for images, pids in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract features
            z = cnn_model.extract_features(images)
            
            # Store features per patient
            for i, pid in enumerate(pids):
                patient_feats[pid].append(z[i].cpu().numpy())

    # Average features across slices for each patient
    patient_embeddings = {
        pid: np.mean(v, axis=0)
        for pid, v in patient_feats.items()
    }
    
    print(f"\n✓ Extracted embeddings for {len(patient_embeddings)} patients")
    sample_emb = list(patient_embeddings.values())[0]
    print(f"✓ Embedding shape: {sample_emb.shape}")

    # =============================================================================
    # COMPUTE NORMALIZATION STATS (TRAIN ONLY)
    # =============================================================================

    feature_stats = compute_feature_stats(
        handcrafted_dict=features_data,
        patient_ids=train_ids,
        feature_names=NORMALIZE_FEATURES
    )

    print("\n📊 Feature normalization stats (TRAIN only):")
    for k, v in feature_stats.items():
        print(f"   {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")

    # Compute FVC normalization stats from TRAINING SET ONLY
    print("\n" + "="*80)
    print("[FVC Stats] COMPUTING FVC NORMALIZATION STATS (TRAIN ONLY)")
    print("="*80)
    
    label_df = pd.read_csv(CSV_PATH_LABEL_52)
    
    # Get FVC values only from training patients
    train_fvc_values = label_df[label_df['Patient'].isin(train_ids)]['fvc_52'].values
    
    fvc_mean = float(np.mean(train_fvc_values))
    fvc_std = float(np.std(train_fvc_values))
    
    # Handle edge case
    if fvc_std < 1e-6:
        fvc_std = 1.0
    
    fvc_norm_stats = {'mean': fvc_mean, 'std': fvc_std}
    
    print(f"✓ FVC Statistics (from {len(train_fvc_values)} TRAIN patients):")
    print(f"  Mean: {fvc_mean:.2f} mL")
    print(f"  Std: {fvc_std:.2f} mL")
    print(f"  Range: {train_fvc_values.min():.2f} - {train_fvc_values.max():.2f} mL")
    print("="*80)

    # =============================================================================
    # STEP 5: CREATE MLP DATASETS
    # =============================================================================
    print("\n" + "="*80)
    print("[5/10] CREATING MLP DATASETS")
    print("="*80)

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
    
    print(f"✓ Train dataset: {len(train_ds)} patients")
    print(f"✓ Val dataset: {len(val_ds)} patients")
    print(f"✓ Test dataset: {len(test_ds)} patients")
    
    # Test dataset
    print("\n📋 Sample from train dataset:")
    sample = train_ds[0]
    print(f"   x_img shape: {sample['x_img'].shape}")
    print(f"   x_hand shape: {sample['x_hand'].shape}")
    print(f"   y: {sample['y'].item()}")
    print(f"   patient_id: {sample['patient_id']}")

    sample = train_ds[0]['x_hand']
    print("Mean:", sample.mean().item())
    print("Std:", sample.std().item())
    
    print("\n✅ Data preparation complete!")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Get embedding dimension from a sample
    sample = train_ds[0]
    img_dim = sample['x_img'].shape[0]
    hand_dim = sample['x_hand'].shape[0]
    
    print(f"\n📊 Feature dimensions:")
    print(f"   Image embedding: {img_dim}")
    print(f"   Handcrafted features: {hand_dim}")
    

    # =============================================================================
    # MLP MODEL 
    # =============================================================================
    print("\n" + "="*80)
    print("[6/10] INITIALIZING AND TRAINING MLP MODEL")
    print("="*80)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = SimpleFusionMLP(
        img_dim=320,
        hand_dim=13,  # 12 original + 1 (fvc_baseline) ⭐
        hidden=32,       
        dropout=0.5         
    ).to('cuda')
    
    print(f"\n✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,  # Your train DataLoader with batch_size=32
        val_loader=val_loader,      # Your validation DataLoader
        epochs=100,
        lr=1e-4,
        weight_decay=5e-2,
        grad_clip=1.0,
        use_class_weights=False,
        device='cuda',
        fvc_norm_stats=fvc_norm_stats  # Pass the computed FVC normalization stats
    )

    
    
    

    # Plotta la storia del training
    plot_training_history(history, save_path='training_history.png')


    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_fusion_mlp.pth'))
    
    criterion = CombinedRegressionLoss(alpha=0.7)
    test_metrics, test_preds, test_labels, test_pids = validate(
        model, test_loader, criterion, device
    )

    # Denormalize test predictions
    test_preds_denorm = np.array(test_preds) * fvc_norm_stats['std'] + fvc_norm_stats['mean']
    test_labels_denorm = np.array(test_labels) * fvc_norm_stats['std'] + fvc_norm_stats['mean']
    
    # Compute denormalized metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    test_mae_denorm = mean_absolute_error(test_labels_denorm, test_preds_denorm)
    test_rmse_denorm = np.sqrt(mean_squared_error(test_labels_denorm, test_preds_denorm))
    test_r2_denorm = r2_score(test_labels_denorm, test_preds_denorm)

    print(f"\n📊 Test Set Results [DENORMALIZED - in mL]:")
    print(f"   MAE: {test_mae_denorm:.2f} mL")
    print(f"   RMSE: {test_rmse_denorm:.2f} mL")
    print(f"   R²: {test_r2_denorm:.4f}")
    
    print(f"\n📊 Test Set Results [NORMALIZED - for reference]:")
    print(f"   MAE: {test_metrics['mae']:.4f}")
    print(f"   RMSE: {test_metrics['rmse']:.4f}")
    print(f"   R²: {test_metrics['r2']:.4f}")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': test_pids,
        'true_fvc_denorm': test_labels_denorm,
        'pred_fvc_denorm': test_preds_denorm,
        'true_fvc_norm': test_labels,
        'pred_fvc_norm': test_preds,
        'error_mL': test_labels_denorm - test_preds_denorm,
        'error_percent': ((test_labels_denorm - test_preds_denorm) / test_labels_denorm) * 100
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("\n✅ Predictions saved to 'test_predictions.csv'")
    
    # Print top errors
    print_top_errors(test_preds_denorm, test_labels_denorm, test_pids, n=5,
                    title="Test Set - Top 5 Worst Predictions [DENORMALIZED]")


    
