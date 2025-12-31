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
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold

from xgboost.callback import EarlyStopping

def compute_early_fvc_features(patient_data, max_week=12.0):
    """
    Returns dict: pid -> np.array([baseline_fvc, early_slope, early_pct_change])
    Uses only measurements with Weeks <= max_week.
    """
    out = {}
    for pid, pdata in patient_data.items():
        if "weeks" not in pdata or "fvc_values" not in pdata:
            continue

        weeks = np.asarray(pdata["weeks"], dtype=float)
        fvc = np.asarray(pdata["fvc_values"], dtype=float)

        # Keep only early window
        m = weeks <= max_week
        if m.sum() < 2:
            continue

        w = weeks[m]
        y = fvc[m]

        # Baseline FVC = earliest in window (weeks already sorted)
        fvc0 = y[0]

        # Slope (ml/week): fit y = a*w + b
        slope = np.polyfit(w, y, 1)[0]

        # Percent change from baseline to last early point
        pct_change = (y[-1] - fvc0) / fvc0 if fvc0 != 0 else 0.0

        out[pid] = np.array([fvc0, slope, pct_change], dtype=float)

    return out


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

def ds_to_numpy(ds):
    
    X = []
    y = []
    pids = []

    for i in range(len(ds)):
        sample = ds[i]
        x_concat = np.concatenate([sample['x_img'].numpy(),sample['x_hand'].numpy()])
        X.append(x_concat)
        y.append(int(sample['y'].item()))
        pids.append(sample['patient_id'])
        
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y, pids




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
        r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', 
        map_location=torch.device('cpu')
    ))

    # =============================================================================
    # PATHS AND HYPERPARAMETERS
    # =============================================================================
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_PATH_LABEL_52 = 'Training/Progression_prediction_risk/data/patient_progression_52w.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

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

    train_ids = pd.read_csv("Training/Progression_prediction_risk/data/train_patients_52w.csv")['Patient'].tolist()
    val_ids   = pd.read_csv("Training/Progression_prediction_risk/data/val_patients_52w.csv")['Patient'].tolist()
    test_ids  = pd.read_csv("Training/Progression_prediction_risk/data/test_patients_52w.csv")['Patient'].tolist()

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
        feature_stats=feature_stats
    )
    
    val_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=val_ids,
        feature_stats=feature_stats
    )
    
    test_ds = PatientMLPDataset(
        label_csv=CSV_PATH_LABEL_52,
        embeddings_dict=patient_embeddings,
        handcrafted_dict=features_data,
        patient_list=test_ids,
        feature_stats=feature_stats
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
    
    X_train,y_train,pids_train = ds_to_numpy(train_ds)
    X_val,y_val,pids_val = ds_to_numpy(val_ds)
    X_test,y_test,pids_test = ds_to_numpy(test_ds)
    
    # ============================================================
    # STRATIFIED CROSS-VALIDATION (HANDCRAFTED-ONLY)
    # ============================================================

    print("\n" + "="*80)
    print("[XGBOOST] STRATIFIED 5-FOLD CROSS-VALIDATION (HANDCRAFTED ONLY)")
    print("="*80)

    # ------------------------------------------------------------
    # 1. Build FULL dataset for CV (train + val + test)
    # ------------------------------------------------------------
    all_ds = torch.utils.data.ConcatDataset([train_ds, val_ds, test_ds])

    X_all, y_all, pids_all = ds_to_numpy(all_ds)
    
    
    early_fvc_dict = compute_early_fvc_features(patient_data, max_week=12.0)
    keep_idx = [i for i, pid in enumerate(pids_all) if pid in early_fvc_dict]
    


    X_all = X_all[keep_idx]
    y_all = y_all[keep_idx]
    pids_all = [pids_all[i] for i in keep_idx]

    X_hand = X_all[:, -12:]
    X_fvc = np.vstack([early_fvc_dict[pid] for pid in pids_all])

    X_final = np.concatenate([X_hand, X_fvc], axis=1)
    print("Patients with early FVC:", len(early_fvc_dict))
    print("After filtering:", X_final.shape, y_all.shape)
    print("Any NaN in X_final:", np.isnan(X_final).any())
    

    print(f"✓ Total patients for CV: {len(y_all)}")
    print(f"✓ Positive cases: {y_all.sum()}")
    print(f"✓ Feature shape: {X_hand.shape}")

    # ------------------------------------------------------------
    # 2. Stratified K-Fold
    # ------------------------------------------------------------
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []

    # ------------------------------------------------------------
    # 3. CV loop
    # ------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_all), 1):
        X_tr, X_va = X_final[train_idx], X_final[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]

        # class imbalance per fold
        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 3,
            "eta": 0.05,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "seed": 42,
            "nthread": -1,
        }

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        y_va_prob = booster.predict(dval)
        auc = roc_auc_score(y_va, y_va_prob)
        auc_scores.append(auc)

        print(
            f"Fold {fold} | "
            f"AUC = {auc:.3f} | "
            f"best_iter = {booster.best_iteration}"
        )

    # ------------------------------------------------------------
    # 4. CV summary
    # ------------------------------------------------------------
    auc_scores = np.array(auc_scores)

    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(f"Mean AUC: {auc_scores.mean():.3f}")
    print(f"Std  AUC: {auc_scores.std():.3f}")
    print(f"AUCs per fold: {np.round(auc_scores, 3)}")