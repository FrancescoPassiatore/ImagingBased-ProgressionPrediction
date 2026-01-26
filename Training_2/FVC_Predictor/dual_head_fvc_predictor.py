import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utilities import(ImprovedSliceLevelCNNExtractor,SliceFeatureDataset,IPFDataLoaderPredictorProgression,compute_feature_stats,PatientMLPDataset)

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


class DualHeadPatientMLPDataset(Dataset):
    """Dataset for dual-head FVC prediction with ΔFVC targets"""
    
    def __init__(self, label_csv, embeddings_dict, handcrafted_dict, 
                 patient_list, feature_stats=None, fvc_norm_stats=None):
        """
        Args:
            label_csv: Path to CSV with FVC@52 labels
            embeddings_dict: CNN embeddings
            handcrafted_dict: Handcrafted features
            patient_list: Patient IDs
            feature_stats: Normalization stats for features
            fvc_norm_stats: Normalization stats for FVC values
        """
        self.df = pd.read_csv(label_csv)
        self.df = self.df[self.df['Patient'].isin(patient_list)].reset_index(drop=True)
        
        self.embeddings = embeddings_dict
        self.handcrafted = handcrafted_dict
        
        # Filter valid patients
        valid_patients = []
        for pid in self.df['Patient']:
            if pid in self.embeddings and pid in self.handcrafted:
                valid_patients.append(pid)
        
        self.df = self.df[self.df['Patient'].isin(valid_patients)].reset_index(drop=True)
        
        self.feature_stats = feature_stats
        self.fvc_norm_stats = fvc_norm_stats or {'mean': 0, 'std': 1}
        
        # Load baseline FVC for ΔFVC computation
        self._load_baseline_fvc()
        
        print(f"✓ DualHeadDataset: {len(self.df)} patients loaded")
    
    def _load_baseline_fvc(self):
        """Load baseline FVC values for each patient"""
        train_csv_path = 'Training/CNN_Slope_Prediction/train.csv'
        train_df = pd.read_csv(train_csv_path)
        
        self.baseline_fvc = {}
        for patient_id in train_df['Patient'].unique():
            patient_weeks = train_df[train_df['Patient'] == patient_id]['Weeks'].values
            if len(patient_weeks) > 0:
                earliest_idx = np.argmin(patient_weeks)
                baseline_val = train_df[train_df['Patient'] == patient_id].iloc[earliest_idx]['FVC']
                self.baseline_fvc[patient_id] = baseline_val
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['Patient']
        
        # Normalize handcrafted features
        raw_feats = self.handcrafted[pid]
        norm_feats = []
        
        for k in HAND_FEATURE_ORDER:
            v = raw_feats[k]
            if k in self.feature_stats:
                mean = self.feature_stats[k]['mean']
                std = self.feature_stats[k]['std']
                norm_feats.append((v - mean) / std)
            else:
                norm_feats.append(float(v))
        
        # Get FVC values
        fvc52_true = row['fvc_52']
        baseline_fvc = self.baseline_fvc.get(pid, fvc52_true)  # Fallback if missing
        
        # Compute ΔFVC target
        delta_fvc_true = fvc52_true - baseline_fvc
        
        # Normalize targets
        fvc52_norm = (fvc52_true - self.fvc_norm_stats['mean']) / self.fvc_norm_stats['std']
        delta_fvc_norm = delta_fvc_true / self.fvc_norm_stats['std']  # Only scale, don't shift
        
        x_img = torch.FloatTensor(self.embeddings[pid])
        x_hand = torch.FloatTensor(norm_feats)
        
        return {
            'x_img': x_img,
            'x_hand': x_hand,
            'y_fvc52': torch.FloatTensor([fvc52_norm]),      # Target for reconstruction
            'y_delta_fvc': torch.FloatTensor([delta_fvc_norm]), # Target for Head B
            'patient_id': pid,
            'baseline_fvc': baseline_fvc  # For denormalization later
        }

class DualHeadLoss(nn.Module):
    """
    Composite loss for dual-head architecture
    
    Loss = α * L_reconstruction + β * L_delta + γ * L_proxy
    
    Where:
        - L_reconstruction: MSE(FVC52_pred, FVC52_true)
        - L_delta: MSE(ΔFVC_pred, ΔFVC_true)
        - L_proxy: Optional regularization on FVC_proxy
    """
    
    def __init__(self, alpha=0.6, beta=0.4, gamma=0.0):
        """
        Args:
            alpha: Weight for reconstruction loss
            beta: Weight for delta FVC loss
            gamma: Weight for proxy regularization (usually 0)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        print(f"\n✓ DualHeadLoss initialized:")
        print(f"   α (reconstruction): {alpha}")
        print(f"   β (delta FVC): {beta}")
        print(f"   γ (proxy reg): {gamma}")
    
    def forward(self, fvc_proxy, delta_fvc, fvc52_pred, y_fvc52, y_delta_fvc):
        """
        Compute composite loss
        
        Args:
            fvc_proxy: [B, 1] - Predicted FVC proxy (Head A)
            delta_fvc: [B, 1] - Predicted ΔFVC (Head B)
            fvc52_pred: [B, 1] - Reconstructed FVC@52 (proxy + delta)
            y_fvc52: [B, 1] - True FVC@52 (normalized)
            y_delta_fvc: [B, 1] - True ΔFVC (normalized)
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dictionary with individual losses
        """
        # Reconstruction loss (main supervision)
        loss_recon = F.mse_loss(fvc52_pred, y_fvc52)
        
        # Delta FVC loss (direct supervision for Head B)
        loss_delta = F.mse_loss(delta_fvc, y_delta_fvc)
        
        # Optional: Regularize FVC proxy to be reasonable
        # (Can add bounds or stats-based loss here if needed)
        loss_proxy = torch.tensor(0.0, device=fvc_proxy.device)
        
        # Total loss
        total_loss = (
            self.alpha * loss_recon +
            self.beta * loss_delta +
            self.gamma * loss_proxy
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'delta': loss_delta.item(),
            'proxy': loss_proxy.item()
        }
        
        return total_loss, loss_dict

class DualHeadFVCPredictor(nn.Module):
    """
    Dual-head architecture for FVC@52 prediction
    
    Architecture:
        CT embeddings + demographics → shared embedding z
        
        Head A (FVC proxy): z → absolute lung capacity estimate
        Head B (ΔFVC): z → predicted change from baseline
        
        Final prediction: FVC@52 = FVC_proxy + ΔFVC
    
    Training:
        - ΔFVC_true = FVC@52_true - baseline_FVC (computed from labels)
        - Head B supervised with ΔFVC_true
        - Head A supervised indirectly via reconstruction loss
        - Final loss combines both heads
    """
    
    def __init__(self, img_dim=320, hand_dim=12, shared_hidden=128, 
                 head_hidden=64, dropout=0.3):
        """
        Args:
            img_dim: CNN embedding dimension
            hand_dim: Number of handcrafted features
            shared_hidden: Hidden dimension for shared representation
            head_hidden: Hidden dimension for each head
            dropout: Dropout rate
        """
        super().__init__()
        
        # =====================================================================
        # SHARED ENCODER: CT + Demographics → z
        # =====================================================================
        self.shared_encoder = nn.Sequential(
            # Combine image and handcrafted features
            nn.Linear(img_dim + hand_dim, shared_hidden * 2),
            nn.LayerNorm(shared_hidden * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(shared_hidden * 2, shared_hidden),
            nn.LayerNorm(shared_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # =====================================================================
        # HEAD A: FVC Proxy (absolute lung capacity)
        # =====================================================================
        self.fvc_proxy_head = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 2),  # Less dropout for stable predictions
            
            nn.Linear(head_hidden, head_hidden // 2),
            nn.LeakyReLU(),
            
            nn.Linear(head_hidden // 2, 1)
        )
        
        # =====================================================================
        # HEAD B: ΔFVC (change from baseline)
        # =====================================================================
        self.delta_fvc_head = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(head_hidden, head_hidden // 2),
            nn.LeakyReLU(),
            
            nn.Linear(head_hidden // 2, 1)
        )
    
    def forward(self, img_emb, hand_feat, return_components=False):
        """
        Forward pass
        
        Args:
            img_emb: [B, img_dim] - CNN embeddings
            hand_feat: [B, hand_dim] - Handcrafted features
            return_components: If True, return (fvc_proxy, delta_fvc, fvc52_pred)
        
        Returns:
            fvc52_pred: [B, 1] - Predicted FVC@52
            or tuple if return_components=True
        """
        # Concatenate features
        x = torch.cat([img_emb, hand_feat], dim=1)  # [B, img_dim + hand_dim]
        
        # Shared embedding
        z = self.shared_encoder(x)  # [B, shared_hidden]
        
        # Head A: FVC proxy (absolute scale)
        fvc_proxy = self.fvc_proxy_head(z)  # [B, 1]
        
        # Head B: ΔFVC (change from baseline)
        delta_fvc = self.delta_fvc_head(z)  # [B, 1]
        
        # Reconstruction
        fvc52_pred = fvc_proxy + delta_fvc  # [B, 1]
        
        if return_components:
            return fvc_proxy, delta_fvc, fvc52_pred
        
        return fvc52_pred
    


def train_epoch_dual_head(model, loader, criterion, optimizer, device, grad_clip=0.5):
    """Train dual-head model for one epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'recon': 0.0,
        'delta': 0.0,
        'proxy': 0.0
    }
    
    all_preds_fvc52 = []
    all_labels_fvc52 = []
    all_preds_delta = []
    all_labels_delta = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x_img = batch['x_img'].to(device)
        x_hand = batch['x_hand'].to(device)
        y_fvc52 = batch['y_fvc52'].to(device)
        y_delta_fvc = batch['y_delta_fvc'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with component outputs
        fvc_proxy, delta_fvc, fvc52_pred = model(x_img, x_hand, return_components=True)
        
        # Compute composite loss
        loss, loss_dict = criterion(fvc_proxy, delta_fvc, fvc52_pred, 
                                    y_fvc52, y_delta_fvc)
        
        # Backward pass
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Track losses
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        
        # Track predictions
        all_preds_fvc52.extend(fvc52_pred.detach().cpu().numpy().flatten())
        all_labels_fvc52.extend(y_fvc52.cpu().numpy().flatten())
        all_preds_delta.extend(delta_fvc.detach().cpu().numpy().flatten())
        all_labels_delta.extend(y_delta_fvc.cpu().numpy().flatten())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{loss_dict['recon']:.4f}",
            'delta': f"{loss_dict['delta']:.4f}"
        })
    
    # Compute averages
    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    
    # Compute MAE for both heads
    mae_fvc52 = mean_absolute_error(all_labels_fvc52, all_preds_fvc52)
    mae_delta = mean_absolute_error(all_labels_delta, all_preds_delta)
    
    return avg_losses, mae_fvc52, mae_delta


def validate_dual_head(model, loader, criterion, device):
    """Validate dual-head model"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'recon': 0.0,
        'delta': 0.0,
        'proxy': 0.0
    }
    
    all_preds_fvc52 = []
    all_labels_fvc52 = []
    all_preds_delta = []
    all_labels_delta = []
    all_preds_proxy = []
    all_patient_ids = []
    all_baselines = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            x_img = batch['x_img'].to(device)
            x_hand = batch['x_hand'].to(device)
            y_fvc52 = batch['y_fvc52'].to(device)
            y_delta_fvc = batch['y_delta_fvc'].to(device)
            
            # Forward pass
            fvc_proxy, delta_fvc, fvc52_pred = model(x_img, x_hand, return_components=True)
            
            # Compute loss
            loss, loss_dict = criterion(fvc_proxy, delta_fvc, fvc52_pred, 
                                       y_fvc52, y_delta_fvc)
            
            # Track losses
            for key in total_losses.keys():
                total_losses[key] += loss_dict[key]
            
            # Track predictions
            all_preds_fvc52.extend(fvc52_pred.cpu().numpy().flatten())
            all_labels_fvc52.extend(y_fvc52.cpu().numpy().flatten())
            all_preds_delta.extend(delta_fvc.cpu().numpy().flatten())
            all_labels_delta.extend(y_delta_fvc.cpu().numpy().flatten())
            all_preds_proxy.extend(fvc_proxy.cpu().numpy().flatten())
            all_patient_ids.extend(batch['patient_id'])
            all_baselines.extend(batch['baseline_fvc'].numpy())
    
    # Compute averages
    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    
    # Compute metrics
    mae_fvc52 = mean_absolute_error(all_labels_fvc52, all_preds_fvc52)
    rmse_fvc52 = np.sqrt(mean_squared_error(all_labels_fvc52, all_preds_fvc52))
    r2_fvc52 = r2_score(all_labels_fvc52, all_preds_fvc52)
    
    mae_delta = mean_absolute_error(all_labels_delta, all_preds_delta)
    
    metrics = {
        'losses': avg_losses,
        'mae_fvc52': mae_fvc52,
        'rmse_fvc52': rmse_fvc52,
        'r2_fvc52': r2_fvc52,
        'mae_delta': mae_delta
    }
    
    return metrics, all_preds_fvc52, all_labels_fvc52, all_patient_ids, all_preds_proxy, all_preds_delta, all_baselines



# Nel tuo progression_fvc@52_kfold.py
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
    PATIENCE = 20
    
    # Paths
    KFOLD_SPLITS_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\Kfold_cyclic\kfold_cyclic_splits.pkl'
    CNN_MODELS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\checkpoints_trainings\checkpoints_mse')
    
    RESULTS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\results_cyclic_kfold\sample_weighting')
    MODELS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\models_cyclic_kfold\sample_weighting')
    PREDICTIONS_DIR = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\predictions_cyclic_kfold\sample_weighting')
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True,exist_ok=True)

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
        
        # Create datasets
        train_ds = DualHeadPatientMLPDataset(
            label_csv=CSV_PATH_LABEL_52,
            embeddings_dict=patient_embeddings,
            handcrafted_dict=features_data,
            patient_list=train_ids,
            feature_stats=feature_stats,
            fvc_norm_stats=fvc_norm_stats
        )

        val_ds = DualHeadPatientMLPDataset(
            label_csv=CSV_PATH_LABEL_52,
            embeddings_dict=patient_embeddings,
            handcrafted_dict=features_data,
            patient_list=val_ids,
            feature_stats=feature_stats,
            fvc_norm_stats=fvc_norm_stats
        )
        
        test_ds = DualHeadPatientMLPDataset(
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

        print(f"\n Creating fusion model...")

        #Determine actual feature dimension from dataset
        sample_batch = next(iter(train_loader))
        actual_hand_dim = sample_batch['x_hand'].shape[1]
        print(f"✓ Detected hand features dimension: {actual_hand_dim}")

        # Create model
        model = DualHeadFVCPredictor(
            img_dim=320,
            hand_dim=actual_hand_dim,
            shared_hidden=128,
            head_hidden=64,
            dropout=0.3
        ).to(device)

        # Create loss
        criterion = DualHeadLoss(alpha=0.6, beta=0.4, gamma=0.0)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode='min',factor=0.5,patience=10)

        #  Early stopping variables
        best_val_mae = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Train
        for epoch in range(EPOCHS):
            train_losses, train_mae_fvc52, train_mae_delta = train_epoch_dual_head(
                model, train_loader, criterion, optimizer, device, GRAD_CLIP
            )
            
            val_metrics, _, _, _, _, _, _ = validate_dual_head(
                model, val_loader, criterion, device
            )

            # Denormalize for display
            val_mae_denorm = val_metrics['mae_fvc52'] * fvc_std
            train_mae_denorm = train_mae_fvc52 * fvc_std

            #Update scheduler
            scheduler.step(val_metrics['mae_fvc52'])
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{EPOCHS}")
                print(f"  Train - Loss: {train_losses['total']:.4f} | MAE: {train_mae_denorm:.2f} mL | Delta MAE: {train_mae_delta:.4f}")
                print(f"  Val   - Loss: {val_metrics['losses']['total']:.4f} | MAE: {val_mae_denorm:.2f} mL | R²: {val_metrics['r2_fvc52']:.4f}")
                print(f"          Recon: {val_metrics['losses']['recon']:.4f} | Delta: {val_metrics['losses']['delta']:.4f}")

            # Save best model
            if val_metrics['mae_fvc52'] < best_val_mae:
                best_val_mae = val_metrics['mae_fvc52']
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = MODELS_DIR / f'model_fold{fold_idx}_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': val_metrics['mae_fvc52'],
                    'val_r2': val_metrics['r2_fvc52'],
                    'fvc_norm_stats': fvc_norm_stats,
                    'feature_stats': feature_stats
                }, checkpoint_path)
                print(f"  ✓ New best model! MAE: {val_mae_denorm:.2f} mL (epoch {best_epoch})")
            else:
                patience_counter += 1

            #  Early stopping
            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                print(f"   Best validation MAE: {best_val_mae * fvc_std:.2f} mL (epoch {best_epoch})")
                break
        
        #  Load best model for final evaluation
        print(f"\n📊 Loading best model for final evaluation...")
        checkpoint_path = MODELS_DIR / f'model_fold{fold_idx}_best.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])



        print(f"\n📊 Evaluating on all splits...")
    
        # Train predictions
        train_metrics, train_preds, train_labels, train_pids, train_proxy, train_delta, train_baselines = validate_dual_head(
            model, train_loader, criterion, device
        )
        train_preds_denorm = np.array(train_preds) * fvc_std + fvc_mean
        train_labels_denorm = np.array(train_labels) * fvc_std + fvc_mean

        # Val predictions
        val_metrics, val_preds, val_labels, val_pids, val_proxy, val_delta, val_baselines = validate_dual_head(
            model, val_loader, criterion, device
        )
        val_preds_denorm = np.array(val_preds) * fvc_std + fvc_mean
        val_labels_denorm = np.array(val_labels) * fvc_std + fvc_mean
        
        # Test predictions
        test_metrics, test_preds, test_labels, test_pids, test_proxy, test_delta, test_baselines = validate_dual_head(
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
            'best_epoch': best_epoch,
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

        # ✅ Store predictions with components (proxy, delta, final)
        all_predictions[f'fold_{fold_idx}'] = {
            'train': {
                'fvc52': {pid: pred for pid, pred in zip(train_pids, train_preds_denorm)},
                'proxy': {pid: proxy * fvc_std + fvc_mean for pid, proxy in zip(train_pids, train_proxy)},
                'delta': {pid: delta * fvc_std for pid, delta in zip(train_pids, train_delta)},
                'baseline': {pid: baseline for pid, baseline in zip(train_pids, train_baselines)}
            },
            'val': {
                'fvc52': {pid: pred for pid, pred in zip(val_pids, val_preds_denorm)},
                'proxy': {pid: proxy * fvc_std + fvc_mean for pid, proxy in zip(val_pids, val_proxy)},
                'delta': {pid: delta * fvc_std for pid, delta in zip(val_pids, val_delta)},
                'baseline': {pid: baseline for pid, baseline in zip(val_pids, val_baselines)}
            },
            'test': {
            'fvc52': {pid: pred for pid, pred in zip(test_pids, test_preds_denorm)},
            'proxy': {pid: proxy * fvc_std + fvc_mean for pid, proxy in zip(test_pids, test_proxy)},
            'delta': {pid: delta * fvc_std for pid, delta in zip(test_pids, test_delta)},
            'baseline': {pid: baseline for pid, baseline in zip(test_pids, test_baselines)}
            }
        }


        # Print fold results
        print(f"\n✓ FOLD {fold_idx} RESULTS (Best epoch: {best_epoch}):")
        print(f"   Train - MAE: {train_mae:.2f} mL, RMSE: {train_rmse:.2f} mL, R²: {train_r2:.4f}")
        print(f"   Val   - MAE: {val_mae:.2f} mL, RMSE: {val_rmse:.2f} mL, R²: {val_r2:.4f}")
        print(f"   Test  - MAE: {test_mae:.2f} mL, RMSE: {test_rmse:.2f} mL, R²: {test_r2:.4f}")
        
        print(f"\n Component analysis:")
        train_delta_denorm = np.array(train_delta) * fvc_std
        val_delta_denorm = np.array(val_delta) * fvc_std
        test_delta_denorm = np.array(test_delta) * fvc_std

        print(f"   Delta FVC (train): {train_delta_denorm.mean():.2f} ± {train_delta_denorm.std():.2f} mL")
        print(f"   Delta FVC (val):   {val_delta_denorm.mean():.2f} ± {val_delta_denorm.std():.2f} mL")
        print(f"   Delta FVC (test):  {test_delta_denorm.mean():.2f} ± {test_delta_denorm.std():.2f} mL")
                


        #Dopo aver calcolato test_r2, aggiungi:
        print(f"\n🔍 Collapse Check:")
        print(f"   Test predictions std: {np.std(test_preds_denorm):.2f} mL")
        print(f"   Test labels std: {np.std(test_labels_denorm):.2f} mL")
        print(f"   Ratio (pred/true): {np.std(test_preds_denorm) / np.std(test_labels_denorm):.3f}")

        if np.std(test_preds_denorm) < 50:  # Arbitraria soglia
            print(f"   ⚠️  WARNING: Low prediction variance - possible collapse!")
        else:
            print(f"   ✓ Healthy prediction variance")

        # Free up memory
        del model
        torch.cuda.empty_cache()

        

    #=============================================================================
    # STEP 4: SAVE RESULTS
    # =============================================================================
    print("\n" + "="*80)
    print("[4/4] SAVING RESULTS")
    print("="*80)

    # Save fold results
    results_df = pd.DataFrame(all_fold_results)
    results_path = RESULTS_DIR / 'dual_head_cyclic_kfold_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved results to {results_path}")

    # Save predictions
    predictions_path = PREDICTIONS_DIR / 'all_predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump(all_predictions, f)
    print(f"✓ Saved predictions to {predictions_path}")

    #Summary

    print("\n" + "="*80)
    print("FINAL SUMMARY")
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

    print(f"\n⏱️  Average best epoch: {results_df['best_epoch'].mean():.1f} ± {results_df['best_epoch'].std():.1f}")

    print("\n" + "="*80)
    print("ANALYZING DUAL-HEAD COMPONENTS")
    print("="*80)

    # Analizza quanto contribuisce ciascun componente
    for fold_idx in range(len(all_fold_results)):
        fold_key = f'fold_{fold_idx}'
        if fold_key not in all_predictions:
            continue
        
        test_data = all_predictions[fold_key]['test']
        
        # Get arrays
        pids = list(test_data['fvc52'].keys())
        fvc52_preds = np.array([test_data['fvc52'][pid] for pid in pids])
        proxy_preds = np.array([test_data['proxy'][pid] for pid in pids])
        delta_preds = np.array([test_data['delta'][pid] for pid in pids])
        
        print(f"\nFold {fold_idx}:")
        print(f"   FVC Proxy mean: {proxy_preds.mean():.2f} ± {proxy_preds.std():.2f} mL")
        print(f"   Delta FVC mean: {delta_preds.mean():.2f} ± {delta_preds.std():.2f} mL")
        print(f"   |Delta| / |Proxy|: {abs(delta_preds.mean()) / proxy_preds.mean():.3f}")
        
    print("\n✅ DUAL-HEAD CYCLIC K-FOLD CROSS-VALIDATION COMPLETE!")
