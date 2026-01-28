"""
Dual-Head CNN Training: CT-Only Prediction of FVC Trajectory
============================================================

Architecture:
  CT → Z (lung state embedding)
  Z → FVC(0)_predicted (inferred baseline from imaging)
  Z → Slope_predicted (progression rate)
  FVC(52) = FVC(0)_pred + Slope_pred × 52 (reconstruction)
  
Loss:
  L = MSE(FVC_52_true, FVC_52_pred) + λ × MSE(Slope_true, Slope_pred)
  
✓ NO BASELINE FVC NEEDED - Everything predicted from CT!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import copy

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader,
    create_kfold_splits,
    IPFSliceDataset,
    PatientBatchSampler,
    patient_group_collate,
    compute_metrics,
    PredictionTracker,
    ExtremeOversamplingBatchSampler,
    compute_exponential_weights,
    SpatialAttentionModule
)

import timm

# =============================================================================
# DUAL-HEAD MODEL (CT-ONLY INPUT)
# =============================================================================

class DualHeadSliceLevelCNN(nn.Module):
    """
    Dual-head architecture with PROPER slice aggregation:
    1. Extract embeddings from each slice
    2. Aggregate embeddings into patient-level representation
    3. Predict baseline/slope from aggregated embedding
    
    ✓ Input: CT scan (multiple slices)
    ✓ Output: Single FVC(0), Slope, FVC(52) per patient
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b1', 
                 pretrained: bool = True, dropout: float = 0.3,
                 aggregation: str = 'attention'):  # 'mean', 'attention', 'max'
        super().__init__()
        
        self.aggregation = aggregation
        
        # Shared backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            features_only=True
        )
        
        feat_dims = self.backbone.feature_info.channels()
        
        # Spatial attention (per-slice)
        self.spatial_attention = SpatialAttentionModule(feat_dims[-1])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Per-slice embedding
        self.slice_embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dims[-1], 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # ✓ SLICE AGGREGATION: Attention weights for slice importance
        if self.aggregation == 'attention':
            self.slice_attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        # Patient-level embedding (post-aggregation)
        self.patient_embedding = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # ✓ HEAD 1: Predict FVC(0) from patient embedding
        self.baseline_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.33),
            nn.Linear(128, 1)
        )
        
        # ✓ HEAD 2: Predict slope from patient embedding
        self.slope_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.33),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, lengths=None, return_attention=False):
        """
        Args:
            x: (total_slices, C, H, W) - batched slices from multiple patients
            lengths: (num_patients,) - number of slices per patient
            return_attention: if True, return slice attention weights
        
        Returns:
            baseline_preds: (num_patients,) predicted FVC at week 0
            slope_preds: (num_patients,) predicted slope
            fvc_52_preds: (num_patients,) reconstructed FVC at week 52
            attention_weights: optional (num_patients, max_slices)
        """
        batch_size = x.size(0)
        
        # =====================================================================
        # STEP 1: Extract per-slice embeddings
        # =====================================================================
        features = self.backbone(x)
        last_feature_map = features[-1]  # (total_slices, C, h, w)
        
        # Apply spatial attention
        attended_features, spatial_attn = self.spatial_attention(last_feature_map)
        
        # Global pooling
        pooled = self.global_pool(attended_features).flatten(1)  # (total_slices, C)
        
        # Per-slice embeddings
        slice_embeddings = self.slice_embedding(pooled)  # (total_slices, 256)
        
        # =====================================================================
        # STEP 2: Aggregate slices into patient-level embeddings
        # =====================================================================
        if lengths is None:
            # Single patient case
            if self.aggregation == 'attention':
                attn_scores = self.slice_attention(slice_embeddings)  # (total_slices, 1)
                attn_weights = F.softmax(attn_scores, dim=0)
                patient_embedding = (slice_embeddings * attn_weights).sum(dim=0, keepdim=True)
                slice_attn_weights = attn_weights.squeeze(-1)
            elif self.aggregation == 'max':
                patient_embedding, _ = torch.max(slice_embeddings, dim=0, keepdim=True)
                slice_attn_weights = None
            else:  # mean
                patient_embedding = slice_embeddings.mean(dim=0, keepdim=True)
                slice_attn_weights = None
        else:
            # Multiple patients: aggregate per patient
            slice_blocks = torch.split(slice_embeddings, lengths.tolist())
            patient_embeddings = []
            slice_attn_weights_list = []
            
            for slice_block in slice_blocks:
                if self.aggregation == 'attention':
                    # Attention-weighted aggregation
                    attn_scores = self.slice_attention(slice_block)  # (n_slices, 1)
                    attn_weights = F.softmax(attn_scores, dim=0)
                    patient_emb = (slice_block * attn_weights).sum(dim=0)
                    patient_embeddings.append(patient_emb)
                    slice_attn_weights_list.append(attn_weights.squeeze(-1))
                elif self.aggregation == 'max':
                    patient_emb, _ = torch.max(slice_block, dim=0)
                    patient_embeddings.append(patient_emb)
                else:  # mean
                    patient_emb = slice_block.mean(dim=0)
                    patient_embeddings.append(patient_emb)
            
            patient_embedding = torch.stack(patient_embeddings)  # (num_patients, 256)
            slice_attn_weights = slice_attn_weights_list if self.aggregation == 'attention' else None
        
        # Further processing of patient embedding
        patient_embedding = self.patient_embedding(patient_embedding)  # (num_patients, 256)
        
        # =====================================================================
        # STEP 3: Predict from patient-level embedding
        # =====================================================================
        baseline_preds = self.baseline_head(patient_embedding).squeeze(-1)  # (num_patients,)
        slope_preds = self.slope_head(patient_embedding).squeeze(-1)  # (num_patients,)
        
        # ✓ Reconstruct FVC(52) using physics constraint
        fvc_52_preds = baseline_preds + slope_preds * 52  # (num_patients,)
        
        if return_attention:
            return baseline_preds, slope_preds, fvc_52_preds, slice_attn_weights
        
        return baseline_preds, slope_preds, fvc_52_preds


# =============================================================================
# DUAL-HEAD LOSS (NO BASELINE SUPERVISION)
# =============================================================================

class DualHeadLoss(nn.Module):
    """
    Combined loss WITHOUT baseline FVC supervision:
    L = MSE(FVC_52_true, FVC_52_pred) + λ_slope × MSE(Slope_true, Slope_pred)
    
    The baseline head learns IMPLICITLY through the FVC_52 reconstruction loss.
    """
    
    def __init__(self, lambda_slope: float = 0.5):
        super().__init__()
        self.lambda_slope = lambda_slope
    
    def forward(self, baseline_pred, slope_pred, fvc_52_pred, 
                slope_true, fvc_52_true):
        """
        Args:
            baseline_pred: (B,) predicted FVC(0) - NO ground truth!
            slope_pred: (B,) predicted slope
            fvc_52_pred: (B,) reconstructed FVC(52)
            slope_true: (B,) true slope (from linear regression)
            fvc_52_true: (B,) true FVC(52) (computed or measured)
        """
        # ✓ PRIMARY LOSS: FVC at week 52 (clinical outcome)
        loss_fvc_52 = F.mse_loss(fvc_52_pred, fvc_52_true)
        
        # ✓ AUXILIARY LOSS: slope (helps guide learning)
        loss_slope = F.mse_loss(slope_pred, slope_true)
        
        # Combined loss (NO baseline loss!)
        total_loss = loss_fvc_52 + self.lambda_slope * loss_slope
        
        return {
            'total': total_loss,
            'fvc_52': loss_fvc_52.item(),
            'slope': loss_slope.item()
        }


# =============================================================================
# MODIFIED DATASET (NO BASELINE FVC INPUT)
# =============================================================================

class DualHeadIPFDataset(IPFSliceDataset):
    """
    Dataset that returns:
    - image: CT scan
    - slope: true slope (from regression)
    - fvc_52: true FVC at week 52
    
    ✓ NO baseline_fvc input needed!
    """
    
    def __init__(self, patient_ids, patient_data, features_data, 
                 image_size=(224, 224), normalize_slope=True, 
                 normalize_fvc=True, slope_scaler=None, fvc_scaler=None,
                 augment=False):
        
        # Initialize parent without intercept normalization
        super().__init__(
            patient_ids, patient_data, features_data,
            image_size, normalize_slope, False,
            slope_scaler, None, augment
        )
        
        self.normalize_fvc = normalize_fvc
        self.fvc_scaler = fvc_scaler
        
        # Compute FVC at week 52 for each patient
        fvc_52_values = []
        for s in self.slices:
            pid = s['patient_id']
            # Compute FVC(52) = intercept + slope × 52
            # (intercept here is baseline FVC from regression)
            baseline = patient_data[pid].get('intercept', 0)
            slope = patient_data[pid]['slope']
            fvc_52 = baseline + slope * 52
            fvc_52_values.append(fvc_52)
        
        # Fit FVC scaler on FVC(52) values
        if self.normalize_fvc and self.fvc_scaler is None:
            # Get unique FVC(52) per patient
            unique_fvc52 = {}
            for s, fvc52 in zip(self.slices, fvc_52_values):
                pid = s['patient_id']
                if pid not in unique_fvc52:
                    unique_fvc52[pid] = fvc52
            
            fvc52_array = np.array(list(unique_fvc52.values())).reshape(-1, 1)
            from sklearn.preprocessing import StandardScaler
            self.fvc_scaler = StandardScaler()
            self.fvc_scaler.fit(fvc52_array)
        
        # Update slices with FVC(52)
        for i, fvc_52 in enumerate(fvc_52_values):
            self.slices[i]['fvc_52'] = fvc_52
    
    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        
        slice_info = self.slices[idx]
        
        try:
            # Load image
            img = np.load(slice_info['npy_path'])
            
            if img.shape != (224, 224):
                raise ValueError(f"Unexpected image shape {img.shape}")
            
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=0)
            
            img_tensor = torch.FloatTensor(img)
        
        except Exception as e:
            print(f"Error loading {slice_info['npy_path']}: {e}")
            self._cache[idx] = None
            return None
        
        # Normalize slope
        slope = slice_info['slope']
        if self.normalize_slope:
            slope = self.slope_scaler.transform([[slope]])[0][0]
        
        # Normalize FVC(52)
        fvc_52 = slice_info['fvc_52']
        if self.normalize_fvc:
            fvc_52 = self.fvc_scaler.transform([[fvc_52]])[0][0]
        
        result = {
            'image': img_tensor,
            'slope': torch.FloatTensor([slope]),
            'fvc_52': torch.FloatTensor([fvc_52]),
            'patient_id': slice_info['patient_id'],
            'slice_path': slice_info['npy_path']
        }
        
        self._cache[idx] = result
        return result


# Modified collate function
def dual_head_collate(batch):
    """Collate function for dual-head model (no baseline FVC)"""
    batch = [b for b in batch if b is not None]
    
    if not batch:
        return {
            'images': torch.empty(0, 3, 224, 224),
            'slopes': torch.empty(0),
            'fvc_52s': torch.empty(0),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long)
        }
    
    images = torch.stack([item['image'] for item in batch])
    slopes = torch.stack([item['slope'] for item in batch])
    fvc_52s = torch.stack([item['fvc_52'] for item in batch])
    
    # Count slices per patient
    lengths = []
    pid_order = []
    i = 0
    while i < len(batch):
        current_pid = batch[i]['patient_id']
        j = i
        while j < len(batch) and batch[j]['patient_id'] == current_pid:
            j += 1
        lengths.append(j - i)
        pid_order.append(current_pid)
        i = j
    
    return {
        'images': images,
        'slopes': slopes,
        'fvc_52s': fvc_52s,
        'patient_ids': pid_order,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch_dual_head(model, dataloader, optimizer, criterion, device,
                          gradient_clip=1.0, accumulation_steps=1,
                          use_attention=False, sample_weights_dict=None):
    """Train one epoch with dual-head model (proper aggregation)"""
    model.train()
    total_loss = 0.0
    total_fvc_52_loss = 0.0
    total_slope_loss = 0.0
    n_patients = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        images = batch['images'].to(device)
        slopes = batch['slopes'].to(device)
        fvc_52s = batch['fvc_52s'].to(device)
        lengths = batch['lengths'].to(device)
        patient_ids = batch['patient_ids']
        
        # ✓ Forward pass: aggregates slices INSIDE model
        if use_attention:
            baseline_preds, slope_preds, fvc_52_preds, slice_attn = model(
                images, lengths=lengths, return_attention=True
            )
        else:
            baseline_preds, slope_preds, fvc_52_preds = model(
                images, lengths=lengths
            )
        
        # Get true values (take first of each patient block)
        slope_true_blocks = torch.split(slopes, lengths.tolist())
        fvc_52_true_blocks = torch.split(fvc_52s, lengths.tolist())
        
        patient_slope_true = torch.stack([b[0] for b in slope_true_blocks])
        patient_fvc_52_true = torch.stack([b[0] for b in fvc_52_true_blocks])
        
        # ✓ Compute loss (predictions are already patient-level!)
        loss_dict = criterion(
            baseline_preds,      # (num_patients,)
            slope_preds,         # (num_patients,)
            fvc_52_preds,        # (num_patients,)
            patient_slope_true,  # (num_patients,)
            patient_fvc_52_true  # (num_patients,)
        )
        
        loss = loss_dict['total']
        
        # Apply sample weighting if provided
        if sample_weights_dict is not None:
            weights = torch.tensor(
                [sample_weights_dict.get(pid, 1.0) for pid in patient_ids],
                dtype=torch.float32, device=device
            )
            loss = loss * weights.mean()
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        n_patients_in_batch = len(lengths)
        total_loss += loss.item() * n_patients_in_batch * accumulation_steps
        total_fvc_52_loss += loss_dict['fvc_52'] * n_patients_in_batch
        total_slope_loss += loss_dict['slope'] * n_patients_in_batch
        n_patients += n_patients_in_batch
    
    return {
        'total': total_loss / n_patients if n_patients > 0 else 0.0,
        'fvc_52': total_fvc_52_loss / n_patients if n_patients > 0 else 0.0,
        'slope': total_slope_loss / n_patients if n_patients > 0 else 0.0
    }


def validate_dual_head(model, dataloader, criterion, device, 
                       return_predictions=False, use_attention=False):
    """Validate dual-head model (proper aggregation)"""
    model.eval()
    total_loss = 0.0
    total_fvc_52_loss = 0.0
    total_slope_loss = 0.0
    n_patients = 0
    
    all_baseline_preds = []
    all_fvc_52_preds = []
    all_fvc_52_true = []
    all_slope_preds = []
    all_slope_true = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            slopes = batch['slopes'].to(device)
            fvc_52s = batch['fvc_52s'].to(device)
            lengths = batch['lengths'].to(device)
            patient_ids = batch['patient_ids']
            
            # ✓ Forward pass: aggregates slices INSIDE model
            if use_attention:
                baseline_preds, slope_preds, fvc_52_preds, slice_attn = model(
                    images, lengths=lengths, return_attention=True
                )
            else:
                baseline_preds, slope_preds, fvc_52_preds = model(
                    images, lengths=lengths
                )
            
            # Get true values
            slope_true_blocks = torch.split(slopes, lengths.tolist())
            fvc_52_true_blocks = torch.split(fvc_52s, lengths.tolist())
            
            patient_slope_true = torch.stack([b[0] for b in slope_true_blocks])
            patient_fvc_52_true = torch.stack([b[0] for b in fvc_52_true_blocks])
            
            # Loss
            loss_dict = criterion(
                baseline_preds,
                slope_preds,
                fvc_52_preds,
                patient_slope_true,
                patient_fvc_52_true
            )
            
            n_patients_in_batch = len(lengths)
            total_loss += loss_dict['total'].item() * n_patients_in_batch
            total_fvc_52_loss += loss_dict['fvc_52'] * n_patients_in_batch
            total_slope_loss += loss_dict['slope'] * n_patients_in_batch
            n_patients += n_patients_in_batch
            
            all_baseline_preds.extend(baseline_preds.cpu().numpy())
            all_fvc_52_preds.extend(fvc_52_preds.cpu().numpy())
            all_fvc_52_true.extend(patient_fvc_52_true.cpu().numpy())
            all_slope_preds.extend(slope_preds.cpu().numpy())
            all_slope_true.extend(patient_slope_true.cpu().numpy())
            all_patient_ids.extend(patient_ids)
    
    avg_losses = {
        'total': total_loss / n_patients if n_patients > 0 else 0.0,
        'fvc_52': total_fvc_52_loss / n_patients if n_patients > 0 else 0.0,
        'slope': total_slope_loss / n_patients if n_patients > 0 else 0.0
    }
    
    # Compute metrics
    fvc_52_preds = np.array(all_fvc_52_preds)
    fvc_52_true = np.array(all_fvc_52_true)
    slope_preds = np.array(all_slope_preds)
    slope_true = np.array(all_slope_true)
    
    fvc_metrics = compute_metrics(fvc_52_true, fvc_52_preds)
    slope_metrics = compute_metrics(slope_true, slope_preds)
    
    metrics = {
        'fvc_52': fvc_metrics,
        'slope': slope_metrics
    }
    
    if return_predictions:
        return avg_losses, metrics, {
            'baseline_pred': np.array(all_baseline_preds),
            'fvc_52_pred': fvc_52_preds,
            'fvc_52_true': fvc_52_true,
            'slope_pred': slope_preds,
            'slope_true': slope_true,
            'patient_ids': all_patient_ids
        }
    
    return avg_losses, metrics


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_fold_dual_head(model, dataloader, device, slope_scaler, fvc_scaler, 
                           use_attention=False):
    """Generate predictions for all patients in a fold (proper aggregation)"""
    model.eval()
    predictions = {
        'baseline_fvc': {},
        'slope': {},
        'fvc_52': {}
    }
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            lengths = batch['lengths'].to(device)
            patient_ids = batch['patient_ids']
            
            # ✓ Forward pass: aggregates slices INSIDE model
            if use_attention:
                baseline_preds, slope_preds, fvc_52_preds, slice_attn = model(
                    images, lengths=lengths, return_attention=True
                )
            else:
                baseline_preds, slope_preds, fvc_52_preds = model(
                    images, lengths=lengths
                )
            
            # Predictions are already patient-level!
            for i, pid in enumerate(patient_ids):
                base_norm = baseline_preds[i].cpu().item()
                slope_norm = slope_preds[i].cpu().item()
                fvc_norm = fvc_52_preds[i].cpu().item()
                
                # Denormalize
                if fvc_scaler is not None:
                    base_denorm = fvc_scaler.inverse_transform([[base_norm]])[0][0]
                    fvc_denorm = fvc_scaler.inverse_transform([[fvc_norm]])[0][0]
                else:
                    base_denorm = base_norm
                    fvc_denorm = fvc_norm
                
                if slope_scaler is not None:
                    slope_denorm = slope_scaler.inverse_transform([[slope_norm]])[0][0]
                else:
                    slope_denorm = slope_norm
                
                predictions['baseline_fvc'][pid] = base_denorm
                predictions['slope'][pid] = slope_denorm
                predictions['fvc_52'][pid] = fvc_denorm
    
    return predictions


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    
    # Best params from Optuna
    'best_params_path': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\optuna\best_params.yaml'),
    
    # Training
    'n_folds': 5,
    'n_epochs': 150,
    'patience': 30,
    'image_size': (224, 224),
    'backbone': 'efficientnet_b1',
    'pretrained': True,
    'normalize_slope': True,
    'normalize_fvc': False,
    
    # Dual-head loss weights
    'lambda_slope': 0.3,  # Weight for slope loss (tune this: 0.3-0.7)
    
    # Stratified sampling
    'use_stratified_sampling': False,
    'n_strata_bins': 4,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Output
    'checkpoint_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_3\CNN_Training\Cyclic_kfold\checkpoints_dual_head_no_norm_fvc'),
    'predictions_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_3\CNN_Training\Cyclic_kfold\predictions_dual_head_no_norm_fvc'),
    'results_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_3\CNN_Training\Cyclic_kfold\final_results_dual_head_no_norm_fvc'),
    'diagnostics_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_3\CNN_Training\Cyclic_kfold\diagnostics_dual_head_no_norm_fvc')
}

# Create directories
for dir_path in [CONFIG['checkpoint_dir'], CONFIG['predictions_dir'], 
                 CONFIG['results_dir'], CONFIG['diagnostics_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAIN FOLD
# =============================================================================

def train_fold(fold_idx, train_ids, val_ids, test_ids, patient_data, features_data,
               best_params, config):
    """Train one fold with dual-head model"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*80}")
    print(f"Train: {len(train_ids)} patients | Val: {len(val_ids)} patients | Test: {len(test_ids)} patients")
    
    # =========================================================================
    # CREATE DATASETS
    # =========================================================================
    
    train_dataset = DualHeadIPFDataset(
        train_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        normalize_fvc=config['normalize_fvc'],
        augment=False
    )
    
    val_dataset = DualHeadIPFDataset(
        val_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        normalize_fvc=config['normalize_fvc'],
        slope_scaler=train_dataset.slope_scaler,
        fvc_scaler=train_dataset.fvc_scaler,
        augment=False
    )
    
    print(f"Train slices: {len(train_dataset)} | Val slices: {len(val_dataset)}")
    
    # =========================================================================
    # COMPUTE SAMPLE WEIGHTS
    # =========================================================================
    
    print(f"\n⚖️  Computing sample weights...")
    train_slopes = np.array([patient_data[pid]['slope'] for pid in train_ids])
    weight_result = compute_exponential_weights(train_slopes, n_bins=6, strength=2.0)
    sample_weights = weight_result['weights']
    sample_weights_dict = {pid: float(weight) for pid, weight in zip(train_ids, sample_weights)}
    
    print(f"\n📊 Sample Weight Distribution:")
    print(f"{'─'*80}")
    print(f"{'Slope Range':<20} {'Count':<10} {'Weight':<10} {'Samples %':<15}")
    print(f"{'─'*80}")
    for bin_info in weight_result['bin_info']:
        print(f"[{bin_info['range'][0]:6.2f}, {bin_info['range'][1]:6.2f})  "
              f"{bin_info['count']:>8d}  "
              f"{bin_info['weight']:>8.3f}  "
              f"{bin_info['samples_pct']:>12.1f}%")
    print(f"{'─'*80}\n")
    
    # =========================================================================
    # CREATE DATALOADERS
    # =========================================================================
    
    if config['use_stratified_sampling']:
        print(f"🎯 Using stratified sampling")
        train_sampler = ExtremeOversamplingBatchSampler(
            train_dataset,
            patients_per_batch=best_params['batch_size'],
            extreme_percentile=25,
            oversample_factor=4.0,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=dual_head_collate,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PatientBatchSampler(
                train_dataset,
                patients_per_batch=best_params['batch_size'],
                shuffle=True
            ),
            collate_fn=dual_head_collate,
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
        collate_fn=dual_head_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # Clean loaders for prediction
    train_pred_loader = DataLoader(
        train_dataset,
        batch_sampler=PatientBatchSampler(
            train_dataset,
            patients_per_batch=best_params['batch_size'],
            shuffle=False
        ),
        collate_fn=dual_head_collate,
        num_workers=4,
        pin_memory=True
    )
    
    val_pred_loader = DataLoader(
        val_dataset,
        batch_sampler=PatientBatchSampler(
            val_dataset,
            patients_per_batch=best_params['batch_size'],
            shuffle=False
        ),
        collate_fn=dual_head_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    
    model = DualHeadSliceLevelCNN(
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=best_params['dropout']
    ).to(config['device'])
    
    print(f"\n✓ Model: DualHeadSliceLevelCNN")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Pretrained: {config['pretrained']}")
    print(f"  Dropout: {best_params['dropout']}")
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.spatial_attention.parameters()) +
        list(model.slice_embedding.parameters()) +
        list(model.patient_embedding.parameters()) +
        list(model.baseline_head.parameters()) +
        list(model.slope_head.parameters())
    )
     
    if best_params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 2e-5},
            {"params": head_params, "lr": 3e-4}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            momentum=best_params.get('sgd_momentum', 0.9),
            nesterov=True
        )
    
    # =========================================================================
    # SCHEDULER
    # =========================================================================
    
    if best_params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=best_params.get('scheduler_patience', 5)
        )
    elif best_params['scheduler'] == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=best_params.get('step_size', 10), gamma=0.5
        )
    
    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    
    criterion = DualHeadLoss(lambda_slope=config['lambda_slope'])
    print(f"\n✓ Using DualHeadLoss (λ_slope={config['lambda_slope']})")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss_total': [],
        'train_loss_fvc52': [],
        'train_loss_slope': [],
        'val_loss_total': [],
        'val_loss_fvc52': [],
        'val_loss_slope': [],
        'val_fvc52_mae': [],
        'val_fvc52_rmse': [],
        'val_fvc52_r2': [],
        'val_slope_mae': [],
        'val_slope_rmse': [],
        'val_slope_r2': []
    }
    
    pred_tracker = PredictionTracker()
    
    print(f"\n🚀 Starting training...\n")
    
    for epoch in range(config['n_epochs']):
        print(f"Epoch {epoch + 1}/{config['n_epochs']}: ", end='', flush=True)
        
        # Train
        train_losses = train_epoch_dual_head(
            model, train_loader, optimizer, criterion, config['device'],
            gradient_clip=best_params['gradient_clip'],
            accumulation_steps=best_params['accumulation_steps'],
            use_attention=best_params['use_attention'],
            sample_weights_dict=sample_weights_dict
        )
        
        # Validate
        return_preds = (epoch + 1) % 5 == 0
        if return_preds:
            val_losses, val_metrics, preds_dict = validate_dual_head(
                model, val_loader, criterion, config['device'],
                return_predictions=True,
                use_attention=best_params['use_attention']
            )
            pred_tracker.update(epoch + 1, preds_dict['fvc_52_pred'], preds_dict['fvc_52_true'])
        else:
            val_losses, val_metrics = validate_dual_head(
                model, val_loader, criterion, config['device'],
                use_attention=best_params['use_attention']
            )
        
        # Learning rates
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head = optimizer.param_groups[1]['lr']
        
        print(
            f"Train: {train_losses['total']:.6f} "
            f"(FVC52: {train_losses['fvc_52']:.6f}, Slope: {train_losses['slope']:.6f}) | "
            f"Val: {val_losses['total']:.6f} "
            f"(FVC52: {val_losses['fvc_52']:.6f}, Slope: {val_losses['slope']:.6f}) | "
            f"FVC52_MAE: {val_metrics['fvc_52']['mae']:.4f} | "
            f"Slope_MAE: {val_metrics['slope']['mae']:.4f} | "
            f"LR: {lr_backbone:.2e}/{lr_head:.2e}",
            end='', flush=True
        )
        
        # Update scheduler
        if best_params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_losses['total'])
        else:
            scheduler.step()
        
        # Save history
        history['train_loss_total'].append(train_losses['total'])
        history['train_loss_fvc52'].append(train_losses['fvc_52'])
        history['train_loss_slope'].append(train_losses['slope'])
        history['val_loss_total'].append(val_losses['total'])
        history['val_loss_fvc52'].append(val_losses['fvc_52'])
        history['val_loss_slope'].append(val_losses['slope'])
        history['val_fvc52_mae'].append(val_metrics['fvc_52']['mae'])
        history['val_fvc52_rmse'].append(val_metrics['fvc_52']['rmse'])
        history['val_fvc52_r2'].append(val_metrics['fvc_52']['r2'])
        history['val_slope_mae'].append(val_metrics['slope']['mae'])
        history['val_slope_rmse'].append(val_metrics['slope']['rmse'])
        history['val_slope_r2'].append(val_metrics['slope']['r2'])
        
        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(" ✓ Best")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{config['patience']})")
            if patience_counter >= config['patience']:
                print(f"\n  ⏹️  Early stopping at epoch {epoch + 1}")
                break
        
        # Print detailed predictions every 5 epochs
        if return_preds:
            print(f"\n  {'─'*76}")
            print(f"  📊 Sample FVC(52) predictions (first 5 patients):")
            print(f"  {'─'*76}")
            print(f"  {'Patient ID':<20} {'Predicted':<15} {'True':<15} {'Error':<15}")
            print(f"  {'─'*76}")
            for i in range(min(5, len(preds_dict['patient_ids']))):
                pred = float(preds_dict['fvc_52_pred'][i])
                true = float(preds_dict['fvc_52_true'][i])
                error = pred - true
                print(f"  {preds_dict['patient_ids'][i]:<20} {pred:>12.2f}   {true:>12.2f}   {error:>12.2f}")
            print(f"  {'─'*76}\n")
    
    # =========================================================================
    # SAVE CHECKPOINT
    # =========================================================================
    
    model.load_state_dict(best_model_state)
    
    checkpoint_path = config['checkpoint_dir'] / f'dual_head_fold{fold_idx}.pt'
    torch.save({
        'fold': fold_idx,
        'model_state_dict': best_model_state,
        'best_val_loss': best_val_loss,
        'hyperparameters': best_params,
        'slope_scaler': train_dataset.slope_scaler,
        'fvc_scaler': train_dataset.fvc_scaler,
        'history': history,
        'lambda_slope': config['lambda_slope']
    }, checkpoint_path)
    
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")
    
    # =========================================================================
    # GENERATE PREDICTIONS
    # =========================================================================
    
    print(f"\n📊 Generating predictions...")
    
    test_dataset = DualHeadIPFDataset(
        test_ids,
        patient_data,
        features_data,
        image_size=config['image_size'],
        normalize_slope=config['normalize_slope'],
        normalize_fvc=config['normalize_fvc'],
        slope_scaler=train_dataset.slope_scaler,
        fvc_scaler=train_dataset.fvc_scaler,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=PatientBatchSampler(
            test_dataset,
            patients_per_batch=best_params['batch_size'],
            shuffle=False
        ),
        collate_fn=dual_head_collate,
        num_workers=4,
        pin_memory=True
    )
    
    train_preds = predict_fold_dual_head(
        model, train_pred_loader, config['device'],
        train_dataset.slope_scaler, train_dataset.fvc_scaler,
        use_attention=best_params['use_attention']
    )
    
    val_preds = predict_fold_dual_head(
        model, val_pred_loader, config['device'],
        train_dataset.slope_scaler, train_dataset.fvc_scaler,
        use_attention=best_params['use_attention']
    )
    
    test_preds = predict_fold_dual_head(
        model, test_loader, config['device'],
        train_dataset.slope_scaler, train_dataset.fvc_scaler,
        use_attention=best_params['use_attention']
    )
    
    predictions = {
        'train': train_preds,
        'val': val_preds,
        'test': test_preds
    }
    
    pred_path = config['predictions_dir'] / f'dual_head_predictions_fold{fold_idx}.pkl'
    with open(pred_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    print(f"✓ Saved predictions to {pred_path}")
    
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
    print("DUAL-HEAD CNN TRAINING - 5-FOLD CROSS-VALIDATION")
    print("="*80)
    print("\nModel: CT → {FVC(0), Slope} → FVC(52)")
    print(f"Loss: MSE(FVC_52) + {CONFIG['lambda_slope']} × MSE(Slope)")
    
    # Load best hyperparameters
    print(f"\n📋 Loading hyperparameters from {CONFIG['best_params_path']}")
    
    if not CONFIG['best_params_path'].exists():
        raise FileNotFoundError(
            f"Best parameters not found at {CONFIG['best_params_path']}\n"
            f"Run train_cnn_optuna.py first!"
        )
    
    with open(CONFIG['best_params_path'], 'r') as f:
        best_params = yaml.safe_load(f)
    
    print("\n📊 Hyperparameters:")
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
    splits_path = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\Kfold_cyclic\kfold_cyclic_splits.pkl')
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
    
    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL FOLDS")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            'Fold': r['fold'],
            'Val Loss': r['best_val_loss'],
            'FVC52_MAE': r['final_metrics']['fvc_52']['mae'],
            'FVC52_RMSE': r['final_metrics']['fvc_52']['rmse'],
            'FVC52_R²': r['final_metrics']['fvc_52']['r2'],
            'Slope_MAE': r['final_metrics']['slope']['mae'],
            'Slope_RMSE': r['final_metrics']['slope']['rmse'],
            'Slope_R²': r['final_metrics']['slope']['r2']
        }
        for r in all_results
    ])
    
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("AVERAGE METRICS:")
    print("-"*80)
    print(f"Val Loss:    {results_df['Val Loss'].mean():.6f} ± {results_df['Val Loss'].std():.6f}")
    print(f"\nFVC(52) Prediction:")
    print(f"  MAE:       {results_df['FVC52_MAE'].mean():.4f} ± {results_df['FVC52_MAE'].std():.4f}")
    print(f"  RMSE:      {results_df['FVC52_RMSE'].mean():.4f} ± {results_df['FVC52_RMSE'].std():.4f}")
    print(f"  R²:        {results_df['FVC52_R²'].mean():.4f} ± {results_df['FVC52_R²'].std():.4f}")
    print(f"\nSlope Prediction:")
    print(f"  MAE:       {results_df['Slope_MAE'].mean():.4f} ± {results_df['Slope_MAE'].std():.4f}")
    print(f"  RMSE:      {results_df['Slope_RMSE'].mean():.4f} ± {results_df['Slope_RMSE'].std():.4f}")
    print(f"  R²:        {results_df['Slope_R²'].mean():.4f} ± {results_df['Slope_R²'].std():.4f}")
    
    # Save results
    results_path = CONFIG['results_dir'] / 'final_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")
    
    # Save summary
    summary = {
        'model_type': 'DualHeadSliceLevelCNN',
        'lambda_slope': CONFIG['lambda_slope'],
        'hyperparameters': best_params,
        'mean_metrics': {
            'val_loss': float(results_df['Val Loss'].mean()),
            'fvc_52': {
                'mae': float(results_df['FVC52_MAE'].mean()),
                'rmse': float(results_df['FVC52_RMSE'].mean()),
                'r2': float(results_df['FVC52_R²'].mean())
            },
            'slope': {
                'mae': float(results_df['Slope_MAE'].mean()),
                'rmse': float(results_df['Slope_RMSE'].mean()),
                'r2': float(results_df['Slope_R²'].mean())
            },
           'std_metrics': {
            'val_loss': float(results_df['Val Loss'].std()),
            'fvc_52': {
                'mae': float(results_df['FVC52_MAE'].std()),
                'rmse': float(results_df['FVC52_RMSE'].std()),
                'r2': float(results_df['FVC52_R²'].std())
            },
            'slope': {
                'mae': float(results_df['Slope_MAE'].std()),
                'rmse': float(results_df['Slope_RMSE'].std()),
                'r2': float(results_df['Slope_R²'].std())
            }
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
    print("\n" + "="*80)


if __name__ == '__main__':
    main()