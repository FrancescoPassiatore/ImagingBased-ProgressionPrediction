"""
Utilities for 5-Fold Cross-Validation Retraining
=================================================

This file contains all helper functions, data loaders, and model definitions
used across the 4 different training approaches.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict
import timm

# =============================================================================
# CONSTANTS
# =============================================================================

HAND_FEATURE_ORDER = [
    'approx_vol',
    'avg_num_tissue_pixel',
    'avg_tissue',
    'avg_tissue_thickness',
    'avg_tissue_by_total',
    'avg_tissue_by_lung',
    'mean',
    'skew',
    'kurtosis'
]

DEMOGRAPHIC_FEATURES = [
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

# =============================================================================
# DATA LOADING
# =============================================================================

class IPFDataLoader:
    """Load patient data from CSV and NPY files"""
    
    def __init__(self, csv_path: str, features_path: str, npy_dir: str):
        self.csv_path = csv_path
        self.features_path = features_path
        self.npy_dir = Path(npy_dir)
        
    def get_patient_data(self) -> Tuple[Dict, Dict]:
        """
        Load patient data and features
        
        Returns:
            patient_data: Dict with {patient_id: {'weeks', 'fvc_values', 'intercept', 'slope'}}
            features_data: Dict with {patient_id: {feature: value}}
        """
        # Load main CSV
        df = pd.read_csv(self.csv_path)
        
        # Load features
        features_df = pd.read_csv(self.features_path)
        
        # Build patient_data dictionary
        patient_data = {}
        for patient_id in df['Patient'].unique():
            patient_df = df[df['Patient'] == patient_id].sort_values('Weeks')
            
            weeks = patient_df['Weeks'].values
            fvc_values = patient_df['FVC'].values
            
            patient_data[patient_id] = {
                'weeks': weeks.tolist(),
                'fvc_values': fvc_values.tolist(),
                'intercept': float(patient_df['fvc_intercept0'].iloc[0]),
                'slope': float(patient_df['fvc_slope'].iloc[0])
            }
        
        # Build features_data dictionary
        features_data = {}
        for _, row in features_df.iterrows():
            patient_id = row['Patient']
            features_data[patient_id] = {
                'approx_vol': float(row.get('approx_vol', 0)),
                'avg_num_tissue_pixel': float(row.get('avg_num_tissue_pixel', 0)),
                'avg_tissue': float(row.get('avg_tissue', 0)),
                'avg_tissue_thickness': float(row.get('avg_tissue_thickness', 0)),
                'avg_tissue_by_total': float(row.get('avg_tissue_by_total', 0)),
                'avg_tissue_by_lung': float(row.get('avg_tissue_by_lung', 0)),
                'mean': float(row.get('mean', 0)),
                'skew': float(row.get('skew', 0)),
                'kurtosis': float(row.get('kurtosis', 0)),
                'age': float(row.get('Age', 0)),
                'sex': float(row.get('Sex', 0)),
                'smoking_status': float(row.get('SmokingStatus', 0))
            }
        
        return patient_data, features_data


# =============================================================================
# K-FOLD SPLITTING
# =============================================================================

def create_kfold_splits(patient_ids: List[str], n_splits: int = 5, random_state: int = 42) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Create K-Fold splits for patients with train/val/test sets
    
    Strategy: Divide patients into n_splits groups. For each fold:
    - 3 groups -> train (60%)
    - 1 group -> validation (20%)
    - 1 group -> test (20%)
    
    Returns:
        List of (train_ids, val_ids, test_ids) tuples
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    patient_ids = np.array(patient_ids)
    
    # First, create 5 equal groups
    all_folds = []
    for _, fold_idx in kf.split(patient_ids):
        all_folds.append(patient_ids[fold_idx].tolist())
    
    splits = []
    for i in range(n_splits):
        # Rotate which folds are train/val/test
        test_fold = i
        val_fold = (i + 1) % n_splits
        train_folds = [j for j in range(n_splits) if j not in [test_fold, val_fold]]
        
        train_ids = []
        for fold_idx in train_folds:
            train_ids.extend(all_folds[fold_idx])
        
        val_ids = all_folds[val_fold]
        test_ids = all_folds[test_fold]
        
        splits.append((train_ids, val_ids, test_ids))
    
    return splits


# =============================================================================
# CNN DATASET FOR SLOPE PREDICTION
# =============================================================================

class IPFSliceDataset(Dataset):
    """Dataset for per-slice slope prediction"""
    
    def __init__(self, patient_ids: List[str], patient_data: Dict, 
                 features_data: Dict, image_size: Tuple[int, int] = (224, 224),
                 normalize_slope: bool = True, slope_scaler: Optional[StandardScaler] = None):
        
        self.patients = patient_ids
        self.patient_data = patient_data
        self.features_data = features_data
        self.image_size = image_size
        self.normalize_slope = normalize_slope
        self.slope_scaler = slope_scaler
        
        # Build index
        self.samples = []
        self.patient_to_indices = defaultdict(list)
        
        for patient_id in patient_ids:
            if patient_id not in patient_data:
                continue
                
            pdata = patient_data[patient_id]
            slope = pdata['slope']
            
            # Find NPY files for this patient
            patient_dir = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy') / patient_id
            
            if patient_dir.exists():
                npy_files = sorted(list(patient_dir.glob('*.npy')))
                
                for npy_file in npy_files:
                    idx = len(self.samples)
                    self.samples.append({
                        'patient_id': patient_id,
                        'npy_path': str(npy_file),
                        'slope': slope
                    })
                    self.patient_to_indices[patient_id].append(idx)
        
        # Fit slope scaler if needed
        if self.normalize_slope and self.slope_scaler is None:
            slopes = np.array([s['slope'] for s in self.samples]).reshape(-1, 1)
            self.slope_scaler = StandardScaler()
            self.slope_scaler.fit(slopes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            img = np.load(sample['npy_path'])
            
            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img / 255.0
            
            # Resize
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to 3 channels
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=0)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = np.transpose(img, (2, 0, 1))
                img = np.repeat(img, 3, axis=0)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            
            img_tensor = torch.from_numpy(img).float()
            
            # Get slope
            slope = sample['slope']
            if self.normalize_slope and self.slope_scaler is not None:
                slope = self.slope_scaler.transform([[slope]])[0][0]
            
            return {
                'image': img_tensor,
                'slope': torch.tensor(slope, dtype=torch.float32),
                'patient_id': sample['patient_id']
            }
            
        except Exception as e:
            print(f"Error loading {sample['npy_path']}: {e}")
            return None


def patient_group_collate(batch):
    """Collate function that groups slices by patient"""
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    images = torch.stack([b['image'] for b in batch])
    slopes = torch.stack([b['slope'] for b in batch])
    patient_ids = [b['patient_id'] for b in batch]
    
    # Group by patient
    unique_patients = []
    patient_groups = []
    current_patient = None
    current_group = []
    
    for i, pid in enumerate(patient_ids):
        if pid != current_patient:
            if current_group:
                unique_patients.append(current_patient)
                patient_groups.append(current_group)
            current_patient = pid
            current_group = [i]
        else:
            current_group.append(i)
    
    if current_group:
        unique_patients.append(current_patient)
        patient_groups.append(current_group)
    
    return {
        'images': images,
        'slopes': slopes,
        'patient_ids': patient_ids,
        'unique_patients': unique_patients,
        'patient_groups': patient_groups,
        'lengths': torch.tensor([len(g) for g in patient_groups])
    }


class PatientBatchSampler:
    """Batch sampler that groups slices by patient"""
    
    def __init__(self, dataset, patients_per_batch=4, shuffle=True):
        self.dataset = dataset
        self.patients_per_batch = patients_per_batch
        self.shuffle = shuffle
        self.patients = list(dataset.patient_to_indices.keys())
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.patients)
        
        for i in range(0, len(self.patients), self.patients_per_batch):
            batch_patients = self.patients[i:i + self.patients_per_batch]
            
            indices = []
            for patient_id in batch_patients:
                indices.extend(self.dataset.patient_to_indices[patient_id])
            
            yield indices
    
    def __len__(self):
        return (len(self.patients) + self.patients_per_batch - 1) // self.patients_per_batch


# =============================================================================
# SPATIAL ATTENTION MODULE
# =============================================================================

class SpatialAttentionModule(nn.Module):
    """
    Explicit spatial attention that learns to focus on informative regions
    This is better than just relying on the CNN to learn implicitly
    """
    def __init__(self, in_channels):
        super().__init__()

        # Spatial attention pathway
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()  # Output attention map [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            attended_x: (B, C, H, W) spatially weighted features
            attention_map: (B, 1, H, W) for visualization
        """
        attention_map = self.spatial_attention(x)  # (B, 1, H, W)
        attended_x = x * attention_map  # Element-wise multiplication

        return attended_x, attention_map


# =============================================================================
# CNN MODEL
# =============================================================================

class ImprovedSliceLevelCNN(nn.Module):
    """
    CNN with explicit spatial attention
    Forces model to learn WHERE to look
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            features_only=True  # Return intermediate features
        )

        # Get feature dimensions
        # For EfficientNet-B0: [16, 24, 40, 112, 1280]
        feat_dims = self.backbone.feature_info.channels()

        # Add spatial attention to last feature map
        self.spatial_attention = SpatialAttentionModule(feat_dims[-1])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dims[-1], 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def extract_features(self, x, return_attention=False):
        """Extract CNN features (useful for other approaches)"""
        features = self.backbone(x)
        last_feature_map = features[-1]
        
        attended_features, attention_map = self.spatial_attention(last_feature_map)
        
        pooled = self.global_pool(attended_features).flatten(1)  # (B, 1280)
        
        if return_attention:
            return pooled, attention_map
        
        return pooled

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 3, H, W) input images
            return_attention: if True, return attention maps for visualization
        """
        # Extract features
        features = self.backbone(x)
        last_feature_map = features[-1]  # (B, C, h, w)

        # Apply spatial attention
        attended_features, attention_map = self.spatial_attention(last_feature_map)

        # Global pooling
        pooled = self.global_pool(attended_features).flatten(1)  # (B, C)

        # Prediction
        output = self.head(pooled).squeeze(-1)  # (B,)

        if return_attention:
            return output, attention_map
        return output


# =============================================================================
# SLOPE CORRECTOR MODELS (4 APPROACHES)
# =============================================================================

class SlopeCorrectorCNNOnly(nn.Module):
    """Approach 1: CNN-only (no correction, just identity)"""
    
    def __init__(self):
        super().__init__()
        # No parameters - just return input
    
    def forward(self, slope_cnn):
        return slope_cnn


class SlopeCorrectorCNNHandcrafted(nn.Module):
    """Approach 2: CNN + Handcrafted features (with residual connection)"""
    
    def __init__(self, n_handcrafted=9):
        super().__init__()
        input_dim = 1 + n_handcrafted  # slope + handcrafted
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        base = x[:, 0:1]  # slope_cnn_mean (first feature)
        corr = self.mlp(x)
        return (base + corr).squeeze(-1)


class SlopeCorrectorCNNDemographics(nn.Module):
    """Approach 3: CNN + Demographics (with residual connection)"""
    
    def __init__(self, n_demographics=3):
        super().__init__()
        input_dim = 1 + n_demographics  # slope + demographics
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        base = x[:, 0:1]  # slope_cnn_mean (first feature)
        corr = self.mlp(x)
        return (base + corr).squeeze(-1)


class SlopeCorrectorFull(nn.Module):
    """Approach 4: CNN + Handcrafted + Demographics (with residual connection)"""
    
    def __init__(self, n_handcrafted=9, n_demographics=3):
        super().__init__()
        input_dim = 1 + n_handcrafted + n_demographics  # slope + all features
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        base = x[:, 0:1]  # slope_cnn_mean (first feature)
        corr = self.mlp(x)
        return (base + corr).squeeze(-1)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_patient_features(patient_id: str, features_data: Dict, 
                            feature_type: str = 'full') -> np.ndarray:
    """
    Extract features for a patient based on approach type
    
    Args:
        patient_id: Patient ID
        features_data: Feature dictionary
        feature_type: 'none', 'handcrafted', 'demographics', 'full'
    
    Returns:
        feature_vector: numpy array
    """
    if patient_id not in features_data:
        if feature_type == 'none':
            return np.array([])
        elif feature_type == 'handcrafted':
            return np.zeros(len(HAND_FEATURE_ORDER))
        elif feature_type == 'demographics':
            return np.zeros(len(DEMOGRAPHIC_FEATURES))
        else:  # full
            return np.zeros(len(HAND_FEATURE_ORDER) + len(DEMOGRAPHIC_FEATURES))
    
    features = features_data[patient_id]
    
    if feature_type == 'none':
        return np.array([])
    elif feature_type == 'handcrafted':
        return np.array([features.get(f, 0.0) for f in HAND_FEATURE_ORDER])
    elif feature_type == 'demographics':
        return np.array([features.get(f, 0.0) for f in DEMOGRAPHIC_FEATURES])
    else:  # full
        return np.array([features.get(f, 0.0) for f in HAND_FEATURE_ORDER + DEMOGRAPHIC_FEATURES])


def compute_feature_stats(features_data: Dict, patient_ids: List[str], 
                         feature_names: List[str]) -> Dict:
    """Compute mean/std for feature normalization (from training data only)"""
    stats = {}
    
    for feature in feature_names:
        values = []
        for pid in patient_ids:
            if pid in features_data:
                val = features_data[pid].get(feature, 0.0)
                if not np.isnan(val):
                    values.append(val)
        
        if len(values) > 0:
            stats[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) if np.std(values) > 0 else 1.0
            }
        else:
            stats[feature] = {'mean': 0.0, 'std': 1.0}
    
    return stats


# =============================================================================
# CORRECTOR DATASET
# =============================================================================

class CorrectorDataset(Dataset):
    """Dataset for training slope corrector"""
    
    def __init__(self, patient_ids: List[str], patient_data: Dict, 
                 features_data: Dict, cnn_slopes: Dict, 
                 feature_type: str = 'full', 
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            patient_ids: List of patient IDs
            patient_data: Patient data dict
            features_data: Features dict
            cnn_slopes: Dict {patient_id: mean_cnn_slope}
            feature_type: 'none', 'handcrafted', 'demographics', 'full'
            scaler: Pre-fitted StandardScaler (or None to fit new one)
        """
        self.patient_ids = patient_ids
        self.patient_data = patient_data
        self.features_data = features_data
        self.cnn_slopes = cnn_slopes
        self.feature_type = feature_type
        self.scaler = scaler
        
        # Build samples
        self.samples = []
        for pid in patient_ids:
            if pid in cnn_slopes and pid in patient_data:
                self.samples.append(pid)
        
        # Fit scaler if needed
        if self.scaler is None and feature_type != 'none':
            self._fit_scaler()
    
    def _fit_scaler(self):
        """Fit scaler on training data"""
        all_features = []
        
        for pid in self.samples:
            slope_cnn = self.cnn_slopes[pid]
            features = extract_patient_features(pid, self.features_data, self.feature_type)
            
            if self.feature_type == 'none':
                feature_vector = np.array([slope_cnn])
            else:
                feature_vector = np.concatenate([[slope_cnn], features])
            
            all_features.append(feature_vector)
        
        all_features = np.array(all_features)
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid = self.samples[idx]
        
        # Get CNN slope
        slope_cnn = self.cnn_slopes[pid]
        
        # Get true slope
        slope_true = self.patient_data[pid]['slope']
        
        # Get features
        features = extract_patient_features(pid, self.features_data, self.feature_type)
        
        # Combine
        if self.feature_type == 'none':
            feature_vector = np.array([slope_cnn])
        else:
            feature_vector = np.concatenate([[slope_cnn], features])
        
        # Normalize
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        
        return {
            'features': torch.tensor(feature_vector, dtype=torch.float32),
            'slope': torch.tensor(slope_true, dtype=torch.float32),
            'patient_id': pid
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_corrector_epoch(model, dataloader, optimizer, criterion, device):
    """Train corrector for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        slopes = batch['slopes'].to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, slopes)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_corrector(model, dataloader, device):
    """Evaluate corrector"""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            slopes = batch['slopes'].to(device)
            
            predictions = model(features)
            
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(slopes.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': all_preds, 'true': all_true}


# =============================================================================
# FVC@52 PREDICTION
# =============================================================================

def predict_fvc_at_week(baseline_fvc: float, slope: float, target_week: float, 
                       baseline_week: float = 0.0) -> float:
    """Predict FVC at a specific week using linear model"""
    weeks_delta = target_week - baseline_week
    fvc_predicted = baseline_fvc + slope * weeks_delta
    return fvc_predicted


def compute_fvc52_metrics(true_fvc52: np.ndarray, pred_fvc52: np.ndarray) -> Dict:
    """Compute metrics for FVC@52 prediction"""
    mse = mean_squared_error(true_fvc52, pred_fvc52)
    mae = mean_absolute_error(true_fvc52, pred_fvc52)
    r2 = r2_score(true_fvc52, pred_fvc52)
    rmse = np.sqrt(mse)
    
    # Percentage errors
    pct_errors = np.abs(true_fvc52 - pred_fvc52) / true_fvc52 * 100
    mean_pct_error = np.mean(pct_errors)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_pct_error': mean_pct_error
    }


# =============================================================================
# SAVING/LOADING
# =============================================================================

def save_fold_results(fold: int, approach: str, results: Dict, save_dir: Path):
    """Save results for a specific fold and approach"""
    save_path = save_dir / f'fold{fold}_{approach}_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved: {save_path}")


def load_fold_results(fold: int, approach: str, save_dir: Path) -> Dict:
    """Load results for a specific fold and approach"""
    save_path = save_dir / f'fold{fold}_{approach}_results.pkl'
    with open(save_path, 'rb') as f:
        return pickle.load(f)
