"""
Training 2.0 - Utilities with Fixed Feature Normalization
==========================================================

CRITICAL FIXES:
1. Separate scalers for handcrafted and demographic features
2. CNN slope already normalized - don't scale again
3. Proper feature normalization BEFORE concatenation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
from collections import defaultdict
import timm
import matplotlib.pyplot as plt
from scipy import stats

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

# Features that need normalization (continuous variables)
NORMALIZE_HAND_FEATURES = [
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

NORMALIZE_DEMO_FEATURES = ['age']  # sex and smoking_status are categorical

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
            
            # Map column names (CSV uses different naming convention)
            features_data[patient_id] = {
                'approx_vol': float(row['ApproxVol_30_60']),
                'avg_num_tissue_pixel': float(row['Avg_NumTissuePixel_30_60']),
                'avg_tissue': float(row['Avg_Tissue_30_60']),
                'avg_tissue_thickness': float(row['Avg_Tissue_thickness_30_60']),
                'avg_tissue_by_total': float(row['Avg_TissueByTotal_30_60']),
                'avg_tissue_by_lung': float(row['Avg_TissueByLung_30_60']),
                'mean': float(row['Mean_30_60']),
                'skew': float(row['Skew_30_60']),
                'kurtosis': float(row['Kurtosis_30_60']),
                'age': float(row['Age']) if 'Age' in row else 65.0,  # Default if missing
                'sex': int(row['Sex']) if 'Sex' in row else 0,
                'smoking_status': int(row['SmokingStatus']) if 'SmokingStatus' in row else 0
            }
        
        return patient_data, features_data


def create_kfold_splits(patient_ids: List[str], n_folds: int = 5, 
                       random_state: int = 42, save_path: Optional[Path] = None) -> Dict:
    """Create k-fold splits with train/val/test (60/20/20)"""
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = {}
    
    patient_ids = np.array(patient_ids)
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(patient_ids)):
        train_val_ids = patient_ids[train_val_idx]
        test_ids = patient_ids[test_idx]
        
        # Further split train_val into train (75%) and val (25%)
        # This gives 60/20/20 overall
        n_val = len(train_val_ids) // 4
        np.random.seed(random_state + fold_idx)
        val_indices = np.random.choice(len(train_val_ids), n_val, replace=False)
        
        val_mask = np.zeros(len(train_val_ids), dtype=bool)
        val_mask[val_indices] = True
        
        val_ids = train_val_ids[val_mask]
        train_ids = train_val_ids[~val_mask]
        
        splits[fold_idx] = {
            'train': train_ids.tolist(),
            'val': val_ids.tolist(),
            'test': test_ids.tolist()
        }
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"✓ Saved splits to {save_path}")
    
    return splits


def extract_patient_features(patient_id: str, features_data: Dict, 
                             feature_type: str) -> np.ndarray:
    """
    Extract features for a patient
    
    Args:
        patient_id: Patient ID
        features_data: Features dict
        feature_type: 'none', 'handcrafted', 'demographics', 'full'
    
    Returns:
        Feature vector (not normalized yet!)
    """
    if feature_type == 'none':
        return np.array([])
    
    pdata = features_data[patient_id]
    
    if feature_type == 'handcrafted':
        return np.array([pdata[f] for f in HAND_FEATURE_ORDER])
    
    elif feature_type == 'demographics':
        return np.array([pdata[f] for f in DEMOGRAPHIC_FEATURES])
    
    elif feature_type == 'full':
        hand_features = np.array([pdata[f] for f in HAND_FEATURE_ORDER])
        demo_features = np.array([pdata[f] for f in DEMOGRAPHIC_FEATURES])
        return np.concatenate([hand_features, demo_features])
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


# =============================================================================
# FEATURE NORMALIZERS (FIXED!)
# =============================================================================

class FeatureNormalizer:
    """
    Proper feature normalization with separate scalers
    
    CRITICAL: CNN slope is already normalized, don't scale it again!
    Handcrafted and demographic features need separate normalization.
    """
    
    def __init__(self):
        self.handcrafted_scaler = None
        self.demographic_scaler = None
        
    def fit(self, patient_ids: List[str], features_data: Dict, feature_type: str):
        """Fit scalers on training data"""
        
        if feature_type == 'none':
            return
        
        if feature_type in ['handcrafted', 'full']:
            # Fit handcrafted scaler
            hand_features_list = []
            for pid in patient_ids:
                hand_feats = np.array([features_data[pid][f] for f in HAND_FEATURE_ORDER])
                hand_features_list.append(hand_feats)
            
            hand_features_array = np.array(hand_features_list)
            self.handcrafted_scaler = StandardScaler()
            self.handcrafted_scaler.fit(hand_features_array)
        
        if feature_type in ['demographics', 'full']:
            # Fit demographic scaler (only age, sex and smoking are categorical)
            demo_features_list = []
            for pid in patient_ids:
                age = features_data[pid]['age']
                demo_features_list.append([age])
            
            demo_features_array = np.array(demo_features_list)
            self.demographic_scaler = StandardScaler()
            self.demographic_scaler.fit(demo_features_array)
    
    def transform(self, patient_id: str, features_data: Dict, 
                  feature_type: str, cnn_slope: float) -> np.ndarray:
        """
        Transform features for one patient
        
        Returns:
            Normalized feature vector: [cnn_slope, normalized_features...]
            Note: cnn_slope is already normalized, we don't touch it!
        """
        if feature_type == 'none':
            return np.array([cnn_slope])
        
        pdata = features_data[patient_id]
        feature_vector = [cnn_slope]  # Already normalized from CNN
        
        if feature_type in ['handcrafted', 'full']:
            # Normalize handcrafted features
            hand_feats = np.array([pdata[f] for f in HAND_FEATURE_ORDER])
            hand_feats_norm = self.handcrafted_scaler.transform(hand_feats.reshape(1, -1))[0]
            feature_vector.extend(hand_feats_norm)
        
        if feature_type in ['demographics', 'full']:
            # Normalize age, keep sex and smoking as-is (categorical)
            age_norm = self.demographic_scaler.transform([[pdata['age']]])[0][0]
            feature_vector.extend([age_norm, pdata['sex'], pdata['smoking_status']])
        
        return np.array(feature_vector)
    
    def get_feature_dim(self, feature_type: str) -> int:
        """Get total feature dimension"""
        if feature_type == 'none':
            return 1  # Just CNN slope
        elif feature_type == 'handcrafted':
            return 1 + len(HAND_FEATURE_ORDER)  # CNN + handcrafted
        elif feature_type == 'demographics':
            return 1 + len(DEMOGRAPHIC_FEATURES)  # CNN + demographics
        elif feature_type == 'full':
            return 1 + len(HAND_FEATURE_ORDER) + len(DEMOGRAPHIC_FEATURES)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")


# =============================================================================
# DATASETS
# =============================================================================

class IPFSliceDataset(Dataset):
    """Dataset for CNN training at slice level"""
    
    def __init__(self, patient_ids: List[str], patient_data: Dict, 
                 features_data: Dict, image_size: Tuple[int, int] = (224, 224),
                 normalize_slope: bool = True, slope_scaler: Optional[StandardScaler] = None,
                 augment: bool = False):
        
        self.patient_ids = patient_ids
        self.patient_data = patient_data
        self.features_data = features_data
        self.image_size = image_size
        self.normalize_slope = normalize_slope
        self.slope_scaler = slope_scaler
        self.augment = augment
        self._cache = {}
        
        # Build slice index
        self.slices = []
        for pid in patient_ids:
            if pid not in patient_data:
                continue
            
            npy_path = Path(features_data[pid].get('npy_path', 
                           f'D:/FrancescoP/ImagingBased-ProgressionPrediction/Dataset/extracted_npy/extracted_npy/{pid}'))
            
            if not npy_path.exists():
                continue
            
            npy_files = sorted(list(npy_path.glob('*.npy')))
            for npy_file in npy_files:
                self.slices.append({
                    'patient_id': pid,
                    'npy_path': npy_file,
                    'slope': patient_data[pid]['slope']
                })
        
        # Fit slope scaler if needed
        if self.normalize_slope and self.slope_scaler is None:
            unique_slopes = {}
            for s in self.slices:
                unique_slopes[s['patient_id']] = s['slope']

            slopes = np.array(list(unique_slopes.values())).reshape(-1, 1)
            # Use StandardScaler - RobustScaler can cause mode collapse by over-centering
            self.slope_scaler = StandardScaler()
            self.slope_scaler.fit(slopes)
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):

        if idx in self._cache:
            return self._cache[idx]
        
        slice_info = self.slices[idx]

        try:
            # Load image (already normalized [0,1] from preprocessing)
            img = np.load(slice_info['npy_path'])

            #Verify dimension
            if img.shape != (224,224):
                raise ValueError(f"Unexpected image shape {img.shape} for {slice_info['npy_path']}")
            
            # Images come preprocessed with varying background intensities
            # This is NATURAL variation from the preprocessing pipeline
            # The CNN should learn to be robust to this (BatchNorm will help)
            
            #Convert to 3 channels if necessary
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=0) # [3, H, W]

            img_tensor = torch.FloatTensor(img)

        except Exception as e:
            print(f"Error loading image {slice_info['npy_path']}: {e}")
            self._cache[idx] = None
            return None
        
        #Normalized Slope
        slope = slice_info['slope']
        if self.normalize_slope:
            slope = self.slope_scaler.transform([[slope]])[0][0]

        result = {
            'image': img_tensor,
            'slope': torch.FloatTensor([slope]),
            'patient_id': slice_info['patient_id'],
            'slice_path': slice_info['npy_path'],
        }

        self._cache[idx] = result

        return result


class PatientBatchSampler(Sampler):
    """Sample batches with multiple slices from same patients"""
    
    def __init__(self, dataset: IPFSliceDataset, patients_per_batch: int = 4, 
                 shuffle: bool = True):
        self.dataset = dataset
        self.patients_per_batch = patients_per_batch
        self.shuffle = shuffle
        
        # Group indices by patient
        self.patient_to_indices = defaultdict(list)
        for idx, slice_info in enumerate(dataset.slices):
            self.patient_to_indices[slice_info['patient_id']].append(idx)
        
        self.patient_ids = list(self.patient_to_indices.keys())
    
    def __iter__(self):
        patients = list(self.patient_ids)

        if self.shuffle:
            np.random.shuffle(patients)
        
        for i in range(0, len(patients), self.patients_per_batch):
            batch_patients = patients[i:i + self.patients_per_batch]
            batch_indices = []
            for pid in batch_patients:
                batch_indices.extend(self.patient_to_indices[pid])
            yield batch_indices
    
    def __len__(self):
        return (len(self.patient_ids) + self.patients_per_batch - 1) // self.patients_per_batch


def patient_group_collate(batch):
    """Collate function for patient-grouped batches"""

    batch = [b for b in batch if b is not None]

    #Default case if batch is empty
    if not batch:
        return {
            'images': torch.empty(0, 1, 224, 224),
            'slopes': torch.empty(0),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long)
        }
    
    # Stack all slices (patient slices are contiguous due to sampler)
    images = torch.stack([item['image'] for item in batch])
    slopes = torch.stack([item['slope'] for item in batch])
    
    # Compute lengths and patient order
    # Since sampler guarantees contiguity, we can count consecutive same IDs
    lengths = []
    pid_order = []
    
    i = 0

    while i < len(batch):
        current_pid = batch[i]['patient_id']
        j = i
        # Count consecutive slices from same patient
        while j < len(batch) and batch[j]['patient_id'] == current_pid:
            j += 1
        
        lengths.append(j - i)
        pid_order.append(current_pid)
        i = j
    
    return {
        'images': images,                                    # (total_slices, C, H, W)
        'slopes': slopes,                                    # (total_slices,)
        'patient_ids': pid_order,                           # List of patient IDs in order
        'lengths': torch.tensor(lengths, dtype=torch.long)  # Number of slices per patient
    }

class ExtremeOversamplingBatchSampler(Sampler):
    """
    Oversample extreme slope cases AGGRESSIVELY
    
    Strategy:
    - Identify extreme cases (top/bottom 25%)
    - Sample them 3-5x more frequently than middle cases
    - Ensures every batch has at least 1-2 extreme cases
    """
    
    def __init__(self, dataset, patients_per_batch=4, 
                 extreme_percentile=25, oversample_factor=4.0, shuffle=True):
        """
        Args:
            dataset: IPFSliceDataset
            patients_per_batch: batch size in patients
            extreme_percentile: % to consider extreme (25 = top/bottom 25%)
            oversample_factor: how much more to sample extremes (4.0 = 4x more)
            shuffle: whether to shuffle
        """
        self.dataset = dataset
        self.patients_per_batch = patients_per_batch
        self.shuffle = shuffle
        
        # Get patient slopes
        patient_slopes = {}
        for pid in dataset.patient_ids:
            slope = dataset.features_data[
                dataset.features_data['Patient'] == pid
            ]['slope'].values[0]
            patient_slopes[pid] = slope
        
        slopes = np.array(list(patient_slopes.values()))
        patient_ids = np.array(list(patient_slopes.keys()))
        
        # Identify extreme cases
        lower_thresh = np.percentile(slopes, extreme_percentile)
        upper_thresh = np.percentile(slopes, 100 - extreme_percentile)
        
        self.extreme_mask = (slopes <= lower_thresh) | (slopes >= upper_thresh)
        self.middle_mask = ~self.extreme_mask
        
        print(f"\n🎯 Oversampling Configuration:")
        print(f"  Extreme threshold: [{lower_thresh:.2f}, {upper_thresh:.2f}]")
        print(f"  Extreme patients: {self.extreme_mask.sum()} ({self.extreme_mask.sum()/len(slopes)*100:.1f}%)")
        print(f"  Middle patients:  {self.middle_mask.sum()} ({self.middle_mask.sum()/len(slopes)*100:.1f}%)")
        print(f"  Oversample factor: {oversample_factor}x")
        
        # Create sampling probabilities
        # Extreme cases get high probability, middle cases get low
        probs = np.ones(len(slopes))
        probs[self.extreme_mask] *= oversample_factor
        probs = probs / probs.sum()  # Normalize
        
        self.patient_ids = patient_ids
        self.sampling_probs = probs
        
        # Calculate expected extreme cases per batch
        expected_extreme = patients_per_batch * (self.extreme_mask.sum() * oversample_factor) / probs.sum()
        print(f"  Expected extreme cases per batch: {expected_extreme:.1f}/{patients_per_batch}")
    
    def __iter__(self):
        """Generate batches with oversampling"""
        n_patients = len(self.patient_ids)
        n_batches = (n_patients + self.patients_per_batch - 1) // self.patients_per_batch
        
        # Generate more batches to ensure all patients seen at least once
        n_batches = int(n_batches * 1.5)  # 50% more batches
        
        for _ in range(n_batches):
            # Sample patients according to probabilities
            # With replacement to allow extreme cases to appear multiple times
            batch_patients = np.random.choice(
                self.patient_ids,
                size=self.patients_per_batch,
                replace=True,  # Allow duplicates (extreme cases can appear multiple times per epoch)
                p=self.sampling_probs
            )
            
            # Get slice indices for these patients
            batch_indices = []
            for pid in batch_patients:
                indices = [
                    i for i, (p, _) in enumerate(self.dataset.slice_info)
                    if p == pid
                ]
                batch_indices.extend(indices)
            
            if len(batch_indices) > 0:
                yield batch_indices
    
    def __len__(self):
        n_patients = len(self.patient_ids)
        n_batches = (n_patients + self.patients_per_batch - 1) // self.patients_per_batch
        return int(n_batches * 1.5)



class CorrectorDataset(Dataset):
    """Dataset for training slope corrector with FIXED normalization"""
    
    def __init__(self, patient_ids: List[str], patient_data: Dict, 
                 features_data: Dict, cnn_slopes: Dict, 
                 feature_type: str = 'full', 
                 normalizer: Optional[FeatureNormalizer] = None):
        """
        Args:
            patient_ids: List of patient IDs
            patient_data: Patient data dict
            features_data: Features dict
            cnn_slopes: Dict {patient_id: mean_cnn_slope} (already normalized!)
            feature_type: 'none', 'handcrafted', 'demographics', 'full'
            normalizer: Pre-fitted FeatureNormalizer (or None to fit new one)
        """
        self.patient_ids = patient_ids
        self.patient_data = patient_data
        self.features_data = features_data
        self.cnn_slopes = cnn_slopes
        self.feature_type = feature_type
        self.normalizer = normalizer
        
        # Build samples
        self.samples = []
        for pid in patient_ids:
            if pid in cnn_slopes and pid in patient_data:
                self.samples.append(pid)
        
        # Fit normalizer if needed
        if self.normalizer is None:
            self.normalizer = FeatureNormalizer()
            self.normalizer.fit(self.samples, features_data, feature_type)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid = self.samples[idx]
        
        # Get CNN slope (already normalized)
        slope_cnn = self.cnn_slopes[pid]
        
        # Get true slope
        slope_true = self.patient_data[pid]['slope']
        
        # Get normalized features
        feature_vector = self.normalizer.transform(pid, self.features_data, 
                                                   self.feature_type, slope_cnn)
        
        return {
            'features': torch.tensor(feature_vector, dtype=torch.float32),
            'slope': torch.tensor(slope_true, dtype=torch.float32),
            'patient_id': pid
        }


# =============================================================================
# MODELS
# =============================================================================

class SpatialAttentionModule(nn.Module):
    """
    Explicit spatial attention that learns to focus on informative regions
    This is better than just relying on the CNN to learn implicitly
    
    From: Comprehensive_Model_comparison_only_full.ipynb
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


def create_lung_mask_attention(img, threshold=-500):
    """
    Create soft attention mask for lung regions
    This can be used as attention guidance during training
    
    Args:
        img: (H, W) CT image in HU
        threshold: HU threshold for lung tissue (default -500)
    
    Returns:
        mask: (H, W) soft attention mask [0, 1]
    """
    # Create binary lung mask
    lung_mask = (img > -900) & (img < -500)
    
    # Dilate slightly to include peri-bronchial regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lung_mask = cv2.dilate(lung_mask.astype(np.uint8), kernel, iterations=2)
    
    # Create soft mask with Gaussian blur
    soft_mask = cv2.GaussianBlur(lung_mask.astype(float), (15, 15), 0)
    
    # Normalize
    soft_mask = soft_mask / (soft_mask.max() + 1e-8)
    
    return soft_mask


class AttentionGuidedLoss(nn.Module):
    """
    Loss that penalizes attention on non-lung regions
    Encourages CNN to focus on relevant anatomy
    
    From: Comprehensive_Model_comparison_only_full.ipynb
    """
    def __init__(self, lung_mask_weight=0.1):
        super().__init__()
        self.lung_mask_weight = lung_mask_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, attention_maps, lung_masks):
        """
        Args:
            predictions: (batch,) predicted slopes
            targets: (batch,) true slopes
            attention_maps: (batch, H, W) attention from Grad-CAM or attention layer
            lung_masks: (batch, H, W) binary lung masks
        
        Returns:
            total_loss
        """
        # Standard prediction loss
        prediction_loss = self.mse_loss(predictions, targets)
        
        # Attention regularization: penalize attention outside lungs
        # attention_maps should sum to 1 per image
        attention_maps = attention_maps / (attention_maps.sum(dim=(1,2), keepdim=True) + 1e-8)
        
        # Compute attention inside vs outside lungs
        attention_inside = (attention_maps * lung_masks).sum(dim=(1,2))
        attention_outside = (attention_maps * (1 - lung_masks)).sum(dim=(1,2))
        
        # Penalize attention outside lungs
        attention_loss = attention_outside.mean()
        
        # Total loss
        total_loss = prediction_loss + self.lung_mask_weight * attention_loss
        
        return total_loss


class ImprovedSliceLevelCNN(nn.Module):
    """
    CNN with explicit spatial attention
    Forces model to learn WHERE to look
    
    From: Comprehensive_Model_comparison_only_full.ipynb
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout: float = 0.3, 
                 attention_temperature: float = 1.0):
        super().__init__()
        self.attention_temperature = attention_temperature
        
        # Backbone with features_only=True
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,  # 3 channels like RGB (stacked grayscale)
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
            nn.Dropout(dropout),
            nn.Linear(feat_dims[-1], 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout * 0.33),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 1, H, W) input images
            return_attention: if True, return attention maps for visualization
        """
        # Extract features
        features = self.backbone(x)
        last_feature_map = features[-1]  # (B, C, h, w)
        
        # Apply spatial attention
        attended_features, attention_map = self.spatial_attention(last_feature_map)
        
        # Global pooling
        pooled = self.global_pool(attended_features).flatten(1)  # (B, C)
        
        # Prediction (with temperature scaling during inference)
        output = self.head(pooled).squeeze(-1)  # (B,)
        # Scale predictions by temperature to encourage diversity
        output = output * self.attention_temperature
        
        if return_attention:
            return output, attention_map
        return output


class ImprovedSlopeCorrector(nn.Module):
    """
    Improved MLP corrector with proper architecture
    
    Features:
    - Batch normalization
    - Dropout regularization
    - Deeper architecture
    - Residual connections
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rates: List[float] = [0.3, 0.2, 0.1]):
        super().__init__()
        
        assert len(hidden_dims) == len(dropout_rates)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# Create corrector for each approach
class SlopeCorrectorCNNOnly(ImprovedSlopeCorrector):
    def __init__(self, **kwargs):
        super().__init__(input_dim=1, **kwargs)  # Just CNN slope


class SlopeCorrectorCNNHandcrafted(ImprovedSlopeCorrector):
    def __init__(self, n_handcrafted=9, **kwargs):
        super().__init__(input_dim=1 + n_handcrafted, **kwargs)


class SlopeCorrectorCNNDemographics(ImprovedSlopeCorrector):
    def __init__(self, n_demographics=3, **kwargs):
        super().__init__(input_dim=1 + n_demographics, **kwargs)


class SlopeCorrectorFull(ImprovedSlopeCorrector):
    def __init__(self, n_handcrafted=9, n_demographics=3, **kwargs):
        super().__init__(input_dim=1 + n_handcrafted + n_demographics, **kwargs)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def save_fold_results(results: Dict, save_path: Path):
    """Save fold results"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


def load_fold_results(load_path: Path) -> Dict:
    """Load fold results"""
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression metrics"""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def compute_inverse_frequency_weights(slopes: np.ndarray, n_bins: int = 6) -> Dict[str, np.ndarray]:
    """
    Compute inverse frequency weights for slope balancing.
    
    Assigns higher weights to rare slope values and lower weights to common ones.
    This forces the model to pay more attention to extreme/rare cases.
    
    Args:
        slopes: Array of slope values for all patients
        n_bins: Number of bins to divide slope distribution
    
    Returns:
        Dictionary with 'weights' (per-sample weights) and 'bin_info' for diagnostics
    """
    # Create histogram bins
    hist, bin_edges = np.histogram(slopes, bins=n_bins)
    
    # Compute inverse frequency weights
    # Add small epsilon to avoid division by zero
    bin_weights = 1.0 / (hist + 1e-8)
    
    # Apply moderate power to strengthen weights without overfitting
    # Exponent 1.5 balances: strong enough to break mode collapse, stable for long training
    # This gives ~150x difference (vs 868x with ^2.0, or 29x with ^1.0)
    bin_weights = bin_weights ** 1.5
    
    # Normalize so mean weight = 1.0 (keeps loss magnitude similar)
    bin_weights = bin_weights / bin_weights.mean()
    
    # Assign each sample to a bin
    bin_indices = np.digitize(slopes, bin_edges[:-1], right=False) - 1
    # Clamp to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Map bin weights to samples
    sample_weights = bin_weights[bin_indices]
    
    # Create diagnostic info
    bin_info = []
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        bin_info.append({
            'range': (float(bin_edges[i]), float(bin_edges[i+1])),
            'count': int(hist[i]),
            'weight': float(bin_weights[i]),
            'samples_pct': float(hist[i] / len(slopes) * 100)
        })
    
    return {
        'weights': sample_weights,
        'bin_info': bin_info,
        'bin_edges': bin_edges
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_batch_images(batch_data: Dict, save_dir: str, batch_idx: int = 0):
    """
    Salva le immagini di un batch in una cartella senza bloccare il terminale
    
    Args:
        batch_data: Dict con 'image', 'slope', 'patient_id'
        save_dir: Cartella dove salvare le immagini
        batch_idx: Indice del batch (per naming)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    images = batch_data['image']  # [B, 3, H, W]
    slopes = batch_data['slope']  # [B]
    patient_ids = batch_data['patient_id']
    
    # Salva ogni immagine singolarmente
    for i in range(len(images)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Get image [3, H, W] -> [H, W] (primo canale)
        img = images[i][0].cpu().numpy()
        
        # Denormalize se necessario
        if img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Batch {batch_idx} - Sample {i}\nPatient: {patient_ids[i]}\nSlope: {slopes[i].item():.4f}', 
                     fontsize=10)
        ax.axis('off')
        
        filename = save_path / f'batch{batch_idx:04d}_sample{i:02d}_{patient_ids[i][:10]}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Salvate {len(images)} immagini in {save_dir}/")


def save_dataset_samples(dataset: IPFSliceDataset, save_dir: str, 
                         num_samples: int = 20, prefix: str = "sample"):
    """
    Salva samples random dal dataset in una cartella
    
    Args:
        dataset: IPFSliceDataset instance
        save_dir: Cartella dove salvare
        num_samples: Numero di samples da salvare
        prefix: Prefisso per i nomi file
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    saved_count = 0
    for idx in indices:
        sample = dataset[idx]
        
        if sample is None:
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Get image [3, H, W] -> [H, W]
        img = sample['image'][0].cpu().numpy()
        
        # Denormalize
        if img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
        
        ax.imshow(img, cmap='gray')
        
        patient_id = sample['patient_id']
        slope = sample['slope'].item()
        
        title = f'Patient: {patient_id}\nSlope: {slope:.4f}'
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        filename = save_path / f'{prefix}_{saved_count:03d}_{patient_id[:10]}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        saved_count += 1
    
    print(f"✓ Salvati {saved_count} samples in {save_dir}/")

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
# FOCAL LOSS FOR REGRESSION
# =============================================================================

class FocalMSELoss(nn.Module):
    """
    Focal Loss adapted for regression tasks.
    
    Focuses learning on hard examples by down-weighting easy predictions.
    For regression, "difficulty" is measured by the absolute error.
    
    Loss = |y_true - y_pred|^gamma * (y_true - y_pred)^2
    
    Args:
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 → standard MSE
               gamma=2 → typical focal loss (recommended)
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma
        self.reduction = 'none'  # For compatibility with weighted training
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size,) predicted values
            targets: (batch_size,) true values
        
        Returns:
            Scalar loss value
        """
        mse_loss = (predictions - targets) ** 2
        abs_error = torch.abs(predictions - targets)
        
        # Normalize abs_error to [0, 1] for stable gradients
        # Use detach to prevent backprop through normalization
        abs_error_norm = abs_error / (abs_error.max().detach() + 1e-8)
        
        # Focal weight: higher weight for larger errors
        focal_weight = abs_error_norm ** self.gamma
        
        focal_loss = focal_weight * mse_loss
        
        return focal_loss.mean()
    
    def forward_without_reduction(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without reduction (for sample weighting).
        
        Args:
            predictions: (batch_size,) predicted values
            targets: (batch_size,) true values
        
        Returns:
            Per-sample loss: (batch_size,) tensor
        """
        mse_loss = (predictions - targets) ** 2
        abs_error = torch.abs(predictions - targets)
        
        # Normalize abs_error
        abs_error_norm = abs_error / (abs_error.max().detach() + 1e-8)
        
        # Focal weight
        focal_weight = abs_error_norm ** self.gamma
        
        focal_loss = focal_weight * mse_loss
        
        return focal_loss  # No reduction!


class FocalHuberLoss(nn.Module):
    """
    Focal Loss with Huber Loss for robustness to outliers.
    
    Combines benefits of:
    - Huber loss: Robust to extreme outliers
    - Focal loss: Focus on hard examples
    
    Args:
        delta: Huber loss threshold
        gamma: Focal loss focusing parameter
    """
    def __init__(self, delta: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        self.reduction = 'none'  # For compatibility with weighted training
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size,) predicted values
            targets: (batch_size,) true values
        
        Returns:
            Scalar loss value
        """
        huber_loss = self.huber(predictions, targets)
        abs_error = torch.abs(predictions - targets)
        
        # Normalize abs_error
        abs_error_norm = abs_error / (abs_error.max().detach() + 1e-8)
        
        # Focal weight
        focal_weight = abs_error_norm ** self.gamma
        
        focal_huber_loss = focal_weight * huber_loss
        
        return focal_huber_loss.mean()
    
    def forward_without_reduction(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without reduction (for sample weighting).
        
        Args:
            predictions: (batch_size,) predicted values
            targets: (batch_size,) true values
        
        Returns:
            Per-sample loss: (batch_size,) tensor
        """
        huber_loss = self.huber(predictions, targets)
        abs_error = torch.abs(predictions - targets)
        
        # Normalize abs_error
        abs_error_norm = abs_error / (abs_error.max().detach() + 1e-8)
        
        # Focal weight
        focal_weight = abs_error_norm ** self.gamma
        
        focal_huber_loss = focal_weight * huber_loss
        
        return focal_huber_loss  # No reduction!

class FixedFocalMSELoss(nn.Module):
    """
    Focal MSE WITHOUT normalization that cancels sample weights
    
    Key difference: No division by max(error) that was canceling weights
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """Standard forward with mean reduction"""
        mse_loss = (predictions - targets) ** 2
        abs_error = torch.abs(predictions - targets)
        
        # NO NORMALIZATION! Let the weights flow naturally
        focal_weight = abs_error ** self.gamma
        focal_loss = focal_weight * mse_loss
        
        return focal_loss.mean()
    
    def forward_without_reduction(self, predictions, targets):
        """For sample weighting"""
        mse_loss = (predictions - targets) ** 2
        abs_error = torch.abs(predictions - targets)
        
        # NO NORMALIZATION!
        focal_weight = abs_error ** self.gamma
        focal_loss = focal_weight * mse_loss
        
        return focal_loss  # No reduction


def compute_exponential_weights(slopes: np.ndarray, n_bins: int = 6, 
                                strength: float = 2.0) -> Dict:
    """
    Compute EXPONENTIAL inverse frequency weights (stronger than linear)
    
    Args:
        slopes: Array of slope values
        n_bins: Number of bins
        strength: Exponent for weighting (2.0 = quadratic, 3.0 = cubic)
                 Higher = more aggressive weighting of rare cases
    
    Returns:
        Dict with weights and diagnostics
    """
    # Create histogram
    hist, bin_edges = np.histogram(slopes, bins=n_bins)
    
    # Exponential inverse frequency
    # freq = 50 samples → weight = (1/50)^2 = 0.0004 (if strength=2)
    # freq = 5 samples → weight = (1/5)^2 = 0.04 (100x more!)
    bin_weights = (1.0 / (hist + 1e-8)) ** strength
    
    # Normalize
    bin_weights = bin_weights / bin_weights.mean()
    
    # Assign weights
    bin_indices = np.digitize(slopes, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    sample_weights = bin_weights[bin_indices]
    
    # Diagnostics
    bin_info = []
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        bin_info.append({
            'range': (float(bin_edges[i]), float(bin_edges[i+1])),
            'count': int(hist[i]),
            'weight': float(bin_weights[i]),
            'samples_pct': float(hist[i] / len(slopes) * 100),
            'total_contribution': float(bin_weights[i] * hist[i])  # How much this bin contributes to total loss
        })
    
    print(f"\n⚖️  Exponential Sample Weighting (strength={strength}):")
    print(f"{'─'*90}")
    print(f"{'Range':<20} {'Count':<8} {'Weight':<12} {'%':<8} {'Loss Contrib':<15}")
    print(f"{'─'*90}")
    for info in bin_info:
        print(f"[{info['range'][0]:6.2f}, {info['range'][1]:6.2f})  "
              f"{info['count']:>6d}  "
              f"{info['weight']:>10.3f}  "
              f"{info['samples_pct']:>6.1f}%  "
              f"{info['total_contribution']:>13.2f}")
    print(f"{'─'*90}")
    print(f"Weight range: {bin_weights.min():.3f} - {bin_weights.max():.3f} "
          f"(ratio: {bin_weights.max()/bin_weights.min():.1f}x)")
    
    return {
        'weights': sample_weights,
        'bin_info': bin_info,
        'bin_edges': bin_edges
    }

# =============================================================================
# STRATIFIED BATCH SAMPLER
# =============================================================================

class StratifiedPatientSampler(Sampler):
    """
    Stratified batch sampling to ensure diverse slope ranges in each batch.
    
    Strategy:
    1. Bin slopes into quantiles (e.g., quartiles)
    2. Sample patients from each bin proportionally
    3. Ensure each batch has representatives from all bins
    
    This forces the model to see diverse progression patterns in every batch,
    preventing mode collapse to the majority class.
    
    Args:
        dataset: IPFSliceDataset with patient_ids and slopes
        n_bins: Number of bins for stratification (default: 4 for quartiles)
        batch_size: Number of patients per batch
        shuffle: Whether to shuffle within bins
    """
    def __init__(self, dataset, n_bins: int = 4, batch_size: int = 8, shuffle: bool = True):
        self.dataset = dataset
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get unique patients and their slopes
        patient_slopes = {}
        for slice_info in dataset.slices:
            patient_id = slice_info['patient_id']
            slope = slice_info['slope']
            if patient_id not in patient_slopes:
                patient_slopes[patient_id] = slope
        
        self.patient_ids = list(patient_slopes.keys())
        self.slopes = np.array([patient_slopes[pid] for pid in self.patient_ids])
        
        # Create stratified bins
        self._create_bins()
        
        # Map patient_id to indices in dataset
        self.patient_to_indices = defaultdict(list)
        for idx, slice_info in enumerate(dataset.slices):
            patient_id = slice_info['patient_id']
            self.patient_to_indices[patient_id].append(idx)
    
    def _create_bins(self):
        """Bin patients by slope quantiles."""
        # Use quantiles for balanced bins
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(self.slopes, quantiles)
        bin_edges[-1] += 1e-6  # Include maximum value
        
        # Assign patients to bins
        self.bins = [[] for _ in range(self.n_bins)]
        for patient_id, slope in zip(self.patient_ids, self.slopes):
            bin_idx = np.digitize(slope, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))  # Clamp
            self.bins[bin_idx].append(patient_id)
        
        # Print bin statistics
        print(f"\n{'='*80}")
        print(f"Stratified Sampling - Bin Statistics:")
        print(f"{'='*80}")
        for i, bin_patients in enumerate(self.bins):
            bin_slopes = [self.slopes[self.patient_ids.index(pid)] for pid in bin_patients]
            print(f"Bin {i}: {len(bin_patients):3d} patients | "
                  f"Slope range: [{min(bin_slopes):6.2f}, {max(bin_slopes):6.2f}] | "
                  f"Mean: {np.mean(bin_slopes):6.2f}")
        print(f"{'='*80}\n")
    
    def __iter__(self):
        """Generate batches with stratified sampling."""
        # Shuffle within bins if requested
        if self.shuffle:
            bins = [np.random.permutation(bin_list).tolist() for bin_list in self.bins]
        else:
            bins = [list(bin_list) for bin_list in self.bins]
        
        # Calculate samples per bin per batch
        samples_per_bin = max(1, self.batch_size // self.n_bins)
        
        # Generate batches
        while any(len(b) > 0 for b in bins):
            batch_patients = []
            
            # Sample from each bin
            for bin_list in bins:
                n_sample = min(samples_per_bin, len(bin_list))
                if n_sample > 0:
                    batch_patients.extend(bin_list[:n_sample])
                    del bin_list[:n_sample]
            
            # If batch too small, break
            if len(batch_patients) < self.batch_size // 2:
                break
            
            # Convert patient IDs to slice indices for this batch
            batch_indices = []
            for patient_id in batch_patients:
                batch_indices.extend(self.patient_to_indices[patient_id])
            
            # Yield this batch
            yield batch_indices
    
    def __len__(self):
        """Approximate number of batches (not exact due to stratification)."""
        total_patients = len(self.patient_ids)
        return (total_patients + self.batch_size - 1) // self.batch_size


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

class PredictionTracker:
    """
    Track prediction statistics across epochs for diagnostic analysis.
    
    Tracks:
    - Prediction distribution (mean, std, min, max, quantiles)
    - Prediction variance (diversity metric)
    - Correlation with true values
    - Residual statistics
    """
    def __init__(self):
        self.history = {
            'epoch': [],
            'pred_mean': [],
            'pred_std': [],
            'pred_min': [],
            'pred_max': [],
            'pred_q25': [],
            'pred_q75': [],
            'true_mean': [],
            'true_std': [],
            'correlation': [],
            'residual_mean': [],
            'residual_std': [],
            'mode_collapse_score': []  # Ratio of pred_std / true_std
        }
    
    def update(self, epoch: int, predictions: np.ndarray, targets: np.ndarray):
        """Update statistics for current epoch."""
        # Flatten to 1D arrays
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        self.history['epoch'].append(epoch)
        
        # Prediction statistics
        self.history['pred_mean'].append(np.mean(predictions))
        self.history['pred_std'].append(np.std(predictions))
        self.history['pred_min'].append(np.min(predictions))
        self.history['pred_max'].append(np.max(predictions))
        self.history['pred_q25'].append(np.percentile(predictions, 25))
        self.history['pred_q75'].append(np.percentile(predictions, 75))
        
        # Target statistics
        self.history['true_mean'].append(np.mean(targets))
        self.history['true_std'].append(np.std(targets))
        
        # Correlation
        if len(predictions) > 1 and np.std(predictions) > 1e-8 and np.std(targets) > 1e-8:
            corr = np.corrcoef(predictions, targets)[0, 1]
            self.history['correlation'].append(corr if not np.isnan(corr) else 0.0)
        else:
            self.history['correlation'].append(0.0)
        
        # Residuals
        residuals = predictions - targets
        self.history['residual_mean'].append(np.mean(residuals))
        self.history['residual_std'].append(np.std(residuals))
        
        # Mode collapse score: lower is worse (predictions less diverse than truth)
        mode_collapse = np.std(predictions) / (np.std(targets) + 1e-8)
        self.history['mode_collapse_score'].append(mode_collapse)
    
    def print_summary(self, epoch: int):
        """Print diagnostic summary for current epoch."""
        if not self.history['epoch'] or self.history['epoch'][-1] != epoch:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 Diagnostic Summary - Epoch {epoch}")
        print(f"{'='*80}")
        print(f"Predictions: μ={self.history['pred_mean'][-1]:6.2f}, "
              f"σ={self.history['pred_std'][-1]:5.2f}, "
              f"range=[{self.history['pred_min'][-1]:6.2f}, {self.history['pred_max'][-1]:6.2f}]")
        print(f"True values: μ={self.history['true_mean'][-1]:6.2f}, "
              f"σ={self.history['true_std'][-1]:5.2f}")
        print(f"Correlation: r={self.history['correlation'][-1]:5.3f}")
        print(f"Mode collapse score: {self.history['mode_collapse_score'][-1]:5.3f} "
              f"(1.0 = perfect diversity, <0.5 = severe collapse)")
        print(f"{'='*80}\n")
    
    def save(self, filepath: Path):
        """Save tracking history to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved prediction tracking to {filepath}")
    
    def plot(self, save_path: Path):
        """Create diagnostic plots."""
        if len(self.history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = self.history['epoch']
        
        # Plot 1: Prediction vs True statistics
        ax = axes[0, 0]
        ax.plot(epochs, self.history['pred_mean'], label='Pred Mean', marker='o')
        ax.plot(epochs, self.history['true_mean'], label='True Mean', linestyle='--')
        ax.fill_between(epochs, 
                        np.array(self.history['pred_mean']) - np.array(self.history['pred_std']),
                        np.array(self.history['pred_mean']) + np.array(self.history['pred_std']),
                        alpha=0.3, label='Pred ±σ')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Prediction Statistics vs Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prediction range
        ax = axes[0, 1]
        ax.plot(epochs, self.history['pred_min'], label='Min', marker='v')
        ax.plot(epochs, self.history['pred_max'], label='Max', marker='^')
        ax.fill_between(epochs, self.history['pred_q25'], self.history['pred_q75'],
                        alpha=0.3, label='IQR (Q25-Q75)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Prediction Value')
        ax.set_title('Prediction Range Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mode collapse score
        ax = axes[1, 0]
        ax.plot(epochs, self.history['mode_collapse_score'], marker='o', color='red')
        ax.axhline(y=1.0, color='green', linestyle='--', label='Perfect (1.0)')
        ax.axhline(y=0.5, color='orange', linestyle='--', label='Warning (0.5)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score (pred_σ / true_σ)')
        ax.set_title('Mode Collapse Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Correlation
        ax = axes[1, 1]
        ax.plot(epochs, self.history['correlation'], marker='o', color='purple')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Prediction-Truth Correlation')
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved diagnostic plots to {save_path}")


def analyze_batch_diversity(dataloader, num_batches: int = 10):
    """
    Analyze slope diversity within batches.
    
    Useful for validating stratified sampling is working correctly.
    
    Args:
        dataloader: DataLoader to analyze
        num_batches: Number of batches to check
    """
    print(f"\n{'='*80}")
    print(f"Batch Diversity Analysis")
    print(f"{'='*80}")
    
    batch_stats = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        if batch is None:
            continue
        
        # Use 'slopes' (plural) from patient_group_collate
        slopes = batch['slopes'].cpu().numpy()
        
        stats_dict = {
            'batch': i,
            'n_samples': len(slopes),
            'mean': np.mean(slopes),
            'std': np.std(slopes),
            'min': np.min(slopes),
            'max': np.max(slopes),
            'range': np.max(slopes) - np.min(slopes)
        }
        batch_stats.append(stats_dict)
        
        print(f"Batch {i:2d}: n={len(slopes):3d} | "
              f"μ={stats_dict['mean']:6.2f}, σ={stats_dict['std']:5.2f} | "
              f"range=[{stats_dict['min']:6.2f}, {stats_dict['max']:6.2f}] | "
              f"span={stats_dict['range']:5.2f}")
    
    # Summary statistics
    df = pd.DataFrame(batch_stats)
    print(f"\n{'-'*80}")
    print(f"Summary across {len(batch_stats)} batches:")
    print(f"Average batch std:   {df['std'].mean():5.2f} ± {df['std'].std():5.2f}")
    print(f"Average batch range: {df['range'].mean():5.2f} ± {df['range'].std():5.2f}")
    print(f"{'='*80}\n")
    
    return df

