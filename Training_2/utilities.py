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
            self.slope_scaler = RobustScaler()  # Changed from StandardScaler - immune to outliers
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
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
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
        
        # Prediction
        output = self.head(pooled).squeeze(-1)  # (B,)
        
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
