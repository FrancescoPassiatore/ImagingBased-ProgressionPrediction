import os
from pyexpat import features
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torchvision import models,transforms
import pickle
from collections import defaultdict
import timm
import matplotlib.pyplot as plt
from scipy import stats
import glob
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import logging

logger = logging.getLogger(__name__)
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


class PatientBatchSampler(Sampler):
    """
    Sample batches such that all slices from the same patient are in the same batch
    """
    def __init__(self, dataset, patients_per_batch=4, shuffle=True):
        self.ds = dataset
        self.shuffle = shuffle
        self.ppb = patients_per_batch

    def __iter__(self):
        patients = list(self.ds.patients)
        if self.shuffle:
            random.shuffle(patients)

        for i in range(0, len(patients), self.ppb):
            batch_pids = patients[i:i+self.ppb]
            idxs = []
            for pid in batch_pids:
                pidxs = list(self.ds.patient_to_indices[pid])
                idxs.extend(pidxs)
            yield idxs

    def __len__(self):
        return (len(self.ds.patients) + self.ppb - 1) // self.ppb


def patient_group_collate(batch):
    """
    Collate function that groups slices by patient
    """
    # Filtra elementi None
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return {
            'images': torch.empty(0, 3, 224, 224),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long),
            'slice_paths': [],
            'gt_labels': torch.empty(0)
        }

    # Estrai tensori
    images = torch.stack([b['image'] for b in batch])
    slice_paths = [b['slice_path'] for b in batch]
    gt_labels = torch.tensor([b['gt_has_progressed'] for b in batch])

    # Group slices by patient
    lengths, pid_order = [], []
    i = 0
    while i < len(batch):
        pid = batch[i]['patient_id']
        j = i
        while j < len(batch) and batch[j]['patient_id'] == pid:
            j += 1
        lengths.append(j - i)
        pid_order.append(pid)
        i = j

    return {
        'images': images,
        'patient_ids': pid_order,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'slice_paths': slice_paths,
        'gt_labels': gt_labels
    }


class IPFDataLoader:
    """Load patient data from CSV and NPY files"""
    
    def __init__(self, csv_path: str, features_path: str, npy_dir: str):
        self.df_gt = pd.read_csv(csv_path)
        self.df_features= pd.read_csv(features_path)
        self.npy_dir = Path(npy_dir)
        self._verify_npy_availability()


    def _verify_npy_availability(self):
        """Verify that each patient in the CSV has a folder with .npy files"""
        patients_csv = set(self.df_gt['PatientID'].unique())
        patients_npy = set([d for d in os.listdir(self.npy_dir)
                           if os.path.isdir(os.path.join(self.npy_dir, d))])

        missing = patients_csv - patients_npy
        extra = patients_npy - patients_csv

        if missing:
            print(f"⚠️  {len(missing)} patients in CSV without NPY folder: {list(missing)[:5]}...")
        if extra:
            print(f"ℹ️  {len(extra)} NPY folders not in CSV (will be ignored)")

        available = patients_csv & patients_npy
        print(f"✅ {len(available)} patients with complete data (CSV + NPY)")
        
    def get_patient_data(self) -> Tuple[Dict, Dict]:
        """
        Load patient data and features
        
        Returns:
            patient_data: Dict with {patient_id: {'weeks', 'fvc_values', 'intercept', 'slope'}}
            features_data: Dict with {patient_id: {feature: value}}
        """
        # Load main CSV
        df = self.df_gt
        
        # Load features
        features_df = self.df_features

        
        
        # Build patient_data dictionary
        patient_data = {}
        for patient_id in df['PatientID'].unique():

            gt_label = int(df[df['PatientID'] == patient_id]['has_progressed'])


            #Get CT scans
            # Check if patient has NPY files
            patient_npy_folder = os.path.join(self.npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue

            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue
            
            patient_data[patient_id] = {
                'gt_has_progressed': gt_label,
                'slices': npy_files,
                'n_slices': len(npy_files)
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
    

class CTSliceLoaderPatients(Dataset):
    #Load CT slices from NPY files

    def __init__ (self, patient_data, transform=None):
        
        self.transform = transform
        # Build index: list of (patient_id, slice_path, gt_label)
        self.slice_data = []
        self.patients = set()
        self.patient_to_indices = defaultdict(list)

        idx = 0
        for patient_id, patient_info in patient_data.items():
            self.patients.add(patient_id)
            gt_label = patient_info['gt_has_progressed']
            
            for slice_path in patient_info['slices']:
                self.slice_data.append({
                    'patient_id': patient_id,
                    'slice_path': slice_path,
                    'gt_has_progressed': int(gt_label)
                })
                self.patient_to_indices[patient_id].append(idx)
                idx += 1
        
        print(f"Dataset created: {len(self.slice_data)} slices from {len(self.patients)} patients")

    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):

        item = self.slice_data[idx]

        slice_img = np.load(item['slice_path'])        

        # Convert grayscale to 3-channel RGB for pretrained models
        # Image is already [0,1], so we just replicate channels
        if len(slice_img.shape) == 2:
            slice_img = np.stack([slice_img] * 3, axis=-1)  # (H, W) -> (H, W, 3)
        
        # Convert to tensor: (H, W, 3) -> (3, H, W)
        slice_tensor = torch.from_numpy(slice_img).permute(2, 0, 1).float()
    
        
        # Apply transforms (should include ImageNet normalization)
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        
        # Visualize transformed image (denormalize for display)
        slice_np = slice_tensor.cpu().numpy()
        # Denormalize ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        slice_denorm = slice_np * std + mean
        slice_denorm = np.clip(slice_denorm, 0, 1)
        slice_denorm = np.transpose(slice_denorm, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
        
  
        """
        # Convert to 3 channels (grayscale -> RGB) for pre-trained models
        if len(slice_img.shape) == 2:
            print(f"Slice shape before conversion: {slice_img.shape}")
            slice_img = np.stack([slice_img] * 3, axis=0)  # (H, W) -> (3, H, W)
        elif len(slice_img.shape) == 3 and slice_img.shape[-1] == 1:
            print(f"Slice shape before conversion: {slice_img.shape}")
            slice_img = np.repeat(slice_img, 3, axis=-1)  # (H, W, 1) -> (H, W, 3)
            slice_img = np.transpose(slice_img, (2, 0, 1))  # (H, W, 3) -> (3, H, W)

        slice_tensor = torch.from_numpy(slice_img).float()
        print("-----")
        print("After loading slice:")
        print(f"Slice shape after conversion to tensor: {slice_tensor.shape}")
        print(f"Slice tensor dtype: {slice_tensor.dtype}")
        print(f"Slice tensor min/max: {slice_tensor.min()}/{slice_tensor.max()}")
        print(f"Slice tensor mean/std: {slice_tensor.mean()}/{slice_tensor.std()}")
        print("-----")

        # Apply transforms
        if self.transform:
            slice_tensor = self.transform(slice_tensor)


        print("After applying transforms:")
        print(f"Slice tensor shape after transforms: {slice_tensor.shape}")
        print(f"Slice tensor dtype after transforms: {slice_tensor.dtype}")
        #Show image for debugging
        print(f"Slice tensor min/max after transforms: {slice_tensor.min()}/{slice_tensor.max()}")
        print(f"Slice tensor mean/std after transforms: {slice_tensor.mean()}/{slice_tensor.std()}")
        print("Displaying transformed slice image for visual inspection:")
        slice_np = slice_tensor.numpy()
        if slice_np.shape[0] == 3:
            slice_np = np.transpose(slice_np, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
        plt.title(f"Patient ID: {item['patient_id']}")
        plt.imshow(slice_np.astype(np.uint8))
        plt.show()
        print("-----")
    """

        return {
            'image': slice_tensor,
            'patient_id': item['patient_id'],
            'slice_path': item['slice_path'],
            'gt_has_progressed': item['gt_has_progressed']
        }



class CTFMFeatureExtractor:
    """
    Extract 3D volume-level embeddings using the CT-FM foundation model.

    Model: project-lighter/ct_fm_feature_extractor (SegResEncoder backbone)
    Source: https://huggingface.co/project-lighter/ct_fm_feature_extractor

    Preprocessing (matches model card):
        1. EnsureType / channel-first
        2. Orientation → SPL
        3. ScaleIntensityRange: HU [hu_min, hu_max] → [0, 1], clipped
        4. CropForeground (removes air/background)
        5. SlidingWindowInferer for large volumes (optional)

    Output: one 512-dim float32 vector per patient (adaptive_avg_pool3d).
    """

    FEATURE_DIM = 512

    def __init__(
            self,
            hu_min:float=-1024,
            hu_max:float =2048,
            device:str ='cuda',
            sliding_window_size: tuple = (96, 96, 96),
            sliding_window_overlap: float = 0.25,
            pretrained_name: str = "project-lighter/ct_fm_feature_extractor",
    ):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.device = device
        self.sliding_window_size = sliding_window_size
        self.sliding_window_overlap = sliding_window_overlap
        self.pretrained_name = pretrained_name
        
        self.model = None
        self.preprocess = None

    def _load_model(self):
        """Download and initialise CT-FM (called once)."""
        try:
            from lighter_zoo import SegResEncoder
        except ImportError:
            raise ImportError(
                "lighter_zoo is required for CT-FM feature extraction.\n"
                "Install with:  pip install lighter_zoo -U"
            )

        logger.info(f"Loading CT-FM from '{self.pretrained_name}'…")
        model = SegResEncoder.from_pretrained(self.pretrained_name)
        model.eval()
        model.to(self.device)
        logger.info("CT-FM model loaded.")
        return model


    def _build_preprocess_pipeline(self):
        """Build the MONAI preprocessing pipeline as described in the model card."""
        from monai.transforms import (
            Compose, EnsureType, Orientation,
            ScaleIntensityRange, CropForeground
        )
        return Compose([
            EnsureType(),
            Orientation(axcodes="SPL"),
            ScaleIntensityRange(
                a_min=self.hu_min,
                a_max=self.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            CropForeground()
        ])

    def _ensure_loaded(self):
        if self.model is None:
            self.model = self._load_model()
        if self.preprocess is None:
            self.preprocess = self._build_preprocess_pipeline()

    def _extract_single(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract a 512-dim embedding from one 3D CT volume.

        Args:
            volume: numpy array, shape (D, H, W) or (1, D, H, W), HU values

        Returns:
            embedding: numpy array, shape (512,)
        """
        self._ensure_loaded()

        # Ensure channel dimension: (1, D, H, W)
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]   # → (1, D, H, W)

        # Apply MONAI preprocessing (works on tensors / numpy arrays)
        tensor = torch.from_numpy(volume.astype(np.float32))
        tensor = self.preprocess(tensor)       # (1, D', H', W')

        # Batch dimension: (1, 1, D', H', W')
        batch = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # CT-FM returns a list of feature maps; take the last (deepest)
            outputs = self.model(batch)        # list of tensors
            last_feat = outputs[-1]            # (1, C, d, h, w)

            # Global average pool → (1, C) → (C,)
            embedding = torch.nn.functional.adaptive_avg_pool3d(last_feat, 1)
            embedding = embedding.squeeze().cpu().numpy()  # (512,)

        return embedding.astype(np.float32)
    

    def extract_features(
        self,
        patient_data: dict,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Extract CT-FM embeddings for all patients.

        Args:
            patient_data: dict mapping patient_id → dict with keys:
                            'volume' (np.ndarray, shape D×H×W, HU values)
                            'label'  (int, 0 or 1)
            save_path: if provided, cache the result as a CSV (skips
                       already-processed patients on resume)

        Returns:
            DataFrame with columns:
                patient_id | label | volume_feature_0 | ... | volume_feature_511
        """
        save_path = Path(save_path) if save_path is not None else None

        # ---- Resume: load existing results ----
        existing_rows = {}
        if save_path and save_path.exists():
            existing_df = pd.read_csv(save_path)
            for _, row in existing_df.iterrows():
                existing_rows[row['patient_id']] = row.to_dict()
            logger.info(f"Resuming: {len(existing_rows)} patients already cached.")
            print(f"  Resuming CT-FM extraction: {len(existing_rows)}/{len(patient_data)} patients cached.")

        rows = []
        patient_ids = sorted(patient_data.keys())
        total = len(patient_ids)

        for idx, pid in enumerate(patient_ids):
            if pid in existing_rows:
                rows.append(existing_rows[pid])
                continue

            print(f"  [{idx+1}/{total}] Extracting CT-FM features for patient {pid}…", end=' ')
            try:
                info = patient_data[pid]
                volume = info['volume']   # (D, H, W) numpy array in HU
                label  = info['label']

                embedding = self._extract_single(volume)  # (512,)

                row = {'patient_id': pid, 'label': int(label)}
                for i, val in enumerate(embedding):
                    row[f'volume_feature_{i}'] = float(val)
                rows.append(row)
                print(f"OK  (embedding shape: {embedding.shape})")

            except Exception as e:
                logger.error(f"Failed for patient {pid}: {e}")
                print(f"FAILED: {e}")
                # Insert NaN row so the patient can be identified and fixed
                row = {'patient_id': pid, 'label': int(info.get('label', -1))}
                for i in range(self.FEATURE_DIM):
                    row[f'volume_feature_{i}'] = float('nan')
                rows.append(row)

            # Save progress every 10 patients
            if save_path and (idx + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(save_path, index=False)
                print(f"    → Progress saved ({idx+1}/{total})")

        df = pd.DataFrame(rows)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n  CT-FM features saved to: {save_path}")
            print(f"  Shape: {df.shape}  (patients × features)")

        # Sanity check
        n_nan = df[[f'volume_feature_{i}' for i in range(self.FEATURE_DIM)]].isnull().any(axis=1).sum()
        if n_nan > 0:
            logger.warning(f"{n_nan} patients have NaN embeddings – check logs above.")
            print(f"  ⚠ WARNING: {n_nan} patients have NaN embeddings.")

        return df

class PatientSliceDataset(Dataset):
    """
    Dataset that groups slices by patient for progression prediction
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        patient_ids: list,
        hand_feature_cols: List[str] = None,
        demo_feature_cols: List[str] = None,
        encoding_info: dict = None
    ):
        """
        Args:
            features_df: DataFrame with columns [patient_id, slice_index, gt_has_progressed, cnn_feature_0, ...]
            patient_ids: Optional list of patient IDs to include (for train/val/test split)
            hand_feature_cols: List of hand-crafted feature column names
            demo_feature_cols: List of demographic feature column names
            encoding_info: Dict containing demographic preprocessing info (from normalize_features_per_fold)
        """

        self.data = features_df[features_df['patient_id'].isin(patient_ids)].copy()
        self.patient_ids = sorted(self.data['patient_id'].unique())

        #Identifica colonne CNN
        self.cnn_feature_cols = [c for c in self.data.columns if c.startswith('cnn_feature_')]
        self.cnn_feature_dim = len(self.cnn_feature_cols)


        # AGGIUNGI: Identifica hand-crafted features
        if hand_feature_cols is None:
            hand_feature_cols = []
        self.hand_feature_cols = [c for c in hand_feature_cols if c in self.data.columns]
        
        # AGGIUNGI: Identifica demographic features (with encoding info for preprocessing)
        if demo_feature_cols is None:
            demo_feature_cols = []
        self.demo_feature_cols = demo_feature_cols  # Keep original names for reference
        self.encoding_info = encoding_info if encoding_info is not None else {}
        
        # Calculate actual demographic feature dimension (accounting for one-hot encoding)
        # Check for PREPROCESSED columns (not original names)
        self.demo_feature_dim = 0
        if 'Age' in demo_feature_cols and 'Age_normalized' in self.data.columns:
            self.demo_feature_dim += 1  # Age_normalized
        if 'Sex' in demo_feature_cols and 'Sex_encoded' in self.data.columns:
            self.demo_feature_dim += 1  # Sex_encoded
        if 'SmokingStatus' in demo_feature_cols:
            smoking_cols = self.encoding_info.get('smoking_columns', [])
            # Verify smoking columns actually exist in data
            existing_smoking_cols = [c for c in smoking_cols if c in self.data.columns]
            self.demo_feature_dim += len(existing_smoking_cols)  # One-hot encoded (3 features)
        
        # Progression labels
        self.labels = self.data.groupby('patient_id')['gt_has_progressed'].first()
        
        print(f"PatientSliceDataset initialized:")
        print(f"  Patients: {len(self.patient_ids)}")
        print(f"  CNN feature dimension: {self.cnn_feature_dim}")
        print(f"  Hand-crafted features: {len(self.hand_feature_cols)}")
        print(f"  Demographic features: {self.demo_feature_dim} (preprocessed)")
        
        # DEBUG: Show demographic breakdown
        if demo_feature_cols:
            print(f"  Demographic breakdown:")
            if 'Age' in demo_feature_cols:
                has_age = 'Age_normalized' in self.data.columns
                print(f"    Age: {'✓' if has_age else '✗'} (Age_normalized)")
            if 'Sex' in demo_feature_cols:
                has_sex = 'Sex_encoded' in self.data.columns
                print(f"    Sex: {'✓' if has_sex else '✗'} (Sex_encoded)")
            if 'SmokingStatus' in demo_feature_cols:
                smoking_cols = self.encoding_info.get('smoking_columns', [])
                existing_cols = [c for c in smoking_cols if c in self.data.columns]
                print(f"    SmokingStatus: {len(existing_cols)}/{len(smoking_cols)} columns found")
                if existing_cols:
                    print(f"      Columns: {existing_cols}")
        
        print(f"  Total patient-level features: {len(self.hand_feature_cols) + self.demo_feature_dim}")
        print(f"  Progression cases: {self.labels.sum()}")
        
        
        
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.patient_ids[idx]
        patient_slices = self.data[self.data['patient_id'] == patient_id]
        
        # CNN features per slice
        cnn_features = torch.tensor(
            patient_slices[self.cnn_feature_cols].values,
            dtype=torch.float32
        )
        
        # Hand-crafted features (separate)
        if self.hand_feature_cols:
            hand_features = torch.tensor(
                patient_slices[self.hand_feature_cols].iloc[0].values,
                dtype=torch.float32
            )
        else:
            hand_features = None
        
        # Demographic features (separate, using preprocessed columns)
        if self.demo_feature_cols:
            # Extract preprocessed demographic features
            demo_vals = []
            row = patient_slices.iloc[0]
            
            # Age (normalized)
            if 'Age' in self.demo_feature_cols and 'Age_normalized' in row:
                demo_vals.append(row['Age_normalized'])
            
            # Sex (encoded and centered)
            if 'Sex' in self.demo_feature_cols and 'Sex_encoded' in row:
                demo_vals.append(row['Sex_encoded'])
            
            # Smoking (one-hot, centered)
            if 'SmokingStatus' in self.demo_feature_cols:
                smoking_cols = self.encoding_info.get('smoking_columns', [])
                for col in smoking_cols:
                    if col in row:
                        demo_vals.append(row[col])
            
            demo_features = torch.tensor(demo_vals, dtype=torch.float32) if demo_vals else None
        else:
            demo_features = None
        
        # Label
        label = torch.tensor(self.labels[patient_id], dtype=torch.float32)
        
        return {
            'patient_id': patient_id,
            'cnn_features': cnn_features,
            'hand_features': hand_features,
            'demo_features': demo_features,
            'label': label,
            'num_slices': len(patient_slices)
        }


def collate_patient_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for patient batches with multi-modal features
    """
    patient_ids = [item['patient_id'] for item in batch]
    labels = torch.tensor([item['label'].item() for item in batch], dtype=torch.float32)
    lengths = torch.tensor([item['num_slices'] for item in batch], dtype=torch.long)
    
    # Find max number of slices in this batch
    max_slices = max(item['num_slices'] for item in batch)
    cnn_feature_dim = batch[0]['cnn_features'].shape[1]
    
    # Create padded tensor for CNN features
    batch_size = len(batch)
    padded_cnn_features = torch.zeros(batch_size, max_slices, cnn_feature_dim, dtype=torch.float32)
    
    # Fill in CNN features
    for i, item in enumerate(batch):
        num_slices = item['num_slices']
        padded_cnn_features[i, :num_slices, :] = item['cnn_features']
    
    # Stack hand-crafted features
    if batch[0]['hand_features'] is not None:
        hand_features = torch.stack([item['hand_features'] for item in batch])
    else:
        hand_features = None
    
    # Stack demographic features
    if batch[0]['demo_features'] is not None:
        demo_features = torch.stack([item['demo_features'] for item in batch])
    else:
        demo_features = None
    
    return {
        'patient_ids': patient_ids,
        'cnn_features': padded_cnn_features,  # (batch_size, max_slices, cnn_dim)
        'hand_features': hand_features,  # (batch_size, hand_dim) or None
        'demo_features': demo_features,  # (batch_size, demo_dim) or None
        'labels': labels,  # (batch_size,)
        'lengths': lengths  # (batch_size,)
    }


def create_dataloaders(
    features_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    hand_feature_cols: List[str] = None,
    demo_feature_cols: List[str] = None,
    encoding_info: dict = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    """
    # Create datasets
    train_dataset = PatientSliceDataset(features_df, patient_ids=train_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols, encoding_info=encoding_info)
    val_dataset = PatientSliceDataset(features_df, patient_ids=val_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols, encoding_info=encoding_info)
    test_dataset = PatientSliceDataset(features_df, patient_ids=test_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols, encoding_info=encoding_info)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


"""
    Previously defined in utilities.py, now integrated here for completeness
    def compute_class_weights(features_df: pd.DataFrame, patient_ids: List[str]) -> Tuple[float, float]:
    
    Compute class weights for imbalanced data
    
    # Get labels for patients
    patient_labels = features_df[features_df['patient_id'].isin(patient_ids)].groupby('patient_id')['gt_has_progressed'].first()
    
    n_total = len(patient_labels)
    n_pos = patient_labels.sum()
    n_neg = n_total - n_pos
    
    # Compute weights (inverse frequency)
    weight_neg = n_total / (2 * n_neg) if n_neg > 0 else 1.0
    weight_pos = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    
    print(f"\nClass distribution:")
    print(f"  No progression: {n_neg} ({n_neg/n_total*100:.1f}%)")
    print(f"  Progression: {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"\nClass weights:")
    print(f"  No progression: {weight_neg:.4f}")
    print(f"  Progression: {weight_pos:.4f}")
    
    return (weight_neg, weight_pos)
"""


def compute_class_weights(features_df, patient_ids):
    """Compute class weights to handle imbalance"""
    patient_labels = features_df[features_df['patient_id'].isin(patient_ids)].groupby('patient_id')['gt_has_progressed'].first()
    
    # Get label counts
    labels = patient_labels
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    
    # Compute weights
    total = len(labels)
    weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
    weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0
    
    # More aggressive weighting for severe imbalance
    if pos_count < neg_count * 0.3:  # If < 30% positive
        weight_pos *= 1.5  # Boost positive class more
    
    return torch.tensor([weight_neg, weight_pos])