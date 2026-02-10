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




class CNNFeatureExtractor:
    #Create CNN feature extractor model
    def __init__(self, model_name='efficientnetb1', device=None):
        """
        Parameters:
        -----------
        model_name : str
            Pre-trained model: 'resnet50', 'densenet121', 'efficientnet_b1'
        device : str
            'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        print(f"Using {model_name} on {self.device}")

    def _load_model(self):
        """Load pre-trained CNN and remove classification head"""
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove the final FC layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        return model
    

    def extract_features_patient_grouping(self, patient_data, patients_per_batch=4, save_path=None):
        """
        Extract features from multiple slices
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary containing patient information and their slices
        patients_per_batch : int
            Number of patients per batch
            
        Returns:
        --------
        features : np.ndarray
            Array of shape (n_slices, feature_dim)
        slice_paths : list
            Corresponding slice paths
        """

        print("\n" + "="*60)
        print("EXTRACTING CNN FEATURES (Patient-Grouped)")
        print("="*60)

        # Transforms for pre-trained models
        transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
        
        # Create dataset and dataloader
        dataset = CTSliceLoaderPatients(patient_data, transform=transform)
        # Create patient-grouped sampler
        sampler = PatientBatchSampler(
            dataset, 
            patients_per_batch=patients_per_batch,
            shuffle=False  # No shuffle for feature extraction
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=patient_group_collate,
            num_workers=4
        )
        
        # Extract features
        all_features = []
        all_metadata = []
        
        print(f"\nProcessing {len(dataset.patients)} patients in {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch['images'].to(self.device)
                patient_ids = batch['patient_ids']
                lengths = batch['lengths']
                slice_paths = batch['slice_paths']
                gt_labels = batch['gt_labels']
                
                # Extract features
                features = self.model(images)
                features = features.view(features.size(0), -1)  # Flatten
                features = features.cpu().numpy()
                
                # Store features with metadata
                slice_idx = 0
                for i, (patient_id, length) in enumerate(zip(patient_ids, lengths)):
                    patient_slices = features[slice_idx:slice_idx+length]
                    
                    for j in range(length):
                        all_features.append(patient_slices[j])
                        all_metadata.append({
                            'patient_id': patient_id,
                            'slice_path': slice_paths[slice_idx + j],
                            'slice_index': j,
                            'total_slices': int(length),
                            'gt_has_progressed': int(gt_labels[slice_idx + j])
                        })
                    
                    slice_idx += length
        
        # Create DataFrame
        print("\nCreating DataFrame...")
        features_array = np.array(all_features)
        feature_cols = [f'cnn_feature_{i}' for i in range(features_array.shape[1])]
        features_df = pd.DataFrame(features_array, columns=feature_cols)
        
        metadata_df = pd.DataFrame(all_metadata)
        slice_features_df = pd.concat([metadata_df, features_df], axis=1)
        
        # Save if requested
        if save_path:
            slice_features_df.to_csv(save_path, index=False)
            print(f"Saved features to: {save_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total slices: {len(slice_features_df)}")
        print(f"Total patients: {slice_features_df['patient_id'].nunique()}")
        print(f"Feature dimension: {len(feature_cols)}")
        print(f"Avg slices per patient: {slice_features_df.groupby('patient_id').size().mean():.1f}")
        print(f"Patients with progression: {slice_features_df.groupby('patient_id')['gt_has_progressed'].first().sum()}")
        
        # Show example of patient grouping
        print("\nExample - First patient's slices:")
        first_patient = slice_features_df['patient_id'].iloc[0]
        patient_slices = slice_features_df[slice_features_df['patient_id'] == first_patient]
        print(patient_slices[['patient_id', 'slice_index', 'total_slices', 'gt_has_progressed']].head())
        
        return slice_features_df
    


class PatientSliceDataset(Dataset):
    """
    Dataset that groups slices by patient for progression prediction
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        patient_ids: List[str] = None,
        feature_prefix: str = 'cnn_feature_'
    ):
        """
        Args:
            features_df: DataFrame with columns [patient_id, slice_index, gt_has_progressed, cnn_feature_0, ...]
            patient_ids: Optional list of patient IDs to include (for train/val/test split)
            feature_prefix: Prefix for feature columns
        """
        self.feature_prefix = feature_prefix
        
        # Filter by patient IDs if provided
        if patient_ids is not None:
            self.df = features_df[features_df['patient_id'].isin(patient_ids)].copy()
        else:
            self.df = features_df.copy()
        
        # Get feature columns
        self.feature_cols = [col for col in self.df.columns if col.startswith(feature_prefix)]
        self.feature_dim = len(self.feature_cols)
        
        # Group by patient
        self.patients = sorted(self.df['patient_id'].unique())
        
        # Create patient index mapping
        self.patient_data = {}
        for patient_id in self.patients:
            patient_df = self.df[self.df['patient_id'] == patient_id].sort_values('slice_index')
            
            # Extract features and label
            features = patient_df[self.feature_cols].values.astype(np.float32)
            label = int(patient_df['gt_has_progressed'].iloc[0])
            
            self.patient_data[patient_id] = {
                'features': features,
                'label': label,
                'num_slices': len(features)
            }
        
        print(f"PatientSliceDataset initialized:")
        print(f"  Patients: {len(self.patients)}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Progression cases: {sum(data['label'] for data in self.patient_data.values())}")
        print(f"  No progression cases: {sum(1 - data['label'] for data in self.patient_data.values())}")
    
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.patients[idx]
        data = self.patient_data[patient_id]
        
        return {
            'patient_id': patient_id,
            'features': torch.from_numpy(data['features']),  # (num_slices, feature_dim)
            'label': data['label'],
            'num_slices': data['num_slices']
        }


def collate_patient_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for patient batches
    Pads slices to max_slices in batch
    """
    patient_ids = [item['patient_id'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    lengths = torch.tensor([item['num_slices'] for item in batch], dtype=torch.long)
    
    # Find max number of slices in this batch
    max_slices = max(item['num_slices'] for item in batch)
    feature_dim = batch[0]['features'].shape[1]
    
    # Create padded tensor
    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_slices, feature_dim, dtype=torch.float32)
    
    # Fill in features
    for i, item in enumerate(batch):
        num_slices = item['num_slices']
        padded_features[i, :num_slices, :] = item['features']
    
    return {
        'patient_ids': patient_ids,
        'features': padded_features,  # (batch_size, max_slices, feature_dim)
        'labels': labels,  # (batch_size,)
        'lengths': lengths  # (batch_size,)
    }


def create_dataloaders(
    features_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    """
    # Create datasets
    train_dataset = PatientSliceDataset(features_df, patient_ids=train_ids)
    val_dataset = PatientSliceDataset(features_df, patient_ids=val_ids)
    test_dataset = PatientSliceDataset(features_df, patient_ids=test_ids)
    
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