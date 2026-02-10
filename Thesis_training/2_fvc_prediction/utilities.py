import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from typing import Dict, List, Tuple
import glob
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class IPFDataLoader:
    """Load patient data from CSV and NPY files"""
    
    def __init__(self, csv_path: str, features_path: str, npy_dir: str):
        self.df_gt = pd.read_csv(csv_path)
        self.df_features = pd.read_csv(features_path)
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
            patient_data: Dict with {patient_id: {'gt_has_progressed', 'baselinefvc', 'gt_fvc52', 'slices', 'n_slices'}}
            features_data: Dict with {patient_id: {feature: value}}
        """
        # Load main CSV
        df = self.df_gt
        
        # Load features
        features_df = self.df_features
        
        # Build patient_data dictionary
        patient_data = {}
        for patient_id in df['PatientID'].unique():
            patient_rows = df[df['PatientID'] == patient_id]
            
            gt_label = int(patient_rows['has_progressed'].iloc[0])
            baseline_fvc = float(patient_rows['BaselineFVC'].iloc[0])
            gt_fvc52 = float(patient_rows['Week52FVC'].iloc[0])

            # Get CT scans
            # Check if patient has NPY files
            patient_npy_folder = os.path.join(self.npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue

            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue
            
            patient_data[patient_id] = {
                'gt_has_progressed': gt_label,
                'baselinefvc': baseline_fvc,
                'gt_fvc52': gt_fvc52,
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
    """Load CT slices from NPY files"""

    def __init__(self, patient_data, transform=None):
        
        self.transform = transform
        # Build index: list of (patient_id, slice_path, gt_label, baselinefvc, gt_fvc52)
        self.slice_data = []
        self.patients = set()
        self.patient_to_indices = defaultdict(list)

        idx = 0
        for patient_id, patient_info in patient_data.items():
            self.patients.add(patient_id)
            gt_label = patient_info['gt_has_progressed']
            baselinefvc = patient_info['baselinefvc']
            gt_fvc52 = patient_info['gt_fvc52']
            
            for slice_path in patient_info['slices']:
                self.slice_data.append({
                    'patient_id': patient_id,
                    'slice_path': slice_path,
                    'gt_has_progressed': int(gt_label),
                    'baselinefvc': float(baselinefvc),
                    'gt_fvc52': float(gt_fvc52)
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

        return {
            'image': slice_tensor,
            'patient_id': item['patient_id'],
            'slice_path': item['slice_path'],
            'gt_has_progressed': item['gt_has_progressed'],
            'baselinefvc': item['baselinefvc'],
            'gt_fvc52': item['gt_fvc52']
        }


def patient_group_collate(batch):
    """
    Collate function that groups slices by patient
    """
    # Filter None elements
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return {
            'images': torch.empty(0, 3, 224, 224),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long),
            'slice_paths': [],
            'gt_labels': torch.empty(0),
            'baselinefvc': torch.empty(0),
            'gt_fvc52': torch.empty(0)
        }

    # Extract tensors
    images = torch.stack([b['image'] for b in batch])
    slice_paths = [b['slice_path'] for b in batch]
    gt_labels = torch.tensor([b['gt_has_progressed'] for b in batch])
    baselinefvc = torch.tensor([b['baselinefvc'] for b in batch])
    gt_fvc52 = torch.tensor([b['gt_fvc52'] for b in batch])

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
        'gt_labels': gt_labels,
        'baselinefvc': baselinefvc,
        'gt_fvc52': gt_fvc52
    }


class CNNFeatureExtractor:
    """Create CNN feature extractor model"""
    
    def __init__(self, model_name='resnet50', device=None):
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
        features_df : pd.DataFrame
            DataFrame with columns [patient_id, slice_index, baselinefvc, gt_fvc52, gt_has_progressed, cnn_feature_0, ...]
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
        from torch.utils.data import Sampler
        import random
        
        class PatientBatchSampler(Sampler):
            def __init__(self, dataset, patients_per_batch=4, shuffle=False):
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
                baselinefvc = batch['baselinefvc']
                gt_fvc52 = batch['gt_fvc52']
                
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
                            'gt_has_progressed': int(gt_labels[slice_idx + j]),
                            'baselinefvc': float(baselinefvc[slice_idx + j]),
                            'gt_fvc52': float(gt_fvc52[slice_idx + j])
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
        print(f"FVC baseline range: [{slice_features_df['baselinefvc'].min():.1f}, {slice_features_df['baselinefvc'].max():.1f}] mL")
        print(f"FVC 52weeks range: [{slice_features_df['gt_fvc52'].min():.1f}, {slice_features_df['gt_fvc52'].max():.1f}] mL")
        
        # Show example of patient grouping
        print("\nExample - First patient's slices:")
        first_patient = slice_features_df['patient_id'].iloc[0]
        patient_slices = slice_features_df[slice_features_df['patient_id'] == first_patient]
        print(patient_slices[['patient_id', 'slice_index', 'total_slices', 'gt_has_progressed', 'baselinefvc', 'gt_fvc52']].head())
        
        return slice_features_df


class FVCPatientSliceDataset(Dataset):
    """
    Dataset for FVC prediction that includes:
    - CNN features from CT slices
    - Baseline FVC (week 0)
    - Target FVC (week 52)
    - Optional: Hand-crafted features
    - Optional: Demographic features (with improved preprocessing)
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
            features_df: DataFrame with columns [patient_id, slice_index, baselinefvc, gt_fvc52, cnn_feature_0, ...]
            patient_ids: List of patient IDs to include
            hand_feature_cols: List of hand-crafted feature column names
            demo_feature_cols: List of demographic feature column names (raw column names like 'Age', 'Sex', 'SmokingStatus')
            encoding_info: Dict containing demographic preprocessing info (from normalize_features_per_fold)
        """
        self.data = features_df[features_df['patient_id'].isin(patient_ids)].copy()
        self.patient_ids = sorted(self.data['patient_id'].unique())
        self.encoding_info = encoding_info or {}
        
        # Identify CNN feature columns
        self.cnn_feature_cols = [c for c in self.data.columns if c.startswith('cnn_feature_')]
        self.cnn_feature_dim = len(self.cnn_feature_cols)
        
        # Identify hand-crafted features
        if hand_feature_cols is None:
            hand_feature_cols = []
        self.hand_feature_cols = [c for c in hand_feature_cols if c in self.data.columns]
        
        # For demographics, we'll extract preprocessed features dynamically
        # Store original demo column names
        if demo_feature_cols is None:
            demo_feature_cols = []
        self.demo_feature_cols = demo_feature_cols
        
        # Count actual demographic features (after preprocessing)
        self.demo_feature_dim = 0
        if 'Age_normalized' in self.data.columns:
            self.demo_feature_dim += 1
        if 'Sex_encoded' in self.data.columns:
            self.demo_feature_dim += 1
        # Add one-hot encoded smoking columns
        smoking_cols = self.encoding_info.get('smoking_columns', [])
        self.demo_feature_dim += len(smoking_cols)
        
        # Get FVC values per patient (take first value since all slices have same patient-level FVC)
        self.fvc_baseline = self.data.groupby('patient_id')['baselinefvc'].first()
        self.fvc_52weeks = self.data.groupby('patient_id')['gt_fvc52'].first()
        
        print(f"FVCPatientSliceDataset initialized:")
        print(f"  Patients: {len(self.patient_ids)}")
        print(f"  CNN feature dimension: {self.cnn_feature_dim}")
        print(f"  Hand-crafted features: {len(self.hand_feature_cols)}")
        print(f"  Demographic features: {self.demo_feature_dim} (preprocessed)")
        if self.demo_feature_dim > 0:
            demo_breakdown = []
            if 'Age_normalized' in self.data.columns:
                demo_breakdown.append("Age=1")
            if 'Sex_encoded' in self.data.columns:
                demo_breakdown.append("Sex=1")
            if smoking_cols:
                demo_breakdown.append(f"Smoking={len(smoking_cols)} (one-hot)")
            print(f"    ({', '.join(demo_breakdown)})")
        print(f"  Total patient-level features: {len(self.hand_feature_cols) + self.demo_feature_dim}")
        print(f"  FVC baseline range: [{self.fvc_baseline.min():.1f}, {self.fvc_baseline.max():.1f}] mL")
        print(f"  FVC 52weeks range: [{self.fvc_52weeks.min():.1f}, {self.fvc_52weeks.max():.1f}] mL")
    
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
            hand_features = torch.tensor([], dtype=torch.float32)
        
        # Demographic features (separate, using preprocessed columns)
        if self.demo_feature_cols:
            # Extract preprocessed demographic features
            demo_vals = []
            row = patient_slices.iloc[0]
            
            # Age (normalized)
            if 'Age_normalized' in row:
                demo_vals.append(row['Age_normalized'])
            
            # Sex (encoded and centered)
            if 'Sex_encoded' in row:
                demo_vals.append(row['Sex_encoded'])
            
            # Smoking (one-hot, centered)
            smoking_cols = self.encoding_info.get('smoking_columns', [])
            for col in smoking_cols:
                if col in row:
                    demo_vals.append(row[col])
            
            demo_features = torch.tensor(demo_vals, dtype=torch.float32)
        else:
            demo_features = torch.tensor([], dtype=torch.float32)
        
        # FVC values (baseline as input, week 52 as target)
        fvc_baseline = torch.tensor(float(self.fvc_baseline.loc[patient_id]), dtype=torch.float32)
        fvc_52weeks = torch.tensor(float(self.fvc_52weeks.loc[patient_id]), dtype=torch.float32)
        
        return {
            'patient_id': patient_id,
            'cnn_features': cnn_features,
            'hand_features': hand_features,
            'demo_features': demo_features,
            'fvc_baseline': fvc_baseline,
            'fvc_52weeks': fvc_52weeks,
            'num_slices': len(patient_slices)
        }


def collate_fvc_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for FVC prediction batches
    """
    patient_ids = [item['patient_id'] for item in batch]
    fvc_baseline = torch.stack([item['fvc_baseline'] for item in batch]).unsqueeze(-1)  # (batch_size, 1)
    fvc_52weeks = torch.tensor([item['fvc_52weeks'].item() for item in batch], dtype=torch.float32)  # (batch_size,)
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
    if batch[0]['hand_features'].numel() > 0:
        hand_features = torch.stack([item['hand_features'] for item in batch])
    else:
        hand_features = None
    
    # Stack demographic features
    if batch[0]['demo_features'].numel() > 0:
        demo_features = torch.stack([item['demo_features'] for item in batch])
    else:
        demo_features = None
    
    return {
        'patient_ids': patient_ids,
        'cnn_features': padded_cnn_features,  # (batch_size, max_slices, cnn_dim)
        'hand_features': hand_features,  # (batch_size, hand_dim) or None
        'demo_features': demo_features,  # (batch_size, demo_dim) or None
        'fvc_baseline': fvc_baseline,  # (batch_size, 1)
        'fvc_52weeks': fvc_52weeks,  # (batch_size,)
        'lengths': lengths  # (batch_size,)
    }


def create_fvc_dataloaders(
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
    Create train, val, test dataloaders for FVC prediction
    
    Args:
        encoding_info: Dict containing demographic preprocessing info (from normalize_features_per_fold)
    """
    # Create datasets
    train_dataset = FVCPatientSliceDataset(
        features_df, 
        patient_ids=train_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols,
        encoding_info=encoding_info
    )
    val_dataset = FVCPatientSliceDataset(
        features_df,
        patient_ids=val_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols,
        encoding_info=encoding_info
    )
    test_dataset = FVCPatientSliceDataset(
        features_df,
        patient_ids=test_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols,
        encoding_info=encoding_info
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fvc_batch,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues with batch_size=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fvc_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fvc_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader