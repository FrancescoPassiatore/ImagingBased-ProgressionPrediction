import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import os
import glob
from pathlib import Path

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
            patient_rows = df[df['PatientID'] == patient_id]
            
            gt_label = int(patient_rows['has_progressed'].iloc[0])
            baseline_fvc = float(patient_rows['BaselineFVC'].iloc[0])
            gt_fvc52 = float(patient_rows['Week52FVC'].iloc[0])

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
    

def create_dataloaders(
    features_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    hand_feature_cols: List[str] = None,
    demo_feature_cols: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    """
    # Create datasets
    train_dataset = FVCPatientSliceDataset(features_df, patient_ids=train_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols)
    val_dataset =FVCPatientSliceDataset(features_df, patient_ids=val_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols)
    test_dataset = FVCPatientSliceDataset(features_df, patient_ids=test_ids, hand_feature_cols=hand_feature_cols, demo_feature_cols=demo_feature_cols)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fvc_batch,
        num_workers=num_workers,
        pin_memory=True
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


class FVCPatientSliceDataset(Dataset):
    """
    Dataset for FVC prediction that includes:
    - CNN features from CT slices
    - Baseline FVC (week 0)
    - Target FVC (week 52)
    - Optional: Hand-crafted features
    - Optional: Demographic features
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        patient_ids: list,
        hand_feature_cols: List[str] = None,
        demo_feature_cols: List[str] = None
    ):
        """
        Args:
            features_df: DataFrame with columns [patient_id, slice_index, fvc_baseline, fvc_52weeks, cnn_feature_0, ...]
            patient_ids: List of patient IDs to include
            hand_feature_cols: List of hand-crafted feature column names
            demo_feature_cols: List of demographic feature column names
        """
        self.data = features_df[features_df['patient_id'].isin(patient_ids)].copy()
        self.patient_ids = sorted(self.data['patient_id'].unique())
        
        # Identify CNN feature columns
        self.cnn_feature_cols = [c for c in self.data.columns if c.startswith('cnn_feature_')]
        self.cnn_feature_dim = len(self.cnn_feature_cols)
        
        # Identify hand-crafted features
        if hand_feature_cols is None:
            hand_feature_cols = []
        self.hand_feature_cols = [c for c in hand_feature_cols if c in self.data.columns]
        
        # Identify demographic features
        if demo_feature_cols is None:
            demo_feature_cols = []
        self.demo_feature_cols = [c for c in demo_feature_cols if c in self.data.columns]
        
        # Get FVC values per patient (take first value since all slices have same patient-level FVC)
        self.fvc_baseline = self.data.groupby('patient_id')['baselinefvc'].first()
        self.fvc_52weeks = self.data.groupby('patient_id')['gt_fvc52'].first()
        
        print(f"FVCPatientSliceDataset initialized:")
        print(f"  Patients: {len(self.patient_ids)}")
        print(f"  CNN feature dimension: {self.cnn_feature_dim}")
        print(f"  Hand-crafted features: {len(self.hand_feature_cols)}")
        print(f"  Demographic features: {len(self.demo_feature_cols)}")
        print(f"  Total patient-level features: {len(self.hand_feature_cols) + len(self.demo_feature_cols)}")
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
        
        # Patient-level features (hand-crafted + demographics)
        patient_features = []
        if self.hand_feature_cols:
            hand_vals = patient_slices[self.hand_feature_cols].iloc[0].values
            patient_features.append(hand_vals)
        if self.demo_feature_cols:
            demo_vals = patient_slices[self.demo_feature_cols].iloc[0].values
            patient_features.append(demo_vals)
        
        if patient_features:
            patient_features = torch.tensor(
                np.concatenate(patient_features),
                dtype=torch.float32
            )
        else:
            patient_features = torch.tensor([], dtype=torch.float32)
        
        # FVC values (baseline as input, week 52 as target)
        fvc_baseline = torch.tensor(float(self.fvc_baseline.loc[patient_id]), dtype=torch.float32)
        fvc_52weeks = torch.tensor(float(self.fvc_52weeks.loc[patient_id]), dtype=torch.float32)
        
        return {
            'patient_id': patient_id,
            'cnn_features': cnn_features,
            'patient_features': patient_features,
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
    
    # Separate hand-crafted and demographic features
    hand_features = None
    demo_features = None
    
    if 'hand_features' in batch[0] and batch[0]['hand_features'].numel() > 0:
        hand_features = torch.stack([item['hand_features'] for item in batch])
    
    if 'demo_features' in batch[0] and batch[0]['demo_features'].numel() > 0:
        demo_features = torch.stack([item['demo_features'] for item in batch])
    
    return {
        'patient_ids': patient_ids,
        'cnn_features': padded_cnn_features,  # (batch_size, max_slices, cnn_dim)
        'hand_features':hand_features,
        'demo_features':demo_features,
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
    demo_feature_cols: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders for FVC prediction
    """
    # Create datasets
    train_dataset = FVCPatientSliceDataset(
        features_df, 
        patient_ids=train_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols
    )
    val_dataset = FVCPatientSliceDataset(
        features_df,
        patient_ids=val_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols
    )
    test_dataset = FVCPatientSliceDataset(
        features_df,
        patient_ids=test_ids,
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fvc_batch,
        num_workers=num_workers,
        pin_memory=True
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


