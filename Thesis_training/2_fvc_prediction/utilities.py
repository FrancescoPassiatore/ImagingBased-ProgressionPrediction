import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights
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
        extra   = patients_npy - patients_csv
        if missing:
            print(f"⚠️  {len(missing)} patients in CSV without NPY folder: {list(missing)[:5]}...")
        if extra:
            print(f"ℹ️  {len(extra)} NPY folders not in CSV (will be ignored)")
        available = patients_csv & patients_npy
        print(f"✅ {len(available)} patients with complete data (CSV + NPY)")

    def get_patient_data(self) -> Tuple[Dict, Dict]:
        df          = self.df_gt
        features_df = self.df_features

        patient_data = {}
        for patient_id in df['PatientID'].unique():
            patient_rows = df[df['PatientID'] == patient_id]
            gt_label     = int(patient_rows['has_progressed'].iloc[0])
            baseline_fvc = float(patient_rows['BaselineFVC'].iloc[0])
            gt_fvc52     = float(patient_rows['Week52FVC'].iloc[0])

            patient_npy_folder = os.path.join(self.npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue
            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue

            patient_data[patient_id] = {
                'gt_has_progressed': gt_label,
                'baselinefvc':       baseline_fvc,
                'gt_fvc52':          gt_fvc52,
                'slices':            npy_files,
                'n_slices':          len(npy_files),
            }

        features_data = {}
        for _, row in features_df.iterrows():
            patient_id = row['Patient']
            features_data[patient_id] = {
                'approx_vol':           float(row['ApproxVol_30_60']),
                'avg_num_tissue_pixel': float(row['Avg_NumTissuePixel_30_60']),
                'avg_tissue':           float(row['Avg_Tissue_30_60']),
                'avg_tissue_thickness': float(row['Avg_Tissue_thickness_30_60']),
                'avg_tissue_by_total':  float(row['Avg_TissueByTotal_30_60']),
                'avg_tissue_by_lung':   float(row['Avg_TissueByLung_30_60']),
                'mean':                 float(row['Mean_30_60']),
                'skew':                 float(row['Skew_30_60']),
                'kurtosis':             float(row['Kurtosis_30_60']),
                'age':           float(row['Age'])          if 'Age'           in row else 65.0,
                'sex':           int(row['Sex'])            if 'Sex'           in row else 0,
                'smoking_status':int(row['SmokingStatus'])  if 'SmokingStatus' in row else 0,
            }

        return patient_data, features_data


# =============================================================================
# CT SLICE LOADER
# =============================================================================

class CTSliceLoaderPatients(Dataset):
    """Load CT slices from NPY files"""

    def __init__(self, patient_data, transform=None):
        self.transform = transform
        self.slice_data = []
        self.patients   = set()
        self.patient_to_indices = defaultdict(list)

        idx = 0
        for patient_id, patient_info in patient_data.items():
            self.patients.add(patient_id)
            gt_label    = patient_info['gt_has_progressed']
            baselinefvc = patient_info['baselinefvc']
            gt_fvc52    = patient_info['gt_fvc52']

            for slice_path in patient_info['slices']:
                self.slice_data.append({
                    'patient_id':        patient_id,
                    'slice_path':        slice_path,
                    'gt_has_progressed': int(gt_label),
                    'baselinefvc':       float(baselinefvc),
                    'gt_fvc52':          float(gt_fvc52),
                })
                self.patient_to_indices[patient_id].append(idx)
                idx += 1

        print(f"Dataset created: {len(self.slice_data)} slices from {len(self.patients)} patients")

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        item      = self.slice_data[idx]
        slice_img = np.load(item['slice_path'])

        if len(slice_img.shape) == 2:
            slice_img = np.stack([slice_img] * 3, axis=-1)

        slice_tensor = torch.from_numpy(slice_img).permute(2, 0, 1).float()

        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        return {
            'image':             slice_tensor,
            'patient_id':        item['patient_id'],
            'slice_path':        item['slice_path'],
            'gt_has_progressed': item['gt_has_progressed'],
            'baselinefvc':       item['baselinefvc'],
            'gt_fvc52':          item['gt_fvc52'],
        }


def patient_group_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {
            'images': torch.empty(0, 3, 224, 224),
            'patient_ids': [],
            'lengths': torch.empty(0, dtype=torch.long),
            'slice_paths': [],
            'gt_labels': torch.empty(0),
            'baselinefvc': torch.empty(0),
            'gt_fvc52': torch.empty(0),
        }

    images      = torch.stack([b['image'] for b in batch])
    slice_paths = [b['slice_path'] for b in batch]
    gt_labels   = torch.tensor([b['gt_has_progressed'] for b in batch])
    baselinefvc = torch.tensor([b['baselinefvc'] for b in batch])
    gt_fvc52    = torch.tensor([b['gt_fvc52'] for b in batch])

    lengths, pid_order = [], []
    i = 0
    while i < len(batch):
        pid = batch[i]['patient_id']
        j   = i
        while j < len(batch) and batch[j]['patient_id'] == pid:
            j += 1
        lengths.append(j - i)
        pid_order.append(pid)
        i = j

    return {
        'images':      images,
        'patient_ids': pid_order,
        'lengths':     torch.tensor(lengths, dtype=torch.long),
        'slice_paths': slice_paths,
        'gt_labels':   gt_labels,
        'baselinefvc': baselinefvc,
        'gt_fvc52':    gt_fvc52,
    }


# =============================================================================
# CNN FEATURE EXTRACTOR
# =============================================================================

class CNNFeatureExtractor:
    """Create CNN feature extractor model"""

    def __init__(self, model_name='resnet50', device=None):
        self.model_name = model_name
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model      = self._load_model()
        print(f"Using {model_name} on {self.device}")

    def _load_model(self):
        if self.model_name == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'densenet121':
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        model = model.to(self.device)
        model.eval()
        return model

    def extract_features_patient_grouping(self, patient_data, patients_per_batch=4, save_path=None):
        print("\n" + "="*60)
        print("EXTRACTING CNN FEATURES (Patient-Grouped)")
        print("="*60)

        transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = CTSliceLoaderPatients(patient_data, transform=transform)

        from torch.utils.data import Sampler
        import random

        class PatientBatchSampler(Sampler):
            def __init__(self, dataset, patients_per_batch=4, shuffle=False):
                self.ds      = dataset
                self.shuffle = shuffle
                self.ppb     = patients_per_batch

            def __iter__(self):
                patients = list(self.ds.patients)
                if self.shuffle:
                    random.shuffle(patients)
                for i in range(0, len(patients), self.ppb):
                    batch_pids = patients[i:i + self.ppb]
                    idxs = []
                    for pid in batch_pids:
                        idxs.extend(list(self.ds.patient_to_indices[pid]))
                    yield idxs

            def __len__(self):
                return (len(self.ds.patients) + self.ppb - 1) // self.ppb

        sampler = PatientBatchSampler(dataset, patients_per_batch=patients_per_batch, shuffle=False)
        dataloader = DataLoader(dataset, batch_sampler=sampler,
                                collate_fn=patient_group_collate, num_workers=4)

        all_features, all_metadata = [], []
        print(f"\nProcessing {len(dataset.patients)} patients in {len(dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images      = batch['images'].to(self.device)
                patient_ids = batch['patient_ids']
                lengths     = batch['lengths']
                slice_paths = batch['slice_paths']
                gt_labels   = batch['gt_labels']
                baselinefvc = batch['baselinefvc']
                gt_fvc52    = batch['gt_fvc52']

                features = self.model(images)
                features = features.view(features.size(0), -1).cpu().numpy()

                slice_idx = 0
                for i, (patient_id, length) in enumerate(zip(patient_ids, lengths)):
                    patient_slices = features[slice_idx:slice_idx + length]
                    for j in range(length):
                        all_features.append(patient_slices[j])
                        all_metadata.append({
                            'patient_id':        patient_id,
                            'slice_path':        slice_paths[slice_idx + j],
                            'slice_index':       j,
                            'total_slices':      int(length),
                            'gt_has_progressed': int(gt_labels[slice_idx + j]),
                            'baselinefvc':       float(baselinefvc[slice_idx + j]),
                            'gt_fvc52':          float(gt_fvc52[slice_idx + j]),
                        })
                    slice_idx += length

        print("\nCreating DataFrame...")
        features_array = np.array(all_features)
        feature_cols   = [f'cnn_feature_{i}' for i in range(features_array.shape[1])]
        features_df    = pd.DataFrame(features_array, columns=feature_cols)
        metadata_df    = pd.DataFrame(all_metadata)
        slice_features_df = pd.concat([metadata_df, features_df], axis=1)

        if save_path:
            slice_features_df.to_csv(save_path, index=False)
            print(f"Saved features to: {save_path}")

        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total slices:           {len(slice_features_df)}")
        print(f"Total patients:         {slice_features_df['patient_id'].nunique()}")
        print(f"Feature dimension:      {len(feature_cols)}")
        print(f"Avg slices per patient: {slice_features_df.groupby('patient_id').size().mean():.1f}")
        print(f"Patients with progression: {slice_features_df.groupby('patient_id')['gt_has_progressed'].first().sum()}")
        print(f"FVC baseline range: [{slice_features_df['baselinefvc'].min():.1f}, {slice_features_df['baselinefvc'].max():.1f}] mL")
        print(f"FVC 52weeks range:  [{slice_features_df['gt_fvc52'].min():.1f}, {slice_features_df['gt_fvc52'].max():.1f}] mL")

        first_patient = slice_features_df['patient_id'].iloc[0]
        print("\nExample – first patient's slices:")
        print(slice_features_df[slice_features_df['patient_id'] == first_patient][
            ['patient_id', 'slice_index', 'total_slices', 'gt_has_progressed', 'baselinefvc', 'gt_fvc52']
        ].head())

        return slice_features_df


# =============================================================================
# FVC DATASET
# =============================================================================

class FVCPatientSliceDataset(Dataset):
    """
    Dataset for FVC prediction.

    Changes vs original
    -------------------
    1. include_fvc_baseline param: when False (Block 4), FVC(0) is zeroed so
       the model receives no clinical anchor.
    2. demo_feature_dim computed only when demo_feature_cols is non-empty,
       preventing spurious demo features being counted in hand_only / cnn_* configs.
    3. demo_feature_cols are expected to be PREPROCESSED names
       (Age_normalized, Sex_encoded, Smoking_*), not originals.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        patient_ids: list,
        hand_feature_cols: List[str] = None,
        demo_feature_cols: List[str] = None,
        encoding_info: dict = None,
        include_fvc_baseline: bool = True,     # Block 4 flag
    ):
        self.data                 = features_df[features_df['patient_id'].isin(patient_ids)].copy()
        self.patient_ids          = sorted(self.data['patient_id'].unique())
        self.encoding_info        = encoding_info or {}
        self.include_fvc_baseline = include_fvc_baseline

        # CNN feature columns
        self.cnn_feature_cols = [c for c in self.data.columns if c.startswith('cnn_feature_')]
        self.cnn_feature_dim  = len(self.cnn_feature_cols)

        # Hand-crafted feature columns (keep only those present in df)
        if hand_feature_cols is None:
            hand_feature_cols = []
        self.hand_feature_cols = [c for c in hand_feature_cols if c in self.data.columns]

        # Demographic columns: passed as PREPROCESSED names (Age_normalized, Sex_encoded, Smoking_*)
        # Only count dims when demographics were actually requested for this experiment.
        if demo_feature_cols is None:
            demo_feature_cols = []
        self.demo_feature_cols = demo_feature_cols

        smoking_cols = self.encoding_info.get('smoking_columns', [])
        self.demo_feature_dim = 0
        if self.demo_feature_cols:   # only count if this experiment uses demographics
            if 'Age_normalized' in self.data.columns and \
               any('age' in c.lower() or 'normalized' in c.lower() for c in self.demo_feature_cols):
                self.demo_feature_dim += 1
            if 'Sex_encoded' in self.data.columns and \
               any('sex' in c.lower() or 'encoded' in c.lower() for c in self.demo_feature_cols):
                self.demo_feature_dim += 1
            self.demo_feature_dim += sum(1 for c in smoking_cols if c in self.data.columns)

        # FVC caches
        self.fvc_baseline_map = self.data.groupby('patient_id')['baselinefvc'].first()
        self.fvc_52weeks_map  = self.data.groupby('patient_id')['gt_fvc52'].first()

        # Print summary
        print(f"FVCPatientSliceDataset initialized:")
        print(f"  Patients: {len(self.patient_ids)}")
        print(f"  CNN feature dimension: {self.cnn_feature_dim}")
        print(f"  Hand-crafted features: {len(self.hand_feature_cols)}")
        print(f"  Demographic features: {self.demo_feature_dim} (preprocessed)")
        if self.demo_feature_dim > 0:
            demo_breakdown = []
            if 'Age_normalized' in self.data.columns and \
               any('age' in c.lower() or 'normalized' in c.lower() for c in self.demo_feature_cols):
                demo_breakdown.append("Age=1")
            if 'Sex_encoded' in self.data.columns and \
               any('sex' in c.lower() or 'encoded' in c.lower() for c in self.demo_feature_cols):
                demo_breakdown.append("Sex=1")
            if smoking_cols:
                demo_breakdown.append(f"Smoking={len(smoking_cols)} (one-hot)")
            print(f"    ({', '.join(demo_breakdown)})")
        print(f"  Total patient-level features: {len(self.hand_feature_cols) + self.demo_feature_dim}")
        print(f"  FVC(0) as input: {'YES' if include_fvc_baseline else 'NO – Block 4 ablation'}")
        print(f"  FVC baseline range: [{self.fvc_baseline_map.min():.1f}, {self.fvc_baseline_map.max():.1f}] mL")
        print(f"  FVC 52weeks range:  [{self.fvc_52weeks_map.min():.1f},  {self.fvc_52weeks_map.max():.1f}] mL")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict:
        patient_id     = self.patient_ids[idx]
        patient_slices = self.data[self.data['patient_id'] == patient_id]
        row            = patient_slices.iloc[0]

        # CNN features: (n_slices, cnn_dim)
        cnn_features = torch.tensor(
            patient_slices[self.cnn_feature_cols].values,
            dtype=torch.float32
        )

        # Hand-crafted features: (n_hand,) or empty tensor
        if self.hand_feature_cols:
            hand_features = torch.tensor(
                patient_slices[self.hand_feature_cols].iloc[0].values,
                dtype=torch.float32
            )
        else:
            hand_features = torch.tensor([], dtype=torch.float32)

        # Demographic features: (n_demo,) or empty tensor
        # Built from preprocessed column names stored in self.demo_feature_cols
        if self.demo_feature_cols:
            demo_vals = []
            if 'Age_normalized' in row.index and \
               any('age' in c.lower() or 'normalized' in c.lower() for c in self.demo_feature_cols):
                demo_vals.append(float(row['Age_normalized']))
            if 'Sex_encoded' in row.index and \
               any('sex' in c.lower() or 'encoded' in c.lower() for c in self.demo_feature_cols):
                demo_vals.append(float(row['Sex_encoded']))
            smoking_cols = self.encoding_info.get('smoking_columns', [])
            for col in smoking_cols:
                if col in row.index:
                    demo_vals.append(float(row[col]))
            demo_features = torch.tensor(demo_vals, dtype=torch.float32)
        else:
            demo_features = torch.tensor([], dtype=torch.float32)

        # FVC values
        # Block 4 ablation: zero out FVC(0) so model gets no clinical anchor
        raw_fvc      = float(self.fvc_baseline_map.loc[patient_id])
        fvc_baseline = torch.tensor(
            raw_fvc if self.include_fvc_baseline else 0.0,
            dtype=torch.float32
        )
        fvc_52weeks  = torch.tensor(
            float(self.fvc_52weeks_map.loc[patient_id]),
            dtype=torch.float32
        )

        return {
            'patient_id':    patient_id,
            'cnn_features':  cnn_features,   # (n_slices, cnn_dim)
            'hand_features': hand_features,  # (n_hand,) or empty
            'demo_features': demo_features,  # (n_demo,) or empty
            'fvc_baseline':  fvc_baseline,   # scalar
            'fvc_52weeks':   fvc_52weeks,    # scalar
            'num_slices':    len(patient_slices),
        }


# =============================================================================
# COLLATE
# =============================================================================

def collate_fvc_batch(batch: List[Dict]) -> Dict:
    """Collate patient dicts into padded batch tensors."""
    patient_ids  = [item['patient_id'] for item in batch]
    fvc_baseline = torch.stack([item['fvc_baseline'] for item in batch]).unsqueeze(-1)  # (B,1)
    fvc_52weeks  = torch.tensor([item['fvc_52weeks'].item() for item in batch], dtype=torch.float32)
    lengths      = torch.tensor([item['num_slices'] for item in batch], dtype=torch.long)

    # Padded CNN features
    max_slices      = max(item['num_slices'] for item in batch)
    cnn_feature_dim = batch[0]['cnn_features'].shape[1]
    padded_cnn      = torch.zeros(len(batch), max_slices, cnn_feature_dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        n = item['num_slices']
        padded_cnn[i, :n, :] = item['cnn_features']

    # Hand-crafted: stack if non-empty, else None
    hand_features = (
        torch.stack([item['hand_features'] for item in batch])
        if batch[0]['hand_features'].numel() > 0 else None
    )

    # Demographic: stack if non-empty, else None
    demo_features = (
        torch.stack([item['demo_features'] for item in batch])
        if batch[0]['demo_features'].numel() > 0 else None
    )

    return {
        'patient_ids':   patient_ids,
        'cnn_features':  padded_cnn,      # (B, max_slices, cnn_dim)
        'hand_features': hand_features,   # (B, n_hand) or None
        'demo_features': demo_features,   # (B, n_demo) or None
        'fvc_baseline':  fvc_baseline,    # (B, 1)
        'fvc_52weeks':   fvc_52weeks,     # (B,)
        'lengths':       lengths,         # (B,)
    }


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_fvc_dataloaders(
    features_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    hand_feature_cols: List[str] = None,
    demo_feature_cols: List[str] = None,
    encoding_info: dict = None,
    include_fvc_baseline: bool = True,      # Block 4 flag
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders for FVC prediction.

    Parameters
    ----------
    include_fvc_baseline : bool
        When False (Block 4 ablation), FVC(0) is zeroed in the dataset so the
        model receives no clinical anchor — quantifies imaging-only contribution.
    """
    dataset_kwargs = dict(
        hand_feature_cols=hand_feature_cols,
        demo_feature_cols=demo_feature_cols,
        encoding_info=encoding_info,
        include_fvc_baseline=include_fvc_baseline,
    )
    train_dataset = FVCPatientSliceDataset(features_df, patient_ids=train_ids, **dataset_kwargs)
    val_dataset   = FVCPatientSliceDataset(features_df, patient_ids=val_ids,   **dataset_kwargs)
    test_dataset  = FVCPatientSliceDataset(features_df, patient_ids=test_ids,  **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fvc_batch, num_workers=num_workers, pin_memory=True,
        drop_last=True,   # avoids BatchNorm crash on single-sample batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fvc_batch, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fvc_batch, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# CLASS WEIGHTS  (classification pipeline)
# =============================================================================

def compute_class_weights(features_df: pd.DataFrame, patient_ids: List[str]) -> torch.Tensor:
    """Compute class weights to handle label imbalance."""
    patient_labels = (
        features_df[features_df['patient_id'].isin(patient_ids)]
        .groupby('patient_id')['gt_has_progressed']
        .first()
    )
    pos_count = int(patient_labels.sum())
    neg_count = len(patient_labels) - pos_count
    total     = len(patient_labels)

    weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
    weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0

    if pos_count < neg_count * 0.3:
        weight_pos *= 1.5

    return torch.tensor([weight_neg, weight_pos])