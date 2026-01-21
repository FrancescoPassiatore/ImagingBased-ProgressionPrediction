import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader
)
def create_cyclic_5fold_splits(patient_ids, random_state=42, save_path: Path = None): 
    """
    Each patient:
      - test exactly once
      - val exactly once
      - train in remaining folds
    """
    rng = np.random.RandomState(random_state)
    patient_ids = np.array(patient_ids)
    rng.shuffle(patient_ids)

    bins = np.array_split(patient_ids, 5)
    splits = {}

    for fold in range(5):
        test_bin = fold
        val_bin = (fold + 1) % 5
        train_bins = [i for i in range(5) if i not in (test_bin, val_bin)]

        splits[fold] = {
            "test": bins[test_bin].tolist(),
            "val": bins[val_bin].tolist(),
            "train": np.concatenate([bins[i] for i in train_bins]).tolist()
        }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"✓ Saved splits to {save_path}")

    return splits

from collections import defaultdict

def verify_cyclic_5fold_splits(splits, patient_ids, n_folds=5):
    print("\n" + "="*80)
    print("VERIFYING CYCLIC 5-FOLD SPLITS")
    print("="*80)

    patient_ids = set(patient_ids)

    # Track how many times each patient appears in each role
    role_counts = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int),
    }

    # ---- Per-fold checks ----
    for fold in range(n_folds):
        fold_data = splits[fold]

        train = set(fold_data['train'])
        val = set(fold_data['val'])
        test = set(fold_data['test'])

        # Disjointness
        assert train.isdisjoint(val), f"Fold {fold}: train/val overlap"
        assert train.isdisjoint(test), f"Fold {fold}: train/test overlap"
        assert val.isdisjoint(test), f"Fold {fold}: val/test overlap"

        # Coverage
        union = train | val | test
        assert union == patient_ids, f"Fold {fold}: missing or extra patients"

        # Count roles
        for pid in train:
            role_counts['train'][pid] += 1
        for pid in val:
            role_counts['val'][pid] += 1
        for pid in test:
            role_counts['test'][pid] += 1

        print(
            f"Fold {fold}: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

    # ---- Per-patient checks ----
    errors = False
    for pid in patient_ids:
        t = role_counts['test'][pid]
        v = role_counts['val'][pid]
        tr = role_counts['train'][pid]

        if not (t == 1 and v == 1 and tr == n_folds - 2):
            print(
                f"❌ Patient {pid}: "
                f"test={t}, val={v}, train={tr}"
            )
            errors = True

    if errors:
        raise AssertionError("❌ Split verification FAILED")

    print("\n✅ Split verification PASSED")
    print("Each patient:")
    print("  • test exactly once")
    print("  • val exactly once")
    print(f"  • train exactly {n_folds - 2} times")



CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'
}


if __name__ == '__main__':
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    dl = IPFDataLoader(CONFIG['csv_path'], CONFIG['features_path'], CONFIG['npy_dir'])
    patient_data, features_data = dl.get_patient_data()
    patient_ids = list(patient_data.keys())
    print(f"✓ Loaded {len(patient_data)} patients")
    save_path = Path('Training_2/kfold_splits.pkl')
    splits = create_cyclic_5fold_splits(patient_ids, random_state=42, save_path=save_path)
    verify_cyclic_5fold_splits(splits, patient_ids)