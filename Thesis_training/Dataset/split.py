import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from collections import defaultdict


def create_stratified_cyclic_5fold_splits(
    patient_ids, 
    labels, 
    random_state=42, 
    save_path: Path = None
): 
    """
    Create stratified cyclic 5-fold splits ensuring balanced class distribution.
    Each patient:
      - test exactly once
      - val exactly once
      - train in remaining 3 folds
    
    Parameters:
    -----------
    patient_ids : array-like
        Patient identifiers
    labels : array-like
        Binary labels (0 or 1) for each patient
    random_state : int
        Random seed for reproducibility
    save_path : Path, optional
        Path to save the splits pickle file
    
    Returns:
    --------
    splits : dict
        Dictionary with fold indices as keys and train/val/test splits as values
    """
    rng = np.random.RandomState(random_state)
    patient_ids = np.array(patient_ids)
    labels = np.array(labels)
    
    # Separate patients by class
    positive_idx = np.where(labels == 1)[0]
    negative_idx = np.where(labels == 0)[0]
    
    positive_patients = patient_ids[positive_idx]
    negative_patients = patient_ids[negative_idx]
    
    print(f"\nClass distribution:")
    print(f"  Positive (progression): {len(positive_patients)} patients")
    print(f"  Negative (no progression): {len(negative_patients)} patients")
    
    # Shuffle each class separately
    rng.shuffle(positive_patients)
    rng.shuffle(negative_patients)
    
    # Split each class into 5 bins
    positive_bins = np.array_split(positive_patients, 5)
    negative_bins = np.array_split(negative_patients, 5)
    
    splits = {}
    
    for fold in range(5):
        test_bin = fold
        val_bin = (fold + 1) % 5
        train_bins = [i for i in range(5) if i not in (test_bin, val_bin)]
        
        # Combine positive and negative patients for each set
        test_patients = np.concatenate([
            positive_bins[test_bin],
            negative_bins[test_bin]
        ])
        
        val_patients = np.concatenate([
            positive_bins[val_bin],
            negative_bins[val_bin]
        ])
        
        train_patients = np.concatenate([
            np.concatenate([positive_bins[i] for i in train_bins]),
            np.concatenate([negative_bins[i] for i in train_bins])
        ])
        
        splits[fold] = {
            "test": test_patients.tolist(),
            "val": val_patients.tolist(),
            "train": train_patients.tolist()
        }
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"\n✓ Saved stratified splits to {save_path}")
    
    return splits


def verify_stratified_cyclic_5fold_splits(splits, patient_ids, labels, n_folds=5):
    """
    Verify that splits are:
    1. Cyclic (each patient appears once in test, once in val, 3 times in train)
    2. Stratified (class distribution is balanced across folds)
    3. No data leakage (train/val/test are disjoint)
    """
    print("\n" + "="*80)
    print("VERIFYING STRATIFIED CYCLIC 5-FOLD SPLITS")
    print("="*80)
    
    patient_ids = np.array(patient_ids)
    labels = np.array(labels)
    patient_ids_set = set(patient_ids)
    
    # Create patient-to-label mapping
    patient_label_map = dict(zip(patient_ids, labels))
    
    # Track how many times each patient appears in each role
    role_counts = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int),
    }
    
    # Track class distribution per fold
    fold_stats = []
    
    print("\n" + "="*80)
    print("PER-FOLD VERIFICATION")
    print("="*80)
    
    # ---- Per-fold checks ----
    for fold in range(n_folds):
        fold_data = splits[fold]
        
        train = set(fold_data['train'])
        val = set(fold_data['val'])
        test = set(fold_data['test'])
        
        # 1. Disjointness check
        assert train.isdisjoint(val), f"❌ Fold {fold}: train/val overlap"
        assert train.isdisjoint(test), f"❌ Fold {fold}: train/test overlap"
        assert val.isdisjoint(test), f"❌ Fold {fold}: val/test overlap"
        
        # 2. Coverage check
        union = train | val | test
        assert union == patient_ids_set, f"❌ Fold {fold}: missing or extra patients"
        
        # 3. Count roles (for cyclic verification)
        for pid in train:
            role_counts['train'][pid] += 1
        for pid in val:
            role_counts['val'][pid] += 1
        for pid in test:
            role_counts['test'][pid] += 1
        
        # 4. Calculate class distribution
        train_labels = [patient_label_map[pid] for pid in train]
        val_labels = [patient_label_map[pid] for pid in val]
        test_labels = [patient_label_map[pid] for pid in test]
        
        train_pos = sum(train_labels)
        val_pos = sum(val_labels)
        test_pos = sum(test_labels)
        
        train_total = len(train_labels)
        val_total = len(val_labels)
        test_total = len(test_labels)
        
        fold_stats.append({
            'fold': fold,
            'train_total': train_total,
            'train_pos': train_pos,
            'train_pct': train_pos / train_total * 100,
            'val_total': val_total,
            'val_pos': val_pos,
            'val_pct': val_pos / val_total * 100,
            'test_total': test_total,
            'test_pos': test_pos,
            'test_pct': test_pos / test_total * 100,
        })
        
        print(f"\nFOLD {fold}")
        print(f"  TRAIN: {train_total:2d} patients | "
              f"Pos: {train_pos:2d} ({train_pos/train_total*100:5.1f}%) | "
              f"Neg: {train_total-train_pos:2d} ({(train_total-train_pos)/train_total*100:5.1f}%)")
        print(f"  VAL  : {val_total:2d} patients | "
              f"Pos: {val_pos:2d} ({val_pos/val_total*100:5.1f}%) | "
              f"Neg: {val_total-val_pos:2d} ({(val_total-val_pos)/val_total*100:5.1f}%)")
        print(f"  TEST : {test_total:2d} patients | "
              f"Pos: {test_pos:2d} ({test_pos/test_total*100:5.1f}%) | "
              f"Neg: {test_total-test_pos:2d} ({(test_total-test_pos)/test_total*100:5.1f}%)")
    
    # ---- Cyclic property verification ----
    print("\n" + "="*80)
    print("CYCLIC PROPERTY VERIFICATION")
    print("="*80)
    
    cyclic_errors = False
    for pid in patient_ids_set:
        t = role_counts['test'][pid]
        v = role_counts['val'][pid]
        tr = role_counts['train'][pid]
        
        if not (t == 1 and v == 1 and tr == n_folds - 2):
            print(f"❌ Patient {pid}: test={t}, val={v}, train={tr}")
            cyclic_errors = True
    
    if cyclic_errors:
        raise AssertionError("❌ Cyclic property verification FAILED")
    
    print("✅ Cyclic property verified!")
    print("Each patient:")
    print("  • Appears in TEST exactly once")
    print("  • Appears in VAL exactly once")
    print(f"  • Appears in TRAIN exactly {n_folds - 2} times")
    
    # ---- Stratification quality check ----
    print("\n" + "="*80)
    print("STRATIFICATION QUALITY CHECK")
    print("="*80)
    
    fold_stats_df = pd.DataFrame(fold_stats)
    
    overall_pos_pct = labels.sum() / len(labels) * 100
    
    print(f"\nOverall progression %: {overall_pos_pct:.1f}%")
    print(f"\nPer-fold progression %:")
    print(f"  TRAIN: {fold_stats_df['train_pct'].mean():.1f}% ± {fold_stats_df['train_pct'].std():.1f}% "
          f"(range: {fold_stats_df['train_pct'].min():.1f}% - {fold_stats_df['train_pct'].max():.1f}%)")
    print(f"  VAL  : {fold_stats_df['val_pct'].mean():.1f}% ± {fold_stats_df['val_pct'].std():.1f}% "
          f"(range: {fold_stats_df['val_pct'].min():.1f}% - {fold_stats_df['val_pct'].max():.1f}%)")
    print(f"  TEST : {fold_stats_df['test_pct'].mean():.1f}% ± {fold_stats_df['test_pct'].std():.1f}% "
          f"(range: {fold_stats_df['test_pct'].min():.1f}% - {fold_stats_df['test_pct'].max():.1f}%)")
    
    # Calculate max deviation from overall mean
    max_val_dev = max(abs(row['val_pct'] - overall_pos_pct) for _, row in fold_stats_df.iterrows())
    max_test_dev = max(abs(row['test_pct'] - overall_pos_pct) for _, row in fold_stats_df.iterrows())
    
    print(f"\nMaximum deviation from overall mean ({overall_pos_pct:.1f}%):")
    print(f"  VAL  : ±{max_val_dev:.1f}%")
    print(f"  TEST : ±{max_test_dev:.1f}%")
    
    if max_val_dev < 5.0 and max_test_dev < 5.0:
        print(f"\n✅ EXCELLENT stratification! (deviations < 5%)")
        quality = "EXCELLENT"
    elif max_val_dev < 10.0 and max_test_dev < 10.0:
        print(f"\n✅ GOOD stratification! (deviations < 10%)")
        quality = "GOOD"
    elif max_val_dev < 15.0 and max_test_dev < 15.0:
        print(f"\n⚠️  ACCEPTABLE stratification (deviations < 15%)")
        quality = "ACCEPTABLE"
    else:
        print(f"\n⚠️  POOR stratification (deviations > 15%)")
        quality = "POOR"
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print("✅ No data leakage (train/val/test are disjoint)")
    print("✅ Complete coverage (all patients included)")
    print("✅ Cyclic property satisfied")
    print(f"✅ Stratification quality: {quality}")
    print("="*80)
    
    return fold_stats_df


CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv',
    'save_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl'
}


if __name__ == '__main__':
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(CONFIG['csv_path'])
    patient_ids = df['PatientID'].values
    labels = df['has_progressed'].values
    
    print(f"\nLoaded ground truth: {len(patient_ids)} patients")
    print(f"  Progression:    {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"  No progression: {len(labels) - labels.sum()} ({(len(labels) - labels.sum())/len(labels)*100:.1f}%)")
    
    # Create stratified cyclic splits
    print("\n" + "="*80)
    print("CREATING STRATIFIED CYCLIC 5-FOLD SPLITS")
    print("="*80)
    
    save_path = Path(CONFIG['save_path'])
    splits = create_stratified_cyclic_5fold_splits(
        patient_ids=patient_ids,
        labels=labels,
        random_state=42,
        save_path=save_path
    )
    
    # Verify splits
    fold_stats = verify_stratified_cyclic_5fold_splits(
        splits=splits,
        patient_ids=patient_ids,
        labels=labels
    )
    
    # Save statistics
    stats_path = save_path.parent / "fold_distribution_stats_stratified.csv"
    fold_stats.to_csv(stats_path, index=False)
    print(f"\n✓ Fold statistics saved to: {stats_path}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nUse this file in your training script:")
    print(f"  {save_path}")
    print("\nThis replaces your old splits file:")
    print(f"  {CONFIG['csv_path'].replace('ground_truth.csv', '../Dataset/kfold_splits.pkl')}")