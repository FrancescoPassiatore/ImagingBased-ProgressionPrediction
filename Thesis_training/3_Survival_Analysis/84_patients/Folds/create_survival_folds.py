"""
Create stratified K-fold splits for survival analysis (84 patients)
Stratifies by event status to ensure balanced progression rates across folds
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def create_stratified_folds(ground_truth_path, output_path, n_folds=5, random_state=42):
    """
    Create stratified K-fold splits for survival analysis
    
    Args:
        ground_truth_path: Path to ground_truth_survival.csv
        output_path: Path to save the fold splits (pickle file)
        n_folds: Number of folds (default: 5)
        random_state: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"CREATING {n_folds}-FOLD STRATIFIED SPLITS FOR SURVIVAL ANALYSIS")
    print(f"{'='*80}")
    
    # Load ground truth
    df = pd.read_csv(ground_truth_path)
    print(f"\n✓ Loaded ground truth: {len(df)} patients")
    
    # Get patient IDs and event labels
    patient_ids = df['PatientID'].values
    events = df['event'].values.astype(int)
    
    # Print event distribution
    n_events = events.sum()
    n_censored = len(events) - n_events
    print(f"\nEvent distribution:")
    print(f"  Events (progression): {n_events} ({n_events/len(events)*100:.1f}%)")
    print(f"  Censored: {n_censored} ({n_censored/len(events)*100:.1f}%)")
    
    # Create stratified K-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Store splits
    fold_splits = {}
    
    print(f"\n{'='*80}")
    print(f"GENERATING FOLDS")
    print(f"{'='*80}")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, events), start=1):
        # Get train+val and test patients
        train_val_patients = patient_ids[train_val_idx]
        train_val_events = events[train_val_idx]
        test_patients = patient_ids[test_idx]
        test_events = events[test_idx]
        
        # Further split train+val into train and val (80/20 split of train_val)
        train_val_size = len(train_val_idx)
        val_size = int(train_val_size * 0.2)
        
        # Use StratifiedKFold with n_splits=5 to get one fold as val (20%)
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold_idx)
        
        # Get first split as train/val
        inner_train_idx, inner_val_idx = next(skf_inner.split(train_val_patients, train_val_events))
        
        train_patients = train_val_patients[inner_train_idx]
        train_events = train_val_events[inner_train_idx]
        val_patients = train_val_patients[inner_val_idx]
        val_events = train_val_events[inner_val_idx]
        
        # Store fold
        fold_splits[fold_idx] = {
            'train': train_patients.tolist(),
            'val': val_patients.tolist(),
            'test': test_patients.tolist()
        }
        
        # Print fold statistics
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_patients)} patients")
        print(f"    Events: {train_events.sum()} ({train_events.mean()*100:.1f}%)")
        print(f"    Censored: {len(train_events) - train_events.sum()} ({(1-train_events.mean())*100:.1f}%)")
        print(f"  Val: {len(val_patients)} patients")
        print(f"    Events: {val_events.sum()} ({val_events.mean()*100:.1f}%)")
        print(f"    Censored: {len(val_events) - val_events.sum()} ({(1-val_events.mean())*100:.1f}%)")
        print(f"  Test: {len(test_patients)} patients")
        print(f"    Events: {test_events.sum()} ({test_events.mean()*100:.1f}%)")
        print(f"    Censored: {len(test_events) - test_events.sum()} ({(1-test_events.mean())*100:.1f}%)")
    
    # Save to pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(fold_splits, f)
    
    print(f"\n{'='*80}")
    print(f"✓ FOLD SPLITS SAVED")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"\nFold structure:")
    print(f"  fold_splits[fold_num]['train']: List of training patient IDs")
    print(f"  fold_splits[fold_num]['val']: List of validation patient IDs")
    print(f"  fold_splits[fold_num]['test']: List of test patient IDs")
    
    return fold_splits


def verify_folds(fold_splits, ground_truth_path):
    """Verify that fold splits are correct"""
    print(f"\n{'='*80}")
    print(f"VERIFYING FOLD SPLITS")
    print(f"{'='*80}")
    
    df = pd.read_csv(ground_truth_path)
    all_patients = set(df['PatientID'].unique())
    
    for fold_num, split in fold_splits.items():
        train_set = set(split['train'])
        val_set = set(split['val'])
        test_set = set(split['test'])
        
        # Check no overlap
        assert len(train_set & val_set) == 0, f"Fold {fold_num}: Train and val overlap!"
        assert len(train_set & test_set) == 0, f"Fold {fold_num}: Train and test overlap!"
        assert len(val_set & test_set) == 0, f"Fold {fold_num}: Val and test overlap!"
        
        # Check all patients present
        fold_patients = train_set | val_set | test_set
        assert fold_patients == all_patients, f"Fold {fold_num}: Not all patients included!"
        
        print(f"✓ Fold {fold_num}: No overlap, all patients present")
    
    print(f"\n✓ All folds verified successfully!")


if __name__ == '__main__':
    # Paths
    ground_truth_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\ground_truth_survival.csv")
    output_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\survival_folds_stratified.pkl")
    
    # Create folds
    fold_splits = create_stratified_folds(
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        n_folds=5,
        random_state=42
    )
    
    # Verify folds
    verify_folds(fold_splits, ground_truth_path)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE!")
    print(f"{'='*80}")
