"""
Create 5-bin stratified folds for survival analysis (84 patients)

Strategy:
- Divide 84 patients into 5 bins (stratified by event)
- Each fold uses: 3 bins for training, 1 for validation, 1 for test
- Validation and test bins rotate across folds (never repeated)
- Fold 1: train=[bin1,bin2,bin3], val=bin4, test=bin5
- Fold 2: train=[bin2,bin3,bin4], val=bin5, test=bin1
- Fold 3: train=[bin3,bin4,bin5], val=bin1, test=bin2
- Fold 4: train=[bin4,bin5,bin1], val=bin2, test=bin3
- Fold 5: train=[bin5,bin1,bin2], val=bin3, test=bin4
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def create_5bin_stratified_folds(ground_truth_path, output_path, n_bins=5, random_state=42):
    """
    Create 5-bin stratified folds with rotating train/val/test splits
    
    Args:
        ground_truth_path: Path to ground_truth_survival.csv
        output_path: Path to save fold splits (pickle file)
        n_bins: Number of bins (default: 5)
        random_state: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"CREATING 5-BIN STRATIFIED FOLDS FOR SURVIVAL ANALYSIS")
    print(f"{'='*80}")
    print(f"\nStrategy:")
    print(f"  - Divide {84} patients into {n_bins} bins (stratified by event)")
    print(f"  - Each fold: 3 bins train, 1 bin val, 1 bin test")
    print(f"  - Val and test bins rotate (never repeated)")
    
    # Load ground truth
    df = pd.read_csv(ground_truth_path)
    print(f"\n✓ Loaded ground truth: {len(df)} patients")
    
    # Get patient IDs and event labels
    patient_ids = df['PatientID'].values
    events = df['event'].values.astype(int)
    
    # Print event distribution
    n_events = events.sum()
    n_censored = len(events) - n_events
    print(f"\nOverall event distribution:")
    print(f"  Events (progression): {n_events} ({n_events/len(events)*100:.1f}%)")
    print(f"  Censored: {n_censored} ({n_censored/len(events)*100:.1f}%)")
    
    # Create 5 stratified bins using StratifiedKFold
    print(f"\n{'='*80}")
    print(f"CREATING {n_bins} STRATIFIED BINS")
    print(f"{'='*80}")
    
    skf = StratifiedKFold(n_splits=n_bins, shuffle=True, random_state=random_state)
    
    bins = {}
    for bin_idx, (_, bin_patient_indices) in enumerate(skf.split(patient_ids, events), start=1):
        bin_patients = patient_ids[bin_patient_indices]
        bin_events = events[bin_patient_indices]
        
        bins[bin_idx] = {
            'patients': bin_patients.tolist(),
            'n_patients': len(bin_patients),
            'n_events': int(bin_events.sum()),
            'event_rate': bin_events.mean()
        }
        
        print(f"\nBin {bin_idx}:")
        print(f"  Patients: {len(bin_patients)}")
        print(f"  Events: {int(bin_events.sum())} ({bin_events.mean()*100:.1f}%)")
        print(f"  Censored: {len(bin_patients) - int(bin_events.sum())} ({(1-bin_events.mean())*100:.1f}%)")
    
    # Create folds with circular rotation
    print(f"\n{'='*80}")
    print(f"CREATING {n_bins} FOLDS (CIRCULAR ROTATION)")
    print(f"{'='*80}")
    
    fold_splits = {}
    
    for fold_num in range(1, n_bins + 1):
        # Circular rotation strategy:
        # Fold i uses bins [i, i+1, i+2] for train, bin (i+3) for val, bin (i+4) for test
        # All with modulo arithmetic to wrap around
        
        train_bin_indices = [(fold_num - 1 + j) % n_bins + 1 for j in range(3)]
        val_bin_idx = (fold_num - 1 + 3) % n_bins + 1
        test_bin_idx = (fold_num - 1 + 4) % n_bins + 1
        
        # Collect patients from bins
        train_patients = []
        for bin_idx in train_bin_indices:
            train_patients.extend(bins[bin_idx]['patients'])
        
        val_patients = bins[val_bin_idx]['patients']
        test_patients = bins[test_bin_idx]['patients']
        
        # Store fold
        fold_splits[f'fold_{fold_num}'] = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        # Calculate event rates for this fold
        train_events = df[df['PatientID'].isin(train_patients)]['event'].values
        val_events = df[df['PatientID'].isin(val_patients)]['event'].values
        test_events = df[df['PatientID'].isin(test_patients)]['event'].values
        
        print(f"\nFold {fold_num}:")
        print(f"  Train bins: {train_bin_indices} → {len(train_patients)} patients")
        print(f"    Events: {train_events.sum()} ({train_events.mean()*100:.1f}%)")
        print(f"  Val bin: {val_bin_idx} → {len(val_patients)} patients")
        print(f"    Events: {val_events.sum()} ({val_events.mean()*100:.1f}%)")
        print(f"  Test bin: {test_bin_idx} → {len(test_patients)} patients")
        print(f"    Events: {test_events.sum()} ({test_events.mean()*100:.1f}%)")
    
    # Verify no repetition of val/test bins
    print(f"\n{'='*80}")
    print(f"VERIFYING FOLD STRUCTURE")
    print(f"{'='*80}")
    
    val_bins_used = []
    test_bins_used = []
    
    for fold_num in range(1, n_bins + 1):
        train_bin_indices = [(fold_num - 1 + j) % n_bins + 1 for j in range(3)]
        val_bin_idx = (fold_num - 1 + 3) % n_bins + 1
        test_bin_idx = (fold_num - 1 + 4) % n_bins + 1
        
        val_bins_used.append(val_bin_idx)
        test_bins_used.append(test_bin_idx)
        
        # Check no overlap within fold
        assert val_bin_idx not in train_bin_indices, f"Fold {fold_num}: Val bin in train bins!"
        assert test_bin_idx not in train_bin_indices, f"Fold {fold_num}: Test bin in train bins!"
        assert val_bin_idx != test_bin_idx, f"Fold {fold_num}: Val and test are same bin!"
    
    # Check no repetition of val/test bins across folds
    assert len(val_bins_used) == len(set(val_bins_used)), f"Validation bins repeated across folds!"
    assert len(test_bins_used) == len(set(test_bins_used)), f"Test bins repeated across folds!"
    
    print(f"\n✓ Validation bins used: {val_bins_used} (all unique)")
    print(f"✓ Test bins used: {test_bins_used} (all unique)")
    print(f"✓ No overlap within folds")
    
    # Verify all patients covered in each fold
    all_patient_set = set(df['PatientID'].unique())
    for fold_num in range(1, n_bins + 1):
        fold_key = f'fold_{fold_num}'
        fold_patients = set(fold_splits[fold_key]['train'] + 
                          fold_splits[fold_key]['val'] + 
                          fold_splits[fold_key]['test'])
        assert fold_patients == all_patient_set, f"Fold {fold_num}: Not all patients covered!"
    
    print(f"✓ All patients covered in each fold")
    
    # Save to pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(fold_splits, f)
    
    print(f"\n{'='*80}")
    print(f"✓ FOLD SPLITS SAVED")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"\nStructure:")
    print(f"  fold_splits['fold_X']['train']: List of training patient IDs")
    print(f"  fold_splits['fold_X']['val']: List of validation patient IDs")
    print(f"  fold_splits['fold_X']['test']: List of test patient IDs")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    train_sizes = [len(fold_splits[f'fold_{i}']['train']) for i in range(1, n_bins+1)]
    val_sizes = [len(fold_splits[f'fold_{i}']['val']) for i in range(1, n_bins+1)]
    test_sizes = [len(fold_splits[f'fold_{i}']['test']) for i in range(1, n_bins+1)]
    
    print(f"\nSplit sizes across folds:")
    print(f"  Train: {train_sizes[0]}-{max(train_sizes)} patients (mean: {np.mean(train_sizes):.1f})")
    print(f"  Val:   {val_sizes[0]}-{max(val_sizes)} patients (mean: {np.mean(val_sizes):.1f})")
    print(f"  Test:  {test_sizes[0]}-{max(test_sizes)} patients (mean: {np.mean(test_sizes):.1f})")
    
    # Event rates across folds
    train_rates = []
    val_rates = []
    test_rates = []
    
    for fold_num in range(1, n_bins + 1):
        fold_key = f'fold_{fold_num}'
        train_events = df[df['PatientID'].isin(fold_splits[fold_key]['train'])]['event'].mean()
        val_events = df[df['PatientID'].isin(fold_splits[fold_key]['val'])]['event'].mean()
        test_events = df[df['PatientID'].isin(fold_splits[fold_key]['test'])]['event'].mean()
        
        train_rates.append(train_events)
        val_rates.append(val_events)
        test_rates.append(test_events)
    
    print(f"\nEvent rates across folds:")
    print(f"  Train: {np.mean(train_rates)*100:.1f}% ± {np.std(train_rates)*100:.1f}%")
    print(f"  Val:   {np.mean(val_rates)*100:.1f}% ± {np.std(val_rates)*100:.1f}%")
    print(f"  Test:  {np.mean(test_rates)*100:.1f}% ± {np.std(test_rates)*100:.1f}%")
    
    return fold_splits, bins


if __name__ == '__main__':
    # Paths
    ground_truth_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\ground_truth_survival.csv")
    output_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_survival_analysis.pkl")
    
    # Create folds
    fold_splits, bins = create_5bin_stratified_folds(
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        n_bins=5,
        random_state=42
    )
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE!")
    print(f"{'='*80}")
