"""
Create K-Fold splits stratified by event for survival analysis
Uses 5 bins with cyclic train/val/test assignment (3/1/1)
Each patient appears in test exactly once and in validation exactly once
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def create_stratified_folds(survival_data_path, output_path, n_bins=5, random_state=42):
    """
    Create stratified bins with cyclic train/val/test splits
    Strategy: 5 bins → each fold uses 3 for train, 1 for val, 1 for test
    No patient appears in test or val more than once across all folds
    
    Args:
        survival_data_path: Path to CSV with Patient, event, time columns
        output_path: Path to save pickle file with fold splits
        n_bins: Number of bins (default: 5)
        random_state: Random seed for reproducibility
    """
    
    print(f"\n{'='*80}")
    print(f"CREATING STRATIFIED CYCLIC FOLDS (TRAIN/VAL/TEST)")
    print(f"{'='*80}")
    
    # Load survival data
    df = pd.read_csv(survival_data_path)
    print(f"\n📊 Loaded survival data:")
    print(f"   Total patients: {len(df)}")
    print(f"   Events (progression): {df['event'].sum()} ({df['event'].sum()/len(df)*100:.1f}%)")
    print(f"   Censored: {(df['event']==0).sum()} ({(df['event']==0).sum()/len(df)*100:.1f}%)")
    print(f"   Time range: {df['time'].min()}-{df['time'].max()} weeks")
    
    # Extract patient IDs and events
    patient_ids = df['Patient'].values
    events = df['event'].values
    
    # Create stratified bins using KFold
    skf = StratifiedKFold(n_splits=n_bins, shuffle=True, random_state=random_state)
    
    # First, create the bins
    bins = [None] * n_bins
    for bin_idx, (_, fold_idx) in enumerate(skf.split(patient_ids, events)):
        bins[bin_idx] = patient_ids[fold_idx].tolist()
    
    print(f"\n📦 Created {n_bins} stratified bins:")
    for i, bin_patients in enumerate(bins, 1):
        bin_events = df[df['Patient'].isin(bin_patients)]['event']
        print(f"   Bin {i}: {len(bin_patients)} patients "
              f"({bin_events.sum()} events, {(bin_events==0).sum()} censored)")
    
    # Store fold splits with cyclic assignment
    fold_splits = {}
    
    print(f"\n{'='*80}")
    print(f"CREATING {n_bins} CYCLIC FOLDS (3 train / 1 val / 1 test)")
    print(f"{'='*80}")
    
    for fold_num in range(1, n_bins + 1):
        # Cyclic assignment:
        # Fold 1: train=[0,1,2], val=[3], test=[4]
        # Fold 2: train=[1,2,3], val=[4], test=[0]
        # Fold 3: train=[2,3,4], val=[0], test=[1]
        # Fold 4: train=[3,4,0], val=[1], test=[2]
        # Fold 5: train=[4,0,1], val=[2], test=[3]
        
        test_bin_idx = (fold_num - 1) % n_bins
        val_bin_idx = (fold_num - 1 + n_bins - 1) % n_bins  # One before test
        
        train_bin_indices = []
        for i in range(n_bins):
            if i != test_bin_idx and i != val_bin_idx:
                train_bin_indices.append(i)
        
        # Combine patients from bins
        train_patients = []
        for idx in train_bin_indices:
            train_patients.extend(bins[idx])
        
        val_patients = bins[val_bin_idx]
        test_patients = bins[test_bin_idx]
        
        # Get event distributions
        train_df = df[df['Patient'].isin(train_patients)]
        val_df = df[df['Patient'].isin(val_patients)]
        test_df = df[df['Patient'].isin(test_patients)]
        
        # Store fold
        fold_splits[f'fold_{fold_num}'] = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        # Print statistics
        print(f"\n📁 FOLD {fold_num} (Bins: train={train_bin_indices}, val=[{val_bin_idx}], test=[{test_bin_idx}])")
        print(f"   Train: {len(train_patients)} patients")
        print(f"      - Events: {train_df['event'].sum()} ({train_df['event'].sum()/len(train_patients)*100:.1f}%)")
        print(f"      - Censored: {(train_df['event']==0).sum()} ({(train_df['event']==0).sum()/len(train_patients)*100:.1f}%)")
        print(f"   Val: {len(val_patients)} patients")
        print(f"      - Events: {val_df['event'].sum()} ({val_df['event'].sum()/len(val_patients)*100:.1f}%)")
        print(f"      - Censored: {(val_df['event']==0).sum()} ({(val_df['event']==0).sum()/len(val_patients)*100:.1f}%)")
        print(f"   Test: {len(test_patients)} patients")
        print(f"      - Events: {test_df['event'].sum()} ({test_df['event'].sum()/len(test_patients)*100:.1f}%)")
        print(f"      - Censored: {(test_df['event']==0).sum()} ({(test_df['event']==0).sum()/len(test_patients)*100:.1f}%)")
    
    # Save to pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(fold_splits, f)
    
    print(f"\n{'='*80}")
    print(f"✓ FOLD SPLITS SAVED")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Total folds: {n_bins}")
    
    # Verify no duplicates in test/val
    verify_no_duplicates(fold_splits)
    
    # Verify stratification quality
    verify_stratification(df, fold_splits)
    
    return fold_splits


def verify_no_duplicates(fold_splits):
    """Verify that no patient appears in test or val more than once"""
    
    print(f"\n{'='*80}")
    print(f"VERIFYING NO DUPLICATES IN TEST/VAL")
    print(f"{'='*80}")
    
    all_test = []
    all_val = []
    
    for fold_name, fold_data in fold_splits.items():
        all_test.extend(fold_data['test'])
        all_val.extend(fold_data['val'])
    
    # Check for duplicates
    test_counts = Counter(all_test)
    val_counts = Counter(all_val)
    
    test_duplicates = [p for p, count in test_counts.items() if count > 1]
    val_duplicates = [p for p, count in val_counts.items() if count > 1]
    
    if len(test_duplicates) == 0:
        print(f"✅ Test sets: No duplicates (each patient in test exactly once)")
    else:
        print(f"❌ Test sets: Found {len(test_duplicates)} patients appearing multiple times")
    
    if len(val_duplicates) == 0:
        print(f"✅ Validation sets: No duplicates (each patient in val exactly once)")
    else:
        print(f"❌ Validation sets: Found {len(val_duplicates)} patients appearing multiple times")
    
    # Verify coverage
    total_unique_test = len(set(all_test))
    total_unique_val = len(set(all_val))
    
    print(f"\n📊 Coverage:")
    print(f"   Unique patients in test sets: {total_unique_test}")
    print(f"   Unique patients in val sets: {total_unique_val}")


def verify_stratification(df, fold_splits):
    """Verify that stratification is balanced across folds"""
    
    print(f"\n{'='*80}")
    print(f"STRATIFICATION QUALITY CHECK")
    print(f"{'='*80}")
    
    overall_event_rate = df['event'].mean()
    print(f"\n📊 Overall event rate: {overall_event_rate:.1%}")
    
    # Check each fold
    print(f"\n{'Fold':<8} {'Train':<12} {'Val':<12} {'Test':<12}")
    print(f"{'-'*50}")
    
    train_rates = []
    val_rates = []
    test_rates = []
    
    for fold_name, fold_data in fold_splits.items():
        train_patients = fold_data['train']
        val_patients = fold_data['val']
        test_patients = fold_data['test']
        
        train_event_rate = df[df['Patient'].isin(train_patients)]['event'].mean()
        val_event_rate = df[df['Patient'].isin(val_patients)]['event'].mean()
        test_event_rate = df[df['Patient'].isin(test_patients)]['event'].mean()
        
        train_rates.append(train_event_rate)
        val_rates.append(val_event_rate)
        test_rates.append(test_event_rate)
        
        print(f"{fold_name:<8} {train_event_rate:<12.1%} {val_event_rate:<12.1%} {test_event_rate:<12.1%}")
    
    # Calculate statistics
    print(f"\n{'Split':<15} {'Mean':<12} {'Std Dev':<12} {'Range':<15}")
    print(f"{'-'*55}")
    print(f"{'Train':<15} {np.mean(train_rates):<12.1%} {np.std(train_rates):<12.1%} "
          f"{min(train_rates):.1%}-{max(train_rates):.1%}")
    print(f"{'Validation':<15} {np.mean(val_rates):<12.1%} {np.std(val_rates):<12.1%} "
          f"{min(val_rates):.1%}-{max(val_rates):.1%}")
    print(f"{'Test':<15} {np.mean(test_rates):<12.1%} {np.std(test_rates):<12.1%} "
          f"{min(test_rates):.1%}-{max(test_rates):.1%}")
    
    # Quality assessment
    max_std = max(np.std(train_rates), np.std(val_rates), np.std(test_rates))
    
    if max_std < 0.05:
        print(f"\n✅ EXCELLENT stratification (max std < 5%)")
    elif max_std < 0.10:
        print(f"\n✓ GOOD stratification (max std < 10%)")
    else:
        print(f"\n⚠️  WARNING: High variance in stratification (max std > 10%)")


def main():
    """Main execution"""
    
    # Configuration
    survival_data_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\patient_event_slice.csv")
    output_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\survival_folds_stratified.pkl")
    
    # Create new stratified folds
    fold_splits = create_stratified_folds(
        survival_data_path=survival_data_path,
        output_path=output_path,
        n_bins=5,
        random_state=42
    )
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE")
    print(f"{'='*80}")
    print(f"\nFold structure:")
    print(f"  - 5 bins total")
    print(f"  - Each fold: 3 bins train, 1 bin val, 1 bin test")
    print(f"  - Each patient in test exactly once")
    print(f"  - Each patient in val exactly once")
    print(f"\nTo use these folds in Cox analysis, update CONFIG:")
    print(f'  "kfold_splits_path": Path(r"{output_path}")')


if __name__ == '__main__':
    main()
