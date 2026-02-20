"""
Quick verification that the updates work correctly
Tests loading of features and fold splits
"""

import pickle
import pandas as pd
from pathlib import Path

print("="*80)
print("VERIFICATION: Features and Fold Splits")
print("="*80)

# 1. Check features CSV
features_path = r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"
print(f"\n1. Loading features from: {features_path}")

try:
    df_features = pd.read_csv(features_path)
    print(f"   ✓ Loaded successfully")
    print(f"   - Shape: {df_features.shape}")
    print(f"   - Columns: {list(df_features.columns)}")
    print(f"   - Patients: {len(df_features)}")
    print(f"\n   First few rows:")
    print(df_features.head(3))
except Exception as e:
    print(f"   ✗ Error: {e}")

# 2. Check fold splits
splits_path = r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"
print(f"\n2. Loading fold splits from: {splits_path}")

try:
    with open(splits_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of folds: {len(kfold_splits)}")
    
    print(f"\n   Fold breakdown:")
    for fold_idx in range(len(kfold_splits)):
        fold = kfold_splits[fold_idx]
        print(f"   Fold {fold_idx}:")
        print(f"     Train: {len(fold['train'])} patients")
        print(f"     Val:   {len(fold['val'])} patients")
        print(f"     Test:  {len(fold['test'])} patients")
        print(f"     Total: {len(fold['train']) + len(fold['val']) + len(fold['test'])} patients")
    
    # 3. Check overlap between features and splits
    print(f"\n3. Checking patient overlap:")
    feature_patients = set(df_features['Patient'].values)
    
    all_split_patients = set()
    for fold_idx in range(len(kfold_splits)):
        fold = kfold_splits[fold_idx]
        all_split_patients.update(fold['train'])
        all_split_patients.update(fold['val'])
        all_split_patients.update(fold['test'])
    
    print(f"   Patients in features CSV: {len(feature_patients)}")
    print(f"   Patients in fold splits: {len(all_split_patients)}")
    
    in_both = feature_patients & all_split_patients
    only_features = feature_patients - all_split_patients
    only_splits = all_split_patients - feature_patients
    
    print(f"   In both: {len(in_both)}")
    if only_features:
        print(f"   ⚠ Only in features: {len(only_features)}")
        print(f"      Examples: {list(only_features)[:3]}")
    if only_splits:
        print(f"   ⚠ Only in splits: {len(only_splits)}")
        print(f"      Examples: {list(only_splits)[:3]}")
    
    if len(in_both) > 0:
        print(f"\n   ✓ {len(in_both)} patients available for training")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
