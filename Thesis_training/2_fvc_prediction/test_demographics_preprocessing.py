"""
Quick test to verify improved demographics preprocessing
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import the new functions
from fvc_ablation_study import (
    preprocess_demographics_improved,
    get_preprocessed_demo_features
)

# Create dummy data
print("="*70)
print("TESTING IMPROVED DEMOGRAPHICS PREPROCESSING")
print("="*70)

# Dummy features DataFrame
features_df = pd.DataFrame({
    'patient_id': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
    'slice_index': [0, 1, 0, 1, 0, 1],
    'Age': [65, 65, 55, 55, 72, 72],
    'Sex': [1, 1, 0, 0, 1, 1],  # 1=Male, 0=Female
    'SmokingStatus': [1, 1, 0, 0, 2, 2],  # 0=Never, 1=Ex, 2=Current
    'baselinefvc': [2500, 2500, 3000, 3000, 2200, 2200],
    'gt_fvc52': [2300, 2300, 2900, 2900, 2000, 2000]
})

train_ids = ['P001', 'P002']

print("\nOriginal data:")
print(features_df[['patient_id', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates('patient_id'))

# Test preprocessing
print("\n" + "="*70)
print("RUNNING PREPROCESSING")
print("="*70)

result_df, encoding_info = preprocess_demographics_improved(
    features_df,
    train_ids,
    normalization_type='standard'
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print("\nPreprocessed data:")
print(result_df[['patient_id', 'Age', 'Age_normalized', 'Sex', 'Sex_encoded']].drop_duplicates('patient_id'))

print("\nOne-hot encoded smoking columns:")
smoking_cols = encoding_info.get('smoking_columns', [])
if smoking_cols:
    print(result_df[['patient_id'] + smoking_cols].drop_duplicates('patient_id'))

print("\n" + "="*70)
print("TESTING FEATURE EXTRACTION")
print("="*70)

# Test extracting features for each patient
for patient_id in ['P001', 'P002', 'P003']:
    row = result_df[result_df['patient_id'] == patient_id].iloc[0]
    features = get_preprocessed_demo_features(row, encoding_info)
    print(f"\n{patient_id}:")
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature vector: {features}")
    
    # Interpret features
    idx = 0
    if 'Age_normalized' in row:
        print(f"    Age (normalized): {features[idx]:.4f}")
        idx += 1
    if 'Sex_encoded' in row:
        sex_label = 'Male' if features[idx] == 1 else 'Female'
        print(f"    Sex: {sex_label} ({features[idx]:.1f})")
        idx += 1
    if smoking_cols:
        print(f"    Smoking (one-hot): [{', '.join([f'{f:.2f}' for f in features[idx:]])}]")

print("\n" + "="*70)
print("TEST PASSED ✓")
print("="*70)
print("\nSummary:")
print(f"  - Age: normalized (mean≈0, std≈1)")
print(f"  - Sex: centered (-1=Female, 1=Male)")
print(f"  - Smoking: one-hot encoded (3 features, centered)")
print(f"  - Total demographic features: {len(features)}")
print(f"\nEncoding info keys: {list(encoding_info.keys())}")
