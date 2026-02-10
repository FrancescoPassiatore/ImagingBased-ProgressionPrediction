import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Thesis_training\\Dataset\\slice_cnn_features_grouped.csv')
feature_cols = [col for col in df.columns if col.startswith('cnn_feature_')]

print("="*70)
print("FEATURE QUALITY ANALYSIS")
print("="*70)

# 1. Basic statistics
print(f"\n1. FEATURE STATISTICS:")
print(f"   Feature dimension: {len(feature_cols)}")
print(f"   Mean: {df[feature_cols].mean().mean():.4f}")
print(f"   Std: {df[feature_cols].std().mean():.4f}")
print(f"   Min: {df[feature_cols].min().min():.4f}")
print(f"   Max: {df[feature_cols].max().max():.4f}")
print(f"   Any NaN: {df[feature_cols].isna().any().any()}")
print(f"   Any Inf: {np.isinf(df[feature_cols].values).any()}")

# 2. Class separation analysis (IMPORTANT!)
print(f"\n2. CLASS DISCRIMINATION:")
prog_features = df[df['gt_has_progressed']==1][feature_cols].values
no_prog_features = df[df['gt_has_progressed']==0][feature_cols].values

prog_mean = prog_features.mean(axis=0)
no_prog_mean = no_prog_features.mean(axis=0)

# Mean absolute difference per feature
feature_diffs = np.abs(prog_mean - no_prog_mean)
mean_diff = feature_diffs.mean()
max_diff = feature_diffs.max()
min_diff = feature_diffs.min()

print(f"   Progression slices: {len(prog_features)}")
print(f"   No progression slices: {len(no_prog_features)}")
print(f"   Mean absolute feature difference: {mean_diff:.4f}")
print(f"   Max feature difference: {max_diff:.4f}")
print(f"   Min feature difference: {min_diff:.4f}")
print(f"   Std of differences: {feature_diffs.std():.4f}")

if mean_diff < 0.01:
    print(f"   ⚠️ WARNING: Very low discrimination ({mean_diff:.4f}) - features may not be useful!")
elif mean_diff < 0.05:
    print(f"   ⚠️ CAUTION: Low discrimination ({mean_diff:.4f}) - challenging task")
elif mean_diff < 0.15:
    print(f"   ✓ MODERATE: Reasonable discrimination ({mean_diff:.4f})")
else:
    print(f"   ✓ GOOD: Strong discrimination ({mean_diff:.4f})")

# 3. Patient-level aggregation check
print(f"\n3. PATIENT-LEVEL ANALYSIS:")
patient_features = df.groupby('patient_id')[feature_cols].mean()
patient_labels = df.groupby('patient_id')['gt_has_progressed'].first()

prog_patient_features = patient_features[patient_labels==1].values
no_prog_patient_features = patient_features[patient_labels==0].values

patient_mean_diff = np.abs(prog_patient_features.mean(axis=0) - no_prog_patient_features.mean(axis=0)).mean()
print(f"   Total patients: {len(patient_labels)}")
print(f"   Progression patients: {(patient_labels==1).sum()}")
print(f"   No progression patients: {(patient_labels==0).sum()}")
print(f"   Patient-level mean feature difference: {patient_mean_diff:.4f}")

# 4. Feature variance check
print(f"\n4. FEATURE VARIANCE:")
feature_vars = df[feature_cols].var()
print(f"   Mean variance: {feature_vars.mean():.4f}")
print(f"   Min variance: {feature_vars.min():.4f}")
print(f"   Max variance: {feature_vars.max():.4f}")
print(f"   Features with near-zero variance (<0.001): {(feature_vars < 0.001).sum()}")

if (feature_vars < 0.001).sum() > 100:
    print(f"   ⚠️ WARNING: Many low-variance features - consider feature selection")

# 5. Visualize top discriminative features
print(f"\n5. TOP DISCRIMINATIVE FEATURES:")
top_5_idx = np.argsort(feature_diffs)[-5:]
for idx in reversed(top_5_idx):
    feat_name = feature_cols[idx]
    diff = feature_diffs[idx]
    prog_val = prog_mean[idx]
    no_prog_val = no_prog_mean[idx]
    print(f"   {feat_name}: diff={diff:.4f} (prog={prog_val:.4f}, no_prog={no_prog_val:.4f})")

print("\n" + "="*70)