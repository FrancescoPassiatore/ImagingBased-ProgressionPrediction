"""
DIAGNOSTIC SCRIPT: Why Are Features Not Working?
=================================================
Run this to understand if the problem is in the features or the model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import pickle

def diagnose_features(features_df, patient_ids_train, patient_ids_val):
    """
    Comprehensive feature diagnostic
    """
    print("\n" + "="*70)
    print("FEATURE DIAGNOSTIC")
    print("="*70)
    
    # Get patient-level features (aggregate slices)
    feature_cols = [col for col in features_df.columns 
                    if col.startswith('cnn_feature_')]
    
    # Aggregate by MAX (as in your model)
    patient_features = features_df.groupby('patient_id').agg({
        **{col: 'max' for col in feature_cols},
        'gt_has_progressed': 'first'
    }).reset_index()
    
    # Split train/val
    train_df = patient_features[patient_features['patient_id'].isin(patient_ids_train)]
    val_df = patient_features[patient_features['patient_id'].isin(patient_ids_val)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['gt_has_progressed'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['gt_has_progressed'].values
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)} patients")
    print(f"  Val: {len(X_val)} patients")
    print(f"  Features: {X_train.shape[1]}")
    
    # ========================================================================
    # 1. CLASS SEPARABILITY
    # ========================================================================
    
    print("\n" + "="*70)
    print("1. CLASS SEPARABILITY ANALYSIS")
    print("="*70)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # t-SNE to 2D
    print("\nComputing t-SNE (may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_train)-1))
    X_train_tsne = tsne.fit_transform(X_train)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    for label, color, name in [(0, 'blue', 'No Progression'), (1, 'red', 'Progression')]:
        mask_train = y_train == label
        mask_val = y_val == label
        axes[0].scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1], 
                       c=color, label=f'{name} (train)', alpha=0.6, s=100)
        axes[0].scatter(X_val_pca[mask_val, 0], X_val_pca[mask_val, 1], 
                       c=color, marker='x', label=f'{name} (val)', s=100, linewidths=3)
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA Projection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for label, color, name in [(0, 'blue', 'No Progression'), (1, 'red', 'Progression')]:
        mask = y_train == label
        axes[1].scatter(X_train_tsne[mask, 0], X_train_tsne[mask, 1], 
                       c=color, label=name, alpha=0.6, s=100)
    
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_separability.png', dpi=150)
    print("\n✓ Saved: feature_separability.png")
    
    # Interpretation
    print("\n📊 Interpretation:")
    if pca.explained_variance_ratio_.sum() < 0.5:
        print("  ⚠️  Very low PCA variance captured - features are diffuse")
    
    # Check if classes overlap in PCA space
    from scipy.spatial.distance import cdist
    prog_centers = X_train_pca[y_train == 1].mean(axis=0)
    no_prog_centers = X_train_pca[y_train == 0].mean(axis=0)
    separation = np.linalg.norm(prog_centers - no_prog_centers)
    
    print(f"  Class separation in PCA space: {separation:.2f}")
    if separation < 1.0:
        print("  ⚠️  Classes are VERY close - features may not be discriminative!")
    
    # ========================================================================
    # 2. SIMPLE BASELINE MODELS
    # ========================================================================
    
    print("\n" + "="*70)
    print("2. BASELINE MODEL COMPARISON")
    print("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest (10 trees)': RandomForestClassifier(n_estimators=10, random_state=42),
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else 0.5
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
        
        print(f"\n{name}:")
        print(f"  Train AUC: {train_auc:.3f}")
        print(f"  Val AUC:   {val_auc:.3f}")
        print(f"  Gap:       {train_auc - val_auc:.3f}")
        
        if val_auc < 0.6:
            print("  ⚠️  Even simple models struggle - features may be weak!")
    
    # ========================================================================
    # 3. FEATURE STATISTICS
    # ========================================================================
    
    print("\n" + "="*70)
    print("3. FEATURE STATISTICS")
    print("="*70)
    
    # Check for constant/near-constant features
    feature_stds = X_train.std(axis=0)
    low_var_features = (feature_stds < 0.01).sum()
    
    print(f"\nFeature variance:")
    print(f"  Low variance features (<0.01): {low_var_features}/{len(feature_stds)}")
    if low_var_features > len(feature_stds) * 0.1:
        print("  ⚠️  Many low-variance features - may need feature selection!")
    
    # Check for NaNs/Infs
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()
    
    print(f"\nData quality:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  ⚠️  Data quality issues detected!")
    
    # Feature distribution comparison
    prog_mean = X_train[y_train == 1].mean(axis=0)
    no_prog_mean = X_train[y_train == 0].mean(axis=0)
    mean_diff = np.abs(prog_mean - no_prog_mean)
    
    print(f"\nFeature discrimination:")
    print(f"  Mean absolute difference: {mean_diff.mean():.4f}")
    print(f"  Max difference: {mean_diff.max():.4f}")
    print(f"  Features with diff > 0.1: {(mean_diff > 0.1).sum()}/{len(mean_diff)}")
    
    if (mean_diff > 0.1).sum() < 10:
        print("  ⚠️  Very few features show class difference - weak signal!")
    
    # ========================================================================
    # 4. AGGREGATION STRATEGY TEST
    # ========================================================================
    
    print("\n" + "="*70)
    print("4. AGGREGATION STRATEGY COMPARISON")
    print("="*70)
    
    aggregations = ['max', 'mean', 'median', 'std', 'percentile_90']
    
    for agg in aggregations:
        if agg == 'percentile_90':
            patient_agg = features_df.groupby('patient_id').agg({
                **{col: lambda x: np.percentile(x, 90) for col in feature_cols},
                'gt_has_progressed': 'first'
            }).reset_index()
        else:
            patient_agg = features_df.groupby('patient_id').agg({
                **{col: agg for col in feature_cols},
                'gt_has_progressed': 'first'
            }).reset_index()
        
        train_agg = patient_agg[patient_agg['patient_id'].isin(patient_ids_train)]
        val_agg = patient_agg[patient_agg['patient_id'].isin(patient_ids_val)]
        
        X_train_agg = train_agg[feature_cols].values
        y_train_agg = train_agg['gt_has_progressed'].values
        X_val_agg = val_agg[feature_cols].values
        y_val_agg = val_agg['gt_has_progressed'].values
        
        # Quick RF test
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train_agg, y_train_agg)
        val_pred = rf.predict_proba(X_val_agg)[:, 1]
        val_auc = roc_auc_score(y_val_agg, val_pred) if len(np.unique(y_val_agg)) > 1 else 0.5
        
        print(f"\n{agg.upper()} aggregation:")
        print(f"  Val AUC: {val_auc:.3f}")
    
    # ========================================================================
    # 5. RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "="*70)
    print("5. DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    
    # Check various indicators
    if val_auc < 0.65:  # From baseline models
        issues.append("⚠️  WEAK FEATURES: Even simple models achieve <0.65 AUC")
    
    if separation < 1.0:
        issues.append("⚠️  POOR SEPARABILITY: Classes overlap heavily in feature space")
    
    if low_var_features > len(feature_stds) * 0.1:
        issues.append("⚠️  REDUNDANT FEATURES: Many low-variance features")
    
    if (mean_diff > 0.1).sum() < 10:
        issues.append("⚠️  WEAK SIGNAL: Very few features show class differences")
    
    if len(X_train) < 50:
        issues.append("⚠️  SMALL DATASET: <50 training samples - prone to overfitting")
    
    if issues:
        print("\n🚨 ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        print("  1. Try different backbone (ResNet50 instead of EfficientNet)")
        print("  2. Use pretrained features from ImageNet + fine-tuning")
        print("  3. Add clinical features (age, FVC, DLCO, etc.)")
        print("  4. Try mean/median aggregation instead of max")
        print("  5. Consider ensemble of multiple aggregations")
        print("  6. Collect more data if possible")
        print("  7. Try different slice selection (only central slices)")
    else:
        print("\n✅ Features look reasonable - problem may be in model architecture")
    
    return {
        'pca_variance': pca.explained_variance_ratio_.sum(),
        'class_separation': separation,
        'low_var_features': low_var_features,
        'discriminative_features': (mean_diff > 0.1).sum()
    }


# ============================================================================
# QUICK FIX: Try Different Aggregations
# ============================================================================

def quick_test_aggregations(features_df, fold_data):
    """
    Quick test of different aggregation strategies
    """
    print("\n" + "="*70)
    print("QUICK AGGREGATION TEST")
    print("="*70)
    
    feature_cols = [col for col in features_df.columns if col.startswith('cnn_feature_')]
    
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    
    results = []
    
    for agg_name, agg_func in [
        ('max', 'max'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('std', 'std'),
        ('percentile_75', lambda x: x.quantile(0.75)),
        ('percentile_90', lambda x: x.quantile(0.90))
    ]:
        print(f"\nTesting {agg_name} aggregation...")
        
        # Aggregate
        if callable(agg_func):
            patient_df = features_df.groupby('patient_id').agg({
                **{col: agg_func for col in feature_cols},
                'gt_has_progressed': 'first'
            }).reset_index()
        else:
            patient_df = features_df.groupby('patient_id').agg({
                **{col: agg_func for col in feature_cols},
                'gt_has_progressed': 'first'
            }).reset_index()
        
        # Split
        train_df = patient_df[patient_df['patient_id'].isin(train_ids)]
        val_df = patient_df[patient_df['patient_id'].isin(val_ids)]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['gt_has_progressed'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['gt_has_progressed'].values
        
        # Train simple RF
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        train_pred = rf.predict_proba(X_train)[:, 1]
        val_pred = rf.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
        
        print(f"  Train AUC: {train_auc:.3f}, Val AUC: {val_auc:.3f}, Gap: {train_auc-val_auc:.3f}")
        
        results.append({
            'aggregation': agg_name,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'gap': train_auc - val_auc
        })
    
    # Summary
    results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
    
    print("\n" + "="*70)
    print("AGGREGATION COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n✅ BEST AGGREGATION: {best['aggregation']}")
    print(f"   Val AUC: {best['val_auc']:.3f}")
    
    if best['val_auc'] > 0.70:
        print("   🎉 This aggregation works well!")
    else:
        print("   ⚠️  Even best aggregation <0.70 - features may be weak")
    
    return results_df


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Run diagnostics on your features
    """
    
    # Load your data
    features_df = pd.read_csv("D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Thesis_training\\1_progression\\analysis\\slice_cnn_efficientnet_b0_features_grouped.csv")
    
    with open("D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Thesis_training\\Dataset\\kfold_splits.pkl", 'rb') as f:
        splits = pickle.load(f)
    
    print(splits.keys())
    fold_0 = splits[0]
    
    # Run diagnostics
    diagnostic_results = diagnose_features(
        features_df=features_df,
        patient_ids_train=fold_0['train'],
        patient_ids_val=fold_0['val']
    )
    
    # Test aggregations
    agg_results = quick_test_aggregations(
        features_df=features_df,
        fold_data=fold_0
    )