"""
Evaluate MLP Corrector Slope Predictions
=======================================

For each feature type and fold:
- Loads predicted slopes from MLP corrector (test set)
- Prints min, max, mean, std of predicted slopes
- Optionally saves all slopes to CSV
- Plots histogram of slope distribution
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

CONFIG = {
    'mlp_predictions_dir': Path('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\MLP_Corrector\\Cyclic_kfold\\predictions\\mse'),
    'results_dir': Path('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\MLP_Corrector\\Cyclic_kfold\\slope_range_results\\mse'),
    'n_folds': 5,
    'feature_types': ['demographics', 'handcrafted', 'full'],
}

CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

all_slopes = []

for feature_type in CONFIG['feature_types']:
    print(f"\n{'='*80}")
    print(f"MLP CORRECTOR SLOPE RANGE: {feature_type.upper()}")
    print(f"{'='*80}")
    feature_slopes = []
    for fold_idx in range(CONFIG['n_folds']):
        pred_path = CONFIG['mlp_predictions_dir'] / f'{feature_type}_predictions_fold{fold_idx}.pkl'
        
        if not pred_path.exists():
            print(f"  ⚠️ Predictions not found: {pred_path}")
            continue
        with open(pred_path, 'rb') as f:
            predictions_all = pickle.load(f)
        # Collect slopes from test set
        slopes = list(predictions_all.get('test', {}).values())
        if not slopes:
            print(f"  No test predictions in fold {fold_idx}")
            continue
        slopes = np.array(slopes)
        feature_slopes.extend(slopes)
        all_slopes.extend([{'feature_type': feature_type, 'fold': fold_idx, 'slope': s} for s in slopes])
        print(f"  Fold {fold_idx}: N={len(slopes)} | Min={slopes.min():.4f} | Max={slopes.max():.4f} | Mean={slopes.mean():.4f} | Std={slopes.std():.4f}")
        # Plot histogram for this fold
        plt.figure(figsize=(6,4))
        plt.hist(slopes, bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Slope Distribution - {feature_type} Fold {fold_idx}')
        plt.xlabel('Predicted Slope')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(CONFIG['results_dir'] / f'{feature_type}_fold{fold_idx}_slope_hist.png')
        plt.close()
    # Aggregate stats for feature type
    if feature_slopes:
        feature_slopes = np.array(feature_slopes)
        print(f"\n{feature_type.upper()} AGGREGATE:")
        print(f"  N={len(feature_slopes)} | Min={feature_slopes.min():.4f} | Max={feature_slopes.max():.4f} | Mean={feature_slopes.mean():.4f} | Std={feature_slopes.std():.4f}")
        # Plot aggregate histogram
        plt.figure(figsize=(7,5))
        plt.hist(feature_slopes, bins=40, color='salmon', edgecolor='black')
        plt.title(f'Slope Distribution - {feature_type} (All Folds)')
        plt.xlabel('Predicted Slope')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(CONFIG['results_dir'] / f'{feature_type}_all_folds_slope_hist.png')
        plt.close()

print("\n" + "="*80)
print("✅ MLP SLOPE RANGE EVALUATION COMPLETE")
print("="*80)
print(f"\nResults saved in: {CONFIG['results_dir']}")
