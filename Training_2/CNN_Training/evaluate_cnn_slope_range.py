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
    'cnn_predictions_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\predictions_mse_norm_attention'),
    'results_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\slope_range_results\mse_norm_attention'),
    'n_folds': 5,
}

CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

all_slopes = []

#Evaluate slope ranges for each fold
#Compute mean, std, median, min, max

for fold_idx in range(CONFIG['n_folds']):
    print(f"\n{'='*80}")
    print(f"CNN SLOPE RANGE: FOLD {fold_idx+1}/{CONFIG['n_folds']}")
    print(f"{'='*80}")
    pred_path = CONFIG['cnn_predictions_dir'] / f'cnn_predictions_fold{fold_idx}.pkl'
    if not pred_path.exists():
        print(f"  ⚠️ Predictions not found: {pred_path}")
        continue
    with open(pred_path, 'rb') as f:
        predictions_all = pickle.load(f)
    # Collect slopes from test set
    slopes = list(predictions_all.get('test', {}).values())
    if not slopes:
        print(f"  No test predictions in fold {fold_idx+1}")
        continue
    slopes = np.array(slopes)
    all_slopes.extend([{'fold': fold_idx+1, 'slope': s} for s in slopes])
    print(f"  Fold {fold_idx+1}: N={len(slopes)} | Min={slopes.min():.4f} | Max={slopes.max():.4f} | Mean={slopes.mean():.4f} | Std={slopes.std():.4f} | Median={np.median(slopes):.4f}")
    # Plot histogram for this fold
    plt.figure(figsize=(6,4))
    plt.hist(slopes, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Slope Distribution - CNN Fold {fold_idx+1}')
    plt.xlabel('Predicted Slope')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(CONFIG['results_dir'] / f'cnn_fold{fold_idx+1}_slope_hist.png')
    plt.close()

# Compute and print average statistics across all folds
if all_slopes:
    all_slopes_array = np.array([row['slope'] for row in all_slopes])
    print("\n" + "="*80)
    print("CNN SLOPE RANGE: AGGREGATE STATISTICS (All Folds)")
    print("="*80)
    print(f"  N={len(all_slopes_array)} | Min={all_slopes_array.min():.4f} | Max={all_slopes_array.max():.4f} | Mean={all_slopes_array.mean():.4f} | Std={all_slopes_array.std():.4f} | Median={np.median(all_slopes_array):.4f}")
    

# Save all slopes to CSV
if all_slopes:
    df_all_slopes = pd.DataFrame(all_slopes)
    csv_path = CONFIG['results_dir'] / 'all_slopes.csv'
    df_all_slopes.to_csv(csv_path, index=False)
    print(f"\n✓ Saved all slopes to CSV: {csv_path}")

print("\n" + "="*80)
print("✅ CNN SLOPE RANGE EVALUATION COMPLETE")
print("="*80)
print(f"\nResults saved in: {CONFIG['results_dir']}")
