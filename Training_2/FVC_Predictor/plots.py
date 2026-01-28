"""
Plot FVC(52) Prediction Errors by Patient
==========================================

Creates two plots showing prediction errors colored by progression status:
1. Error relative to true FVC(52)
2. Error relative to baseline (intercept)

Loads predictions from all_predictions.pkl with structure:
{
    'fold_0': {
        'train': {pid: slope_pred},
        'val': {pid: slope_pred},
        'test': {pid: slope_pred}
    },
    ...
}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'predictions_file': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\predictions_cyclic_kfold\direct\all_predictions.pkl'),
    'plots_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\error_analysis'),
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'csv_features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    'patient_progression_gt': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv',
    'progression_threshold': 10.0

}

CONFIG['plots_dir'].mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ground_truth_progression(patient_data, threshold_percent=10.0):
    
    #Extrapolate from config['patient_progression_gt'] labels for each patient
    gt_df = pd.read_csv(CONFIG['patient_progression_gt'])
    gt_dict = dict(zip(gt_df['Patient'], gt_df['event_52']))
    print(f"✓ Loaded ground truth progression labels for {len(gt_dict)} patients")
    
    #------------------------------------------------------------
    #                   Patient  event_52  fvc_52  week_52
    #ID00007637202177411956430       1.0  2057.0     57.0
    #ID00009637202177434476278       0.0  3390.0     45.0
    #ID00010637202177584971671       1.0  2518.0     54.0
    #ID00011637202177653955184       0.0  3193.0     58.0
    #ID00012637202177665765362       0.0  3449.0     47.0
    #------------------------------------------------------------

    labels = {}
    for pid in patient_data.keys():
        if pid in gt_dict:
            labels[pid] = bool(gt_dict[pid])
        else:
            continue # remove missing labels
    
    print(f"✓ Generated ground truth progression status for {len(labels)} patients")
    
    
    
    return labels


def get_true_fvc52(patient_data, patient_id):
    """Get true FVC at week 52"""
    gt_df = pd.read_csv(CONFIG['patient_progression_gt'])
    gt_dict = dict(zip(gt_df['Patient'], gt_df['fvc_52']))
    fvc52_gt = {}

    for pid in patient_data.keys():
        if pid in gt_dict:
            fvc52_gt[pid] = gt_dict[pid]
        else:
            continue # remove missing fvc

    return fvc52_gt.get(patient_id, None)


def load_predictions_from_pkl(pkl_file, patient_data, ground_truth_labels):
    """Load predictions from all_predictions.pkl and compute errors"""
    
    print(f"\n📂 Loading predictions from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        all_predictions = pickle.load(f)
    
    print(f"  Keys found: {list(all_predictions.keys())}")
    
    # Collect all data
    all_data = []
    
    # Iterate through folds
    fold_keys = [k for k in all_predictions.keys() if k.startswith('fold_')]
    print(f"  Found {len(fold_keys)} folds")
    
    for fold_key in sorted(fold_keys):
        fold_idx = int(fold_key.split('_')[1])
        fold_data = all_predictions[fold_key]
        
        # Use test set predictions
        test_preds = fold_data['test']
        
        print(f"  Fold {fold_idx}: {len(test_preds)} test patients")
        
        for pid, fvc52_pred in test_preds.items():
            
            if pid not in ground_truth_labels or pid not in patient_data:
                continue
            
            # Get true values
            fvc52_true = get_true_fvc52(patient_data, pid)
            intercept_true = patient_data[pid]['intercept']

            if None in (fvc52_true, intercept_true):
                continue
        
            # Compute errors
            # Error 1: Relative to true FVC(52)
            delta_fvc_true = (abs(fvc52_pred - fvc52_true) / fvc52_true) * 100
            
            # Error 2: Relative to baseline (intercept)
            delta_fvc_baseline = (abs(fvc52_pred - intercept_true) / intercept_true) * 100
            
            # Get progression status
            has_progression = ground_truth_labels[pid]
            
            all_data.append({
                'patient_id': pid,
                'fold': fold_idx,
                'fvc52_pred': fvc52_pred,
                'intercept_true': intercept_true,
                'fvc52_true': fvc52_true,
                'delta_fvc_true': delta_fvc_true,
                'delta_fvc_baseline': delta_fvc_baseline,
                'has_progression': has_progression
            })
    
    print(f"✓ Processed {len(all_data)} patients")
    
    return pd.DataFrame(all_data)


def plot_error_analysis(df, config):
    """Create the two requested plots"""
    
    # Sort by patient ID for consistent ordering
    df = df.sort_values('patient_id').reset_index(drop=True)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Patient indices
    x = np.arange(len(df))
    
    # Separate progression and stable
    prog_mask = df['has_progression'] == True
    stable_mask = df['has_progression'] == False
    
    # =========================================================================
    # PLOT 1: Error relative to TRUE FVC(52)
    # =========================================================================
    ax1 = axes[0]
    
    ax1.scatter(x[prog_mask], df.loc[prog_mask, 'delta_fvc_true'], 
               color='red', s=80, alpha=0.6, label='Progression (Ground Truth)', 
               edgecolors='darkred', linewidths=1.5)
    ax1.scatter(x[stable_mask], df.loc[stable_mask, 'delta_fvc_true'], 
               color='green', s=80, alpha=0.6, label='Stable (Ground Truth)', 
               edgecolors='darkgreen', linewidths=1.5)
    
    # Add horizontal lines
    ax1.axhline(10, color='orange', linestyle='--', linewidth=2, 
               label='10% Error Threshold', alpha=0.7)
    
    ax1.set_xlabel('Patient Index', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Δ FVC (%) = |Predicted - True FVC(52)| / True FVC(52) × 100', 
                   fontsize=12, fontweight='bold')
    ax1.set_title('FVC(52) Prediction Error Relative to True FVC(52)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Statistics text
    prog_mean = df.loc[prog_mask, 'delta_fvc_true'].mean()
    stable_mean = df.loc[stable_mask, 'delta_fvc_true'].mean()
    ax1.text(0.02, 0.98, 
            f'Progression: {prog_mask.sum()} patients (Mean: {prog_mean:.1f}%)\n'
            f'Stable: {stable_mask.sum()} patients (Mean: {stable_mean:.1f}%)',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # PLOT 2: Error relative to BASELINE (intercept)
    # =========================================================================
    ax2 = axes[1]
    
    ax2.scatter(x[prog_mask], df.loc[prog_mask, 'delta_fvc_baseline'], 
               color='red', s=80, alpha=0.6, label='Progression (Ground Truth)', 
               edgecolors='darkred', linewidths=1.5)
    ax2.scatter(x[stable_mask], df.loc[stable_mask, 'delta_fvc_baseline'], 
               color='green', s=80, alpha=0.6, label='Stable (Ground Truth)', 
               edgecolors='darkgreen', linewidths=1.5)
    
    # Add horizontal lines
    ax2.axhline(10, color='orange', linestyle='--', linewidth=2, 
               label='10% Error Threshold', alpha=0.7)
    
    ax2.set_xlabel('Patient Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Δ FVC (%) = |Predicted FVC(52) - Baseline| / Baseline × 100', 
                   fontsize=12, fontweight='bold')
    ax2.set_title('FVC(52) Prediction Error Relative to Baseline (Intercept)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    # Statistics text
    prog_mean_base = df.loc[prog_mask, 'delta_fvc_baseline'].mean()
    stable_mean_base = df.loc[stable_mask, 'delta_fvc_baseline'].mean()
    ax2.text(0.02, 0.98, 
            f'Progression: {prog_mask.sum()} patients (Mean: {prog_mean_base:.1f}%)\n'
            f'Stable: {stable_mask.sum()} patients (Mean: {stable_mean_base:.1f}%)',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = config['plots_dir'] / 'fvc52_prediction_errors_by_patient.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot: {save_path}")
    plt.close()


def plot_error_distributions(df, config):
    """Create distribution plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    prog_mask = df['has_progression'] == True
    
    # Plot 1: Error relative to true FVC(52)
    ax1 = axes[0]
    ax1.hist(df.loc[prog_mask, 'delta_fvc_true'], bins=30, alpha=0.6, 
            color='red', label='Progression', edgecolor='darkred', linewidth=1.2)
    ax1.hist(df.loc[~prog_mask, 'delta_fvc_true'], bins=30, alpha=0.6, 
            color='green', label='Stable', edgecolor='darkgreen', linewidth=1.2)
    ax1.axvline(10, color='orange', linestyle='--', linewidth=2, label='10% Threshold')
    ax1.set_xlabel('Error (%) relative to True FVC(52)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Prediction Errors\n(Relative to True FVC@52)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Error relative to baseline
    ax2 = axes[1]
    ax2.hist(df.loc[prog_mask, 'delta_fvc_baseline'], bins=30, alpha=0.6, 
            color='red', label='Progression', edgecolor='darkred', linewidth=1.2)
    ax2.hist(df.loc[~prog_mask, 'delta_fvc_baseline'], bins=30, alpha=0.6, 
            color='green', label='Stable', edgecolor='darkgreen', linewidth=1.2)
    ax2.axvline(10, color='orange', linestyle='--', linewidth=2, label='10% Threshold')
    ax2.set_xlabel('Error (%) relative to Baseline', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Prediction Errors\n(Relative to Baseline)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = config['plots_dir'] / 'fvc52_error_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution plot: {save_path}")
    plt.close()


def print_error_statistics(df):
    """Print detailed error statistics"""
    
    print("\n" + "="*80)
    print("ERROR STATISTICS")
    print("="*80)
    
    prog_mask = df['has_progression'] == True
    
    print("\n📊 ERROR RELATIVE TO TRUE FVC(52):")
    print("-" * 60)
    print(f"  ALL PATIENTS:")
    print(f"    Mean:   {df['delta_fvc_true'].mean():.2f}%")
    print(f"    Median: {df['delta_fvc_true'].median():.2f}%")
    print(f"    Std:    {df['delta_fvc_true'].std():.2f}%")
    print(f"    Min:    {df['delta_fvc_true'].min():.2f}%")
    print(f"    Max:    {df['delta_fvc_true'].max():.2f}%")
    
    print(f"\n  PROGRESSION PATIENTS (n={prog_mask.sum()}):")
    print(f"    Mean:   {df.loc[prog_mask, 'delta_fvc_true'].mean():.2f}%")
    print(f"    Median: {df.loc[prog_mask, 'delta_fvc_true'].median():.2f}%")
    
    print(f"\n  STABLE PATIENTS (n={(~prog_mask).sum()}):")
    print(f"    Mean:   {df.loc[~prog_mask, 'delta_fvc_true'].mean():.2f}%")
    print(f"    Median: {df.loc[~prog_mask, 'delta_fvc_true'].median():.2f}%")
    
    print("\n📊 ERROR RELATIVE TO BASELINE (INTERCEPT):")
    print("-" * 60)
    print(f"  ALL PATIENTS:")
    print(f"    Mean:   {df['delta_fvc_baseline'].mean():.2f}%")
    print(f"    Median: {df['delta_fvc_baseline'].median():.2f}%")
    print(f"    Std:    {df['delta_fvc_baseline'].std():.2f}%")
    print(f"    Min:    {df['delta_fvc_baseline'].min():.2f}%")
    print(f"    Max:    {df['delta_fvc_baseline'].max():.2f}%")
    
    print(f"\n  PROGRESSION PATIENTS (n={prog_mask.sum()}):")
    print(f"    Mean:   {df.loc[prog_mask, 'delta_fvc_baseline'].mean():.2f}%")
    print(f"    Median: {df.loc[prog_mask, 'delta_fvc_baseline'].median():.2f}%")
    
    print(f"\n  STABLE PATIENTS (n={(~prog_mask).sum()}):")
    print(f"    Mean:   {df.loc[~prog_mask, 'delta_fvc_baseline'].mean():.2f}%")
    print(f"    Median: {df.loc[~prog_mask, 'delta_fvc_baseline'].median():.2f}%")
    
    # Patients with error > 10%
    high_error_true = (df['delta_fvc_true'] > 10).sum()
    high_error_base = (df['delta_fvc_baseline'] > 10).sum()
    
    print("\n" + "-" * 60)
    print(f"📌 Patients with error > 10%:")
    print(f"  Relative to True FVC(52): {high_error_true} / {len(df)} ({100*high_error_true/len(df):.1f}%)")
    print(f"  Relative to Baseline:     {high_error_base} / {len(df)} ({100*high_error_base/len(df):.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("FVC(52) PREDICTION ERROR ANALYSIS")
    print("="*80)
    
    # Load patient data
    print("\n📂 Loading patient data...")
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utilities import IPFDataLoader
    
    dl = IPFDataLoader(CONFIG['csv_path'], CONFIG['csv_features_path'], CONFIG['npy_dir'])
    patient_data, _ = dl.get_patient_data()
    
    print(f"✓ Loaded data for {len(patient_data)} patients")
    
    # Get ground truth labels
    ground_truth_labels = get_ground_truth_progression(patient_data, CONFIG['progression_threshold'])
    print(f"✓ Ground truth labels for {len(ground_truth_labels)} patients")
    
    # Load predictions and compute errors
    df = load_predictions_from_pkl(
        CONFIG['predictions_file'], 
        patient_data, 
        ground_truth_labels
    )
    
    print(f"\n✓ Total patients analyzed: {len(df)}")
    print(f"  Progression: {df['has_progression'].sum()}")
    print(f"  Stable: {(~df['has_progression']).sum()}")
    
    # Save detailed results
    results_path = CONFIG['plots_dir'] / 'fvc52_prediction_errors_detailed.csv'
    df.to_csv(results_path, index=False)
    print(f"\n✓ Saved detailed results: {results_path}")
    
    # Print statistics
    print_error_statistics(df)
    
    # Create plots
    print("\n📈 Creating plots...")
    plot_error_analysis(df, CONFIG)
    plot_error_distributions(df, CONFIG)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Plots saved to: {CONFIG['plots_dir']}")


if __name__ == "__main__":
    main()