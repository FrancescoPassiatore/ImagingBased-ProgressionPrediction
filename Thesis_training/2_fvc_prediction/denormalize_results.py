"""
Script to denormalize FVC prediction results from normalized scale back to mL
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

def load_fvc_statistics(gt_path: Path, kfold_splits_path: Path):
    """
    Load FVC statistics (mean, std) from the ground truth data
    Returns dict with stats for each fold
    """
    # Load ground truth
    gt_df = pd.read_csv(gt_path)
    
    # Load K-fold splits
    with open(kfold_splits_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print("\n" + "="*80)
    print("LOADING FVC STATISTICS FOR DENORMALIZATION")
    print("="*80)
    
    fold_stats = {}
    
    for fold_idx in kfold_splits:
        fold_data = kfold_splits[fold_idx]
        train_patients = fold_data['train']
        
        # Get training set FVC values
        train_df = gt_df[gt_df['PatientID'].isin(train_patients)].copy()
        
        # Compute statistics (same as during training)
        baseline_mean = train_df['BaselineFVC'].mean()
        baseline_std = train_df['BaselineFVC'].std()
        fvc52_mean = train_df['Week52FVC'].mean()
        fvc52_std = train_df['Week52FVC'].std()
        
        # For StandardScaler with 2 columns, it computes separate mean/std for each
        # We need the combined statistics as they were used together
        fvc_values = train_df[['BaselineFVC', 'Week52FVC']].values
        scaler = StandardScaler()
        scaler.fit(fvc_values)
        
        # scaler.mean_[0] = baseline mean, scaler.mean_[1] = fvc52 mean
        # scaler.scale_[0] = baseline std, scaler.scale_[1] = fvc52 std
        
        fold_stats[fold_idx] = {
            'baseline_mean': scaler.mean_[0],
            'baseline_std': scaler.scale_[0],
            'fvc52_mean': scaler.mean_[1],
            'fvc52_std': scaler.scale_[1],
            'n_train_patients': len(train_patients)
        }
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train patients: {len(train_patients)}")
        print(f"  Baseline FVC: mean={scaler.mean_[0]:.2f} mL, std={scaler.scale_[0]:.2f} mL")
        print(f"  Week52 FVC:   mean={scaler.mean_[1]:.2f} mL, std={scaler.scale_[1]:.2f} mL")
    
    return fold_stats


def denormalize_metrics(results_df: pd.DataFrame, fold_stats: dict) -> pd.DataFrame:
    """
    Denormalize MAE and RMSE from normalized scale to mL
    
    For normalized predictions:
    - MAE_normalized = mean(|y_pred_norm - y_true_norm|)
    - MAE_denormalized = MAE_normalized * std_fvc52
    
    Same for RMSE
    R2 is unitless, so it stays the same
    """
    denorm_df = results_df.copy()
    
    print("\n" + "="*80)
    print("DENORMALIZING METRICS")
    print("="*80)
    
    # Add denormalized columns
    denorm_df['val_mae_mL'] = 0.0
    denorm_df['test_mae_mL'] = 0.0
    denorm_df['test_rmse_mL'] = 0.0
    
    for idx, row in denorm_df.iterrows():
        fold = int(row['fold'])
        stats = fold_stats[fold]
        
        # Denormalize: metric_mL = metric_normalized * std_fvc52
        val_mae_mL = row['val_mae'] * stats['fvc52_std']
        test_mae_mL = row['test_mae'] * stats['fvc52_std']
        test_rmse_mL = row['test_rmse'] * stats['fvc52_std']
        
        denorm_df.at[idx, 'val_mae_mL'] = val_mae_mL
        denorm_df.at[idx, 'test_mae_mL'] = test_mae_mL
        denorm_df.at[idx, 'test_rmse_mL'] = test_rmse_mL
        
        print(f"\nFold {fold}:")
        print(f"  FVC52 std: {stats['fvc52_std']:.2f} mL")
        print(f"  Val MAE:   {row['val_mae']:.4f} (norm) → {val_mae_mL:.2f} mL")
        print(f"  Test MAE:  {row['test_mae']:.4f} (norm) → {test_mae_mL:.2f} mL")
        print(f"  Test RMSE: {row['test_rmse']:.4f} (norm) → {test_rmse_mL:.2f} mL")
        print(f"  Test R²:   {row['test_r2']:.4f} (unitless, unchanged)")
    
    # Reorder columns for clarity
    cols = ['fold', 
            'val_mae', 'val_mae_mL',
            'test_mae', 'test_mae_mL', 
            'test_rmse', 'test_rmse_mL',
            'test_r2']
    denorm_df = denorm_df[cols]
    
    return denorm_df


def denormalize_summary(summary_df: pd.DataFrame, fold_stats: dict) -> pd.DataFrame:
    """
    Denormalize aggregate summary metrics
    
    For mean values: denormalized_mean = normalized_mean × avg_std_across_folds
    For std values: denormalized_std = normalized_std × avg_std_across_folds
    """
    denorm_summary = summary_df.copy()
    
    # Compute average FVC52 std across all folds
    avg_fvc52_std = np.mean([stats['fvc52_std'] for stats in fold_stats.values()])
    
    print(f"\n  Average FVC52 std across folds: {avg_fvc52_std:.2f} mL")
    print(f"\n  Denormalizing summary metrics...")
    
    # Add denormalized rows
    rows_to_add = []
    
    for idx, row in denorm_summary.iterrows():
        metric = row['Metric']
        
        if 'MAE' in metric or 'RMSE' in metric:
            # Create denormalized version
            denorm_metric = metric.replace('MAE', 'MAE (mL)').replace('RMSE', 'RMSE (mL)')
            denorm_mean = row['Mean'] * avg_fvc52_std
            denorm_std = row['Std'] * avg_fvc52_std
            
            rows_to_add.append({
                'Metric': denorm_metric,
                'Mean': denorm_mean,
                'Std': denorm_std
            })
            
            print(f"    {metric:15s}: {row['Mean']:.4f} ± {row['Std']:.4f} (norm) → {denorm_mean:.2f} ± {denorm_std:.2f} mL")
    
    # Add denormalized rows to dataframe
    denorm_summary = pd.concat([denorm_summary, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return denorm_summary


def denormalize_ablation_comparison(ablation_dir: Path, fold_stats: dict):
    """
    Denormalize the final ablation_comparison.csv file and add std values
    """
    comparison_file = ablation_dir / "ablation_comparison.csv"
    
    if not comparison_file.exists():
        print(f"\n  ⚠️  ablation_comparison.csv not found")
        return
    
    print("\n" + "="*80)
    print("DENORMALIZING ABLATION COMPARISON WITH STD")
    print("="*80)
    
    # Compute average FVC52 std across all folds
    avg_fvc52_std = np.mean([stats['fvc52_std'] for stats in fold_stats.values()])
    print(f"  Average FVC52 std across folds: {avg_fvc52_std:.2f} mL")
    
    # Load comparison
    comparison_df = pd.read_csv(comparison_file)
    print(f"\n  Loaded {len(comparison_df)} configurations")
    
    # For each configuration, load detailed results to compute std
    enhanced_rows = []
    
    for _, row in comparison_df.iterrows():
        config_name = row['Configuration']
        config_dir = ablation_dir / f"ablation_{config_name}"
        detailed_file = config_dir / "detailed_fold_results.csv"
        
        if detailed_file.exists():
            # Load detailed results for this config
            detailed_df = pd.read_csv(detailed_file)
            
            # Compute std from fold results
            val_mae_std_norm = detailed_df['val_mae'].std()
            test_mae_std_norm = detailed_df['test_mae'].std()
            test_rmse_std_norm = detailed_df['test_rmse'].std()
            test_r2_std = detailed_df['test_r2'].std()
            
            # Denormalize mean values
            val_mae_mL = row['Val_MAE'] * avg_fvc52_std
            test_mae_mL = row['Test_MAE'] * avg_fvc52_std
            test_rmse_mL = row['Test_RMSE'] * avg_fvc52_std
            
            # Denormalize std values
            val_mae_std_mL = val_mae_std_norm * avg_fvc52_std
            test_mae_std_mL = test_mae_std_norm * avg_fvc52_std
            test_rmse_std_mL = test_rmse_std_norm * avg_fvc52_std
            
            enhanced_rows.append({
                'Configuration': config_name,
                'Description': row['Description'],
                'Val_MAE_mL': f"{val_mae_mL:.2f} ± {val_mae_std_mL:.2f}",
                'Test_MAE_mL': f"{test_mae_mL:.2f} ± {test_mae_std_mL:.2f}",
                'Test_RMSE_mL': f"{test_rmse_mL:.2f} ± {test_rmse_std_mL:.2f}",
                'Test_R2': f"{row['Test_R2']:.4f} ± {test_r2_std:.4f}",
                # Keep normalized values too
                'Val_MAE_norm': f"{row['Val_MAE']:.4f} ± {val_mae_std_norm:.4f}",
                'Test_MAE_norm': f"{row['Test_MAE']:.4f} ± {test_mae_std_norm:.4f}",
                'Test_RMSE_norm': f"{row['Test_RMSE']:.4f} ± {test_rmse_std_norm:.4f}"
            })
        else:
            print(f"  ⚠️  No detailed results for {config_name}")
    
    # Create enhanced dataframe
    enhanced_df = pd.DataFrame(enhanced_rows)
    
    # Save
    output_file = ablation_dir / "ablation_comparison_denormalized.csv"
    enhanced_df.to_csv(output_file, index=False)
    
    print(f"\n  ✓ Saved: {output_file.name}")
    print(f"\n  DENORMALIZED COMPARISON (Mean ± Std):")
    print(f"  {'Configuration':<20s} {'Val MAE (mL)':<20s} {'Test MAE (mL)':<20s} {'Test RMSE (mL)':<20s} {'Test R²':<20s}")
    print("  " + "-"*100)
    
    for _, row in enhanced_df.iterrows():
        print(f"  {row['Configuration']:<20s} {row['Val_MAE_mL']:<20s} {row['Test_MAE_mL']:<20s} {row['Test_RMSE_mL']:<20s} {row['Test_R2']:<20s}")


def denormalize_ablation_results(ablation_dir: Path, gt_path: Path, kfold_splits_path: Path):
    """
    Denormalize all ablation study results
    """
    print("\n" + "="*80)
    print("DENORMALIZING ABLATION STUDY RESULTS")
    print("="*80)
    print(f"Ablation directory: {ablation_dir}")
    
    # Load FVC statistics
    fold_stats = load_fvc_statistics(gt_path, kfold_splits_path)
    
    # Find all ablation result directories
    ablation_configs = [d for d in ablation_dir.iterdir() if d.is_dir() and d.name.startswith('ablation_')]
    
    print(f"\nFound {len(ablation_configs)} ablation configurations")
    
    for config_dir in ablation_configs:
        config_name = config_dir.name.replace('ablation_', '')
        print(f"\n{'='*80}")
        print(f"PROCESSING: {config_name}")
        print("="*80)
        
        # Check if detailed_fold_results.csv exists
        detailed_file = config_dir / "detailed_fold_results.csv"
        if not detailed_file.exists():
            print(f"  ⚠️  Skipping {config_name}: no detailed_fold_results.csv found")
            continue
        
        # Load results
        results_df = pd.read_csv(detailed_file)
        print(f"\n  Loaded results: {len(results_df)} folds")
        
        # Denormalize detailed results
        denorm_df = denormalize_metrics(results_df, fold_stats)
        
        # Save denormalized detailed results
        output_file = config_dir / "detailed_fold_results_denormalized.csv"
        denorm_df.to_csv(output_file, index=False)
        print(f"\n  ✓ Saved: {output_file.name}")
        
        # Denormalize aggregate summary
        summary_file = config_dir / "aggregate_metrics_summary.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            denorm_summary = denormalize_summary(summary_df, fold_stats)
            
            # Save denormalized summary
            output_summary = config_dir / "aggregate_metrics_summary_denormalized.csv"
            denorm_summary.to_csv(output_summary, index=False)
            print(f"  ✓ Saved: {output_summary.name}")
        else:
            print(f"  ⚠️  No aggregate_metrics_summary.csv found")
        
        # Print summary statistics
        print(f"\n  SUMMARY (in mL):")
        print(f"    Val MAE:   {denorm_df['val_mae_mL'].mean():.2f} ± {denorm_df['val_mae_mL'].std():.2f} mL")
        print(f"    Test MAE:  {denorm_df['test_mae_mL'].mean():.2f} ± {denorm_df['test_mae_mL'].std():.2f} mL")
        print(f"    Test RMSE: {denorm_df['test_rmse_mL'].mean():.2f} ± {denorm_df['test_rmse_mL'].std():.2f} mL")
        print(f"    Test R²:   {denorm_df['test_r2'].mean():.4f} ± {denorm_df['test_r2'].std():.4f}")
    
    # Denormalize final ablation comparison
    denormalize_ablation_comparison(ablation_dir, fold_stats)


def main():
    """Main execution"""
    
    # Paths
    base_dir = Path(r"d:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\2_fvc_prediction")
    gt_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv")
    kfold_splits_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl")
    ablation_dir = base_dir / "ablation_study_results_stratified"
    
    # Check if directories exist
    if not ablation_dir.exists():
        print(f"❌ Ablation results directory not found: {ablation_dir}")
        return
    
    if not gt_path.exists():
        print(f"❌ Ground truth file not found: {gt_path}")
        return
    
    if not kfold_splits_path.exists():
        print(f"❌ K-fold splits file not found: {kfold_splits_path}")
        return
    
    # Denormalize all results
    denormalize_ablation_results(ablation_dir, gt_path, kfold_splits_path)
    
    print("\n" + "="*80)
    print("DENORMALIZATION COMPLETE!")
    print("="*80)
    print("\nNew files created:")
    print("  - detailed_fold_results_denormalized.csv (per-fold metrics)")
    print("  - aggregate_metrics_summary_denormalized.csv (mean ± std across folds)")
    print("  - ablation_comparison_denormalized.csv (final comparison in mL)")
    print("\nThese files contain:")
    print("  - Original normalized metrics (val_mae, test_mae, test_rmse)")
    print("  - Denormalized metrics in mL (val_mae_mL, test_mae_mL, test_rmse_mL)")
    print("  - R² score (unchanged, unitless)")


if __name__ == "__main__":
    main()
