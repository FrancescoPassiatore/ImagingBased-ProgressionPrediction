"""
Fold Variance Analysis for Ablation Study
Analyzes why some folds perform worse than others
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load kfold splits and patient features"""
    
    # Paths
    kfold_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_3fold_stratified.pkl")
    patient_features_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv")
    train_csv_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv")
    
    # Load kfold splits
    with open(kfold_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"Loaded {len(kfold_splits)} folds")
    
    # Load patient features
    patient_features_df = pd.read_csv(patient_features_path)
    
    # Load demographics
    train_df = pd.read_csv(train_csv_path)
    
    # Merge demographics
    demographics_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].copy()
    patient_features_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    # Encode Sex
    if 'Sex' in patient_features_df.columns:
        patient_features_df['Sex'] = patient_features_df['Sex'].map({'Male': 1, 'Female': 0})
    
    print(f"Loaded patient features: {patient_features_df.shape}")
    
    return kfold_splits, patient_features_df


def analyze_fold_variance(results_dir: Path, kfold_splits: dict, patient_features_df: pd.DataFrame, ablation_name: str = ""):
    """
    Analyzes why some folds perform worse than others
    
    Args:
        results_dir: Directory containing fold results
        kfold_splits: Dictionary of fold splits
        patient_features_df: DataFrame with patient features
        ablation_name: Name of the ablation configuration (for title)
    """
    
    print("\n" + "="*80)
    print(f"ANALYZING FOLD VARIANCE: {ablation_name}")
    print("="*80)
    
    fold_stats = []
    
    for fold_idx in sorted(kfold_splits.keys()):
        fold_data = kfold_splits[fold_idx]
        
        # Paths
        fold_dir = results_dir / f"fold_{fold_idx}"
        preds_path = fold_dir / "test_predictions.csv"
        checkpoint_path = fold_dir / "best_model.pth"
        
        # Check if fold completed
        if not preds_path.exists() or not checkpoint_path.exists():
            print(f"⚠️  Fold {fold_idx}: Missing results, skipping...")
            continue
        
        # Load predictions
        preds_df = pd.read_csv(preds_path)
        
        # Load checkpoint for metrics
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        except Exception as e:
            print(f"⚠️  Fold {fold_idx}: Error loading checkpoint: {e}")
            continue
        
        # Analyze test set characteristics
        test_ids = fold_data['test']
        
        # Get patient features for test set
        test_patients = patient_features_df[patient_features_df['Patient'].isin(test_ids)]
        
        # Calculate statistics
        fold_stat = {
            'fold': fold_idx,
            'test_auc': checkpoint.get('test_metrics_optimal', {}).get('auc', np.nan),
            'test_accuracy': checkpoint.get('test_metrics_optimal', {}).get('accuracy', np.nan),
            'test_f1': checkpoint.get('test_metrics_optimal', {}).get('f1', np.nan),
            'val_auc': checkpoint.get('val_auc', np.nan),
            'n_test': len(test_ids),
            'n_progressors': preds_df['true_label'].sum(),
            'progression_rate': preds_df['true_label'].mean(),
        }
        
        # Add demographic/feature statistics
        if 'Age' in test_patients.columns:
            fold_stat['mean_age'] = test_patients['Age'].mean()
            fold_stat['std_age'] = test_patients['Age'].std()
        
        if 'Sex' in test_patients.columns:
            fold_stat['pct_male'] = test_patients['Sex'].mean()
        
        if 'Avg_Tissue_30_60' in test_patients.columns:
            fold_stat['mean_tissue'] = test_patients['Avg_Tissue_30_60'].mean()
        
        if 'ApproxVol_30_60' in test_patients.columns:
            fold_stat['mean_volume'] = test_patients['ApproxVol_30_60'].mean()
        
        fold_stats.append(fold_stat)
        
        print(f"✓ Fold {fold_idx}: AUC={fold_stat['test_auc']:.3f}, n_test={fold_stat['n_test']}, prog_rate={fold_stat['progression_rate']:.2%}")
    
    if len(fold_stats) == 0:
        print("\n❌ No fold results found!")
        return None
    
    stats_df = pd.DataFrame(fold_stats)
    
    # Save statistics
    stats_df.to_csv(results_dir / "fold_statistics.csv", index=False)
    
    # Create visualization
    plot_fold_analysis(stats_df, results_dir, ablation_name)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("FOLD STATISTICS SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    
    # Identify outlier folds
    identify_outliers(stats_df)
    
    return stats_df


def plot_fold_analysis(stats_df: pd.DataFrame, results_dir: Path, ablation_name: str):
    """Create comprehensive fold analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Fold Variance Analysis: {ablation_name}', fontsize=16, fontweight='bold')
    
    # 1. AUC vs Age
    if 'mean_age' in stats_df.columns and not stats_df['mean_age'].isna().all():
        axes[0, 0].scatter(stats_df['mean_age'], stats_df['test_auc'], s=100, alpha=0.6, c='steelblue')
        axes[0, 0].set_xlabel('Mean Age (Test Set)', fontsize=11)
        axes[0, 0].set_ylabel('Test AUC', fontsize=11)
        axes[0, 0].set_title('AUC vs Age Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        for i, row in stats_df.iterrows():
            axes[0, 0].text(row['mean_age'], row['test_auc'], f"{row['fold']}", fontsize=9, ha='center')
    else:
        axes[0, 0].text(0.5, 0.5, 'Age data not available', ha='center', va='center')
        axes[0, 0].set_title('AUC vs Age Distribution')
    
    # 2. AUC vs Test Set Size
    axes[0, 1].scatter(stats_df['n_test'], stats_df['test_auc'], s=100, alpha=0.6, c='coral')
    axes[0, 1].set_xlabel('Test Set Size', fontsize=11)
    axes[0, 1].set_ylabel('Test AUC', fontsize=11)
    axes[0, 1].set_title('AUC vs Test Set Size')
    axes[0, 1].grid(True, alpha=0.3)
    for i, row in stats_df.iterrows():
        axes[0, 1].text(row['n_test'], row['test_auc'], f"{row['fold']}", fontsize=9, ha='center')
    
    # 3. AUC vs Progression Rate
    axes[0, 2].scatter(stats_df['progression_rate'], stats_df['test_auc'], s=100, alpha=0.6, c='seagreen')
    axes[0, 2].set_xlabel('Progression Rate (Test Set)', fontsize=11)
    axes[0, 2].set_ylabel('Test AUC', fontsize=11)
    axes[0, 2].set_title('AUC vs Progression Rate')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # 4. Distribution of AUC per Fold
    colors = ['red' if x < stats_df['test_auc'].mean() else 'green' for x in stats_df['test_auc']]
    axes[1, 0].bar(stats_df['fold'], stats_df['test_auc'], alpha=0.7, color=colors)
    axes[1, 0].axhline(stats_df['test_auc'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean: {stats_df['test_auc'].mean():.3f}")
    axes[1, 0].set_xlabel('Fold', fontsize=11)
    axes[1, 0].set_ylabel('Test AUC', fontsize=11)
    axes[1, 0].set_title('AUC per Fold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Correlation matrix
    corr_cols = ['test_auc']
    if 'mean_age' in stats_df.columns and not stats_df['mean_age'].isna().all():
        corr_cols.append('mean_age')
    if 'progression_rate' in stats_df.columns:
        corr_cols.append('progression_rate')
    if 'mean_tissue' in stats_df.columns and not stats_df['mean_tissue'].isna().all():
        corr_cols.append('mean_tissue')
    if 'n_test' in stats_df.columns:
        corr_cols.append('n_test')
    
    if len(corr_cols) > 1:
        corr_data = stats_df[corr_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=axes[1, 1], center=0, 
                    fmt='.2f', square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        axes[1, 1].set_title('Feature Correlations')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center')
        axes[1, 1].set_title('Feature Correlations')
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    summary_text = f"""
FOLD VARIANCE ANALYSIS

AUC Statistics:
  Mean:  {stats_df['test_auc'].mean():.4f}
  Std:   {stats_df['test_auc'].std():.4f}
  Min:   {stats_df['test_auc'].min():.4f} (Fold {stats_df.loc[stats_df['test_auc'].idxmin(), 'fold']})
  Max:   {stats_df['test_auc'].max():.4f} (Fold {stats_df.loc[stats_df['test_auc'].idxmax(), 'fold']})
  Range: {stats_df['test_auc'].max() - stats_df['test_auc'].min():.4f}

Test Set Statistics:
  Mean size: {stats_df['n_test'].mean():.1f}
  Range: {stats_df['n_test'].min()}-{stats_df['n_test'].max()}

Progression Rate:
  Mean:  {stats_df['progression_rate'].mean():.2%}
  Range: {stats_df['progression_rate'].min():.2%}-{stats_df['progression_rate'].max():.2%}
"""
    
    if 'mean_age' in stats_df.columns and not stats_df['mean_age'].isna().all():
        summary_text += f"""
Age Statistics:
  Mean:  {stats_df['mean_age'].mean():.1f} years
  Range: {stats_df['mean_age'].min():.1f}-{stats_df['mean_age'].max():.1f}
"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, fontsize=10, family='monospace', 
                    verticalalignment='top', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(results_dir / "fold_variance_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {results_dir / 'fold_variance_analysis.png'}")
    plt.close()


def identify_outliers(stats_df: pd.DataFrame):
    """Identify and report outlier folds"""
    
    mean_auc = stats_df['test_auc'].mean()
    std_auc = stats_df['test_auc'].std()
    
    # Use 1.5 standard deviations as threshold
    outlier_folds = stats_df[np.abs(stats_df['test_auc'] - mean_auc) > 1.5 * std_auc]
    
    if len(outlier_folds) > 0:
        print("\n" + "="*80)
        print("⚠️  OUTLIER FOLDS DETECTED")
        print("="*80)
        
        for _, row in outlier_folds.iterrows():
            status = "UNDERPERFORMING" if row['test_auc'] < mean_auc else "OVERPERFORMING"
            print(f"\n{status} - Fold {row['fold']}:")
            print(f"  Test AUC:         {row['test_auc']:.4f} (mean: {mean_auc:.4f}, std: {std_auc:.4f})")
            print(f"  Test Set Size:    {row['n_test']}")
            print(f"  Progression Rate: {row['progression_rate']:.2%}")
            
            if 'mean_age' in row and not pd.isna(row['mean_age']):
                print(f"  Mean Age:         {row['mean_age']:.1f} years")
            if 'pct_male' in row and not pd.isna(row['pct_male']):
                print(f"  Male %:           {row['pct_male']:.1%}")
            if 'mean_tissue' in row and not pd.isna(row['mean_tissue']):
                print(f"  Mean Tissue:      {row['mean_tissue']:.2f}")
    else:
        print("\n✓ No significant outlier folds detected (within 1.5 std)")


def main():
    """Main analysis function"""
    
    print("="*80)
    print("FOLD VARIANCE ANALYSIS - ABLATION STUDY")
    print("="*80)
    
    # Load data
    kfold_splits, patient_features_df = load_data()
    
    # Base results directory
    base_results_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\ablation_study_best_config_stratified")
    
    # Analyze each ablation configuration
    ablation_configs = ['cnn_only', 'cnn_hand', 'cnn_demo', 'full']
    
    all_stats = {}
    
    for config_name in ablation_configs:
        config_dir = base_results_dir / f"ablation_{config_name}"
        
        if not config_dir.exists():
            print(f"\n⚠️  Skipping {config_name}: directory not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {config_name}")
        print(f"{'='*80}")
        
        stats_df = analyze_fold_variance(
            results_dir=config_dir,
            kfold_splits=kfold_splits,
            patient_features_df=patient_features_df,
            ablation_name=config_name.upper().replace('_', ' ')
        )
        
        if stats_df is not None:
            all_stats[config_name] = stats_df
    
    # Create comparison across configurations
    if len(all_stats) > 1:
        create_config_comparison(all_stats, base_results_dir)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


def create_config_comparison(all_stats: dict, results_dir: Path):
    """Create comparison plot across different ablation configurations"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Configuration Fold Variance Comparison', fontsize=14, fontweight='bold')
    
    # 1. Mean AUC per configuration
    config_means = {name: df['test_auc'].mean() for name, df in all_stats.items()}
    config_stds = {name: df['test_auc'].std() for name, df in all_stats.items()}
    
    configs = list(config_means.keys())
    means = list(config_means.values())
    stds = list(config_stds.values())
    
    axes[0].bar(configs, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue')
    axes[0].set_ylabel('Mean Test AUC', fontsize=11)
    axes[0].set_xlabel('Configuration', fontsize=11)
    axes[0].set_title('Mean AUC ± Std Across Configurations')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    
    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.01, f'{m:.3f}±{s:.3f}', ha='center', fontsize=9)
    
    # 2. AUC distribution per configuration (box plot)
    data_for_box = [df['test_auc'].values for df in all_stats.values()]
    bp = axes[1].boxplot(data_for_box, labels=configs, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Test AUC', fontsize=11)
    axes[1].set_xlabel('Configuration', fontsize=11)
    axes[1].set_title('AUC Distribution Across Folds')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(configs, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(results_dir / "config_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {results_dir / 'config_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    main()
