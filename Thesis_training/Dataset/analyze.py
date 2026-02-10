"""
Comprehensive Stratification Verification for K-Fold Cross-Validation
======================================================================

This script verifies that your K-fold splits are properly stratified across:
1. Target label (progression)
2. Age distribution
3. Sex distribution
4. Smoking status distribution
5. Disease severity metrics (tissue features)

It generates detailed reports and visualizations to identify any imbalances.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_data(kfold_path: Path, patient_features_path: Path, train_csv_path: Path):
    """Load k-fold splits and patient features"""
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load k-fold splits
    with open(kfold_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"✓ Loaded {len(kfold_splits)} folds from: {kfold_path.name}")
    
    # Load patient features
    patient_features = pd.read_csv(patient_features_path)
    print(f"✓ Loaded patient features: {patient_features.shape}")
    
    # Load demographics from train.csv if available
    if train_csv_path.exists():
        train_df = pd.read_csv(train_csv_path)
        
        # Merge demographics
        demo_cols = ['Patient']
        for col in ['Age', 'Sex', 'SmokingStatus']:
            if col in train_df.columns:
                demo_cols.append(col)
        
        demographics = train_df[demo_cols].copy()
        patient_features = patient_features.merge(demographics, on='Patient', how='left')
        print(f"✓ Merged demographics: {demo_cols[1:]}")
    
    return kfold_splits, patient_features


def compute_fold_statistics(kfold_splits: dict, patient_features: pd.DataFrame):
    """
    Compute detailed statistics for each fold
    """
    
    print("\n" + "="*70)
    print("COMPUTING FOLD STATISTICS")
    print("="*70)
    
    fold_stats = []
    
    for fold_idx, fold_data in kfold_splits.items():
        for split_name in ['train', 'val', 'test']:
            patient_ids = fold_data[split_name]
            
            # Filter patient features for this split
            split_patients = patient_features[patient_features['Patient'].isin(patient_ids)]
            
            if len(split_patients) == 0:
                continue
            
            # Basic stats
            stats_dict = {
                'fold': fold_idx,
                'split': split_name,
                'n_patients': len(patient_ids),
            }
            
            # Progression rate
            if 'has_progressed' in split_patients.columns:
                stats_dict['n_progressors'] = split_patients['has_progressed'].sum()
                stats_dict['progression_rate'] = split_patients['has_progressed'].mean()
            
            # Age statistics
            if 'Age' in split_patients.columns:
                stats_dict['age_mean'] = split_patients['Age'].mean()
                stats_dict['age_std'] = split_patients['Age'].std()
                stats_dict['age_min'] = split_patients['Age'].min()
                stats_dict['age_max'] = split_patients['Age'].max()
            
            # Sex distribution
            if 'Sex' in split_patients.columns:
                # Encode if string
                if split_patients['Sex'].dtype == 'object':
                    sex_encoded = split_patients['Sex'].map({'Male': 1, 'Female': 0})
                else:
                    sex_encoded = split_patients['Sex']
                stats_dict['pct_male'] = sex_encoded.mean()
            
            # Smoking status distribution
            if 'SmokingStatus' in split_patients.columns:
                if split_patients['SmokingStatus'].dtype == 'object':
                    smoking_encoded = split_patients['SmokingStatus'].map({
                        'Never smoked': 0,
                        'Ex-smoker': 1,
                        'Currently smokes': 2
                    })
                else:
                    smoking_encoded = split_patients['SmokingStatus']
                
                stats_dict['pct_never_smoked'] = (smoking_encoded == 0).mean()
                stats_dict['pct_ex_smoker'] = (smoking_encoded == 1).mean()
                stats_dict['pct_current_smoker'] = (smoking_encoded == 2).mean()
            
            # Disease severity features
            if 'Avg_Tissue_30_60' in split_patients.columns:
                stats_dict['tissue_mean'] = split_patients['Avg_Tissue_30_60'].mean()
                stats_dict['tissue_std'] = split_patients['Avg_Tissue_30_60'].std()
            
            if 'ApproxVol_30_60' in split_patients.columns:
                stats_dict['volume_mean'] = split_patients['ApproxVol_30_60'].mean()
                stats_dict['volume_std'] = split_patients['ApproxVol_30_60'].std()
            
            fold_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(fold_stats)
    return stats_df


def test_stratification_quality(stats_df: pd.DataFrame):
    """
    Statistical tests to verify stratification quality
    """
    
    print("\n" + "="*70)
    print("STRATIFICATION QUALITY TESTS")
    print("="*70)
    
    results = {}
    
    # Test only on TEST splits (most important)
    test_stats = stats_df[stats_df['split'] == 'test'].copy()
    
    if len(test_stats) == 0:
        print("⚠️ No test splits found!")
        return results
    
    print(f"\nTesting stratification across {len(test_stats)} test folds")
    print("-" * 70)
    
    # 1. Progression rate balance
    if 'progression_rate' in test_stats.columns:
        prog_rates = test_stats['progression_rate'].values
        
        # Chi-square test for equal proportions
        observed = test_stats['n_progressors'].values
        total = test_stats['n_patients'].values
        expected_rate = observed.sum() / total.sum()
        expected = total * expected_rate
        
        chi2_stat = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, len(observed) - 1)
        
        results['progression_balance'] = {
            'metric': 'Progression Rate',
            'mean': prog_rates.mean(),
            'std': prog_rates.std(),
            'range': (prog_rates.min(), prog_rates.max()),
            'cv': prog_rates.std() / prog_rates.mean() if prog_rates.mean() > 0 else 0,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'balanced': p_value > 0.05  # Not significantly different
        }
        
        print(f"\n1. PROGRESSION RATE")
        print(f"   Mean: {prog_rates.mean():.3f} ± {prog_rates.std():.3f}")
        print(f"   Range: [{prog_rates.min():.3f}, {prog_rates.max():.3f}]")
        print(f"   CV: {results['progression_balance']['cv']:.2%}")
        print(f"   Chi-square test: χ² = {chi2_stat:.3f}, p = {p_value:.3f}")
        
        if p_value > 0.05:
            print(f"   ✅ WELL BALANCED (p > 0.05)")
        else:
            print(f"   ⚠️ IMBALANCED (p < 0.05)")
    
    # 2. Age distribution balance
    if 'age_mean' in test_stats.columns:
        age_means = test_stats['age_mean'].values
        
        # ANOVA test
        # We can't do full ANOVA without individual patient data,
        # but we can check if means are similar
        age_cv = test_stats['age_mean'].std() / test_stats['age_mean'].mean()
        
        results['age_balance'] = {
            'metric': 'Age (mean)',
            'mean': age_means.mean(),
            'std': age_means.std(),
            'range': (age_means.min(), age_means.max()),
            'cv': age_cv,
            'balanced': age_cv < 0.05  # CV < 5% is good
        }
        
        print(f"\n2. AGE DISTRIBUTION")
        print(f"   Mean across folds: {age_means.mean():.1f} ± {age_means.std():.1f}")
        print(f"   Range: [{age_means.min():.1f}, {age_means.max():.1f}]")
        print(f"   CV: {age_cv:.2%}")
        
        if age_cv < 0.05:
            print(f"   ✅ WELL BALANCED (CV < 5%)")
        elif age_cv < 0.10:
            print(f"   ⚠️ ACCEPTABLE (CV < 10%)")
        else:
            print(f"   ❌ IMBALANCED (CV > 10%)")
    
    # 3. Sex distribution balance
    if 'pct_male' in test_stats.columns:
        pct_males = test_stats['pct_male'].values
        
        sex_cv = pct_males.std() / pct_males.mean() if pct_males.mean() > 0 else 0
        
        results['sex_balance'] = {
            'metric': 'Sex (% Male)',
            'mean': pct_males.mean(),
            'std': pct_males.std(),
            'range': (pct_males.min(), pct_males.max()),
            'cv': sex_cv,
            'balanced': sex_cv < 0.10
        }
        
        print(f"\n3. SEX DISTRIBUTION")
        print(f"   Mean % Male: {pct_males.mean():.1%} ± {pct_males.std():.1%}")
        print(f"   Range: [{pct_males.min():.1%}, {pct_males.max():.1%}]")
        print(f"   CV: {sex_cv:.2%}")
        
        if sex_cv < 0.10:
            print(f"   ✅ WELL BALANCED (CV < 10%)")
        else:
            print(f"   ⚠️ IMBALANCED (CV > 10%)")
    
    # 4. Smoking status balance
    if 'pct_never_smoked' in test_stats.columns:
        never_cv = test_stats['pct_never_smoked'].std() / test_stats['pct_never_smoked'].mean()
        
        results['smoking_balance'] = {
            'metric': 'Smoking Status',
            'never_mean': test_stats['pct_never_smoked'].mean(),
            'ex_mean': test_stats['pct_ex_smoker'].mean(),
            'current_mean': test_stats['pct_current_smoker'].mean(),
            'cv': never_cv,
            'balanced': never_cv < 0.15
        }
        
        print(f"\n4. SMOKING STATUS DISTRIBUTION")
        print(f"   Never smoked: {test_stats['pct_never_smoked'].mean():.1%}")
        print(f"   Ex-smoker: {test_stats['pct_ex_smoker'].mean():.1%}")
        print(f"   Current: {test_stats['pct_current_smoker'].mean():.1%}")
        print(f"   CV (never): {never_cv:.2%}")
        
        if never_cv < 0.15:
            print(f"   ✅ ACCEPTABLE BALANCE")
        else:
            print(f"   ⚠️ HIGH VARIANCE")
    
    # 5. Disease severity balance (tissue)
    if 'tissue_mean' in test_stats.columns:
        tissue_means = test_stats['tissue_mean'].values
        tissue_cv = tissue_means.std() / tissue_means.mean()
        
        results['severity_balance'] = {
            'metric': 'Disease Severity (tissue)',
            'mean': tissue_means.mean(),
            'std': tissue_means.std(),
            'cv': tissue_cv,
            'balanced': tissue_cv < 0.10
        }
        
        print(f"\n5. DISEASE SEVERITY (Tissue)")
        print(f"   Mean tissue: {tissue_means.mean():.1f} ± {tissue_means.std():.1f}")
        print(f"   CV: {tissue_cv:.2%}")
        
        if tissue_cv < 0.10:
            print(f"   ✅ WELL BALANCED")
        else:
            print(f"   ⚠️ VARIANCE DETECTED (CV > 10%)")
    
    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL STRATIFICATION QUALITY")
    print("="*70)
    
    balanced_count = sum(1 for r in results.values() if r.get('balanced', False))
    total_tests = len(results)
    
    print(f"\nTests passed: {balanced_count}/{total_tests}")
    
    if balanced_count == total_tests:
        print("✅ EXCELLENT STRATIFICATION - All metrics well balanced")
    elif balanced_count >= total_tests * 0.8:
        print("✅ GOOD STRATIFICATION - Most metrics balanced")
    elif balanced_count >= total_tests * 0.6:
        print("⚠️ ACCEPTABLE STRATIFICATION - Some imbalances present")
    else:
        print("❌ POOR STRATIFICATION - Significant imbalances detected")
    
    return results


def visualize_stratification(stats_df: pd.DataFrame, save_dir: Path):
    """
    Create comprehensive visualizations of stratification
    """
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for test splits
    test_stats = stats_df[stats_df['split'] == 'test'].copy()
    
    if len(test_stats) == 0:
        print("⚠️ No test splits to visualize")
        return
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Progression rate across folds
    ax1 = fig.add_subplot(gs[0, 0])
    if 'progression_rate' in test_stats.columns:
        folds = test_stats['fold'].values
        prog_rates = test_stats['progression_rate'].values
        
        ax1.bar(folds, prog_rates, alpha=0.7, color='steelblue')
        ax1.axhline(prog_rates.mean(), color='red', linestyle='--', 
                   label=f'Mean: {prog_rates.mean():.2%}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Progression Rate')
        ax1.set_title('Progression Rate Balance', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(prog_rates) * 1.2)
    
    # 2. Sample size distribution
    ax2 = fig.add_subplot(gs[0, 1])
    n_patients = test_stats['n_patients'].values
    n_progressors = test_stats['n_progressors'].values
    n_non_prog = n_patients - n_progressors
    
    x = np.arange(len(folds))
    ax2.bar(x, n_non_prog, label='Non-progressors', alpha=0.7, color='lightblue')
    ax2.bar(x, n_progressors, bottom=n_non_prog, label='Progressors', 
           alpha=0.7, color='coral')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Sample Size Distribution', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Age distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'age_mean' in test_stats.columns:
        age_means = test_stats['age_mean'].values
        age_stds = test_stats['age_std'].values
        
        ax3.bar(folds, age_means, alpha=0.7, color='seagreen', 
               yerr=age_stds, capsize=5)
        ax3.axhline(age_means.mean(), color='red', linestyle='--',
                   label=f'Mean: {age_means.mean():.1f}')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Mean Age (years)')
        ax3.set_title('Age Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Sex distribution
    ax4 = fig.add_subplot(gs[0, 3])
    if 'pct_male' in test_stats.columns:
        pct_males = test_stats['pct_male'].values * 100
        
        ax4.bar(folds, pct_males, alpha=0.7, color='orchid')
        ax4.axhline(pct_males.mean(), color='red', linestyle='--',
                   label=f'Mean: {pct_males.mean():.1f}%')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('% Male')
        ax4.set_title('Sex Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 100)
    
    # 5. Smoking status stacked bar
    ax5 = fig.add_subplot(gs[1, 0])
    if 'pct_never_smoked' in test_stats.columns:
        never = test_stats['pct_never_smoked'].values * 100
        ex = test_stats['pct_ex_smoker'].values * 100
        current = test_stats['pct_current_smoker'].values * 100
        
        ax5.bar(folds, never, label='Never smoked', alpha=0.7, color='lightgreen')
        ax5.bar(folds, ex, bottom=never, label='Ex-smoker', alpha=0.7, color='gold')
        ax5.bar(folds, current, bottom=never+ex, label='Current', alpha=0.7, color='salmon')
        ax5.set_xlabel('Fold')
        ax5.set_ylabel('Percentage (%)')
        ax5.set_title('Smoking Status Distribution', fontweight='bold')
        ax5.legend()
        ax5.set_ylim(0, 100)
    
    # 6. Disease severity (tissue)
    ax6 = fig.add_subplot(gs[1, 1])
    if 'tissue_mean' in test_stats.columns:
        tissue_means = test_stats['tissue_mean'].values
        tissue_stds = test_stats['tissue_std'].values
        
        ax6.bar(folds, tissue_means, alpha=0.7, color='teal',
               yerr=tissue_stds, capsize=5)
        ax6.axhline(tissue_means.mean(), color='red', linestyle='--',
                   label=f'Mean: {tissue_means.mean():.1f}')
        ax6.set_xlabel('Fold')
        ax6.set_ylabel('Mean Tissue')
        ax6.set_title('Disease Severity (Tissue)', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Volume distribution
    ax7 = fig.add_subplot(gs[1, 2])
    if 'volume_mean' in test_stats.columns:
        volume_means = test_stats['volume_mean'].values
        
        ax7.bar(folds, volume_means, alpha=0.7, color='skyblue')
        ax7.axhline(volume_means.mean(), color='red', linestyle='--',
                   label=f'Mean: {volume_means.mean():.0f}')
        ax7.set_xlabel('Fold')
        ax7.set_ylabel('Mean Volume')
        ax7.set_title('Lung Volume Distribution', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Coefficient of Variation summary
    ax8 = fig.add_subplot(gs[1, 3])
    cv_metrics = []
    cv_values = []
    cv_colors = []
    
    if 'progression_rate' in test_stats.columns:
        cv = test_stats['progression_rate'].std() / test_stats['progression_rate'].mean()
        cv_metrics.append('Progression\nRate')
        cv_values.append(cv * 100)
        cv_colors.append('green' if cv < 0.10 else 'orange' if cv < 0.15 else 'red')
    
    if 'age_mean' in test_stats.columns:
        cv = test_stats['age_mean'].std() / test_stats['age_mean'].mean()
        cv_metrics.append('Age')
        cv_values.append(cv * 100)
        cv_colors.append('green' if cv < 0.05 else 'orange' if cv < 0.10 else 'red')
    
    if 'pct_male' in test_stats.columns:
        cv = test_stats['pct_male'].std() / test_stats['pct_male'].mean()
        cv_metrics.append('Sex')
        cv_values.append(cv * 100)
        cv_colors.append('green' if cv < 0.10 else 'orange' if cv < 0.15 else 'red')
    
    if 'tissue_mean' in test_stats.columns:
        cv = test_stats['tissue_mean'].std() / test_stats['tissue_mean'].mean()
        cv_metrics.append('Tissue')
        cv_values.append(cv * 100)
        cv_colors.append('green' if cv < 0.10 else 'orange' if cv < 0.15 else 'red')
    
    ax8.barh(cv_metrics, cv_values, color=cv_colors, alpha=0.7)
    ax8.axvline(5, color='green', linestyle='--', alpha=0.5, label='Excellent (<5%)')
    ax8.axvline(10, color='orange', linestyle='--', alpha=0.5, label='Good (<10%)')
    ax8.axvline(15, color='red', linestyle='--', alpha=0.5, label='Acceptable (<15%)')
    ax8.set_xlabel('Coefficient of Variation (%)')
    ax8.set_title('Balance Quality (CV)', fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='x')
    
    # 9-12. Distribution comparisons (box plots)
    # Age box plot
    ax9 = fig.add_subplot(gs[2, 0])
    if 'age_mean' in test_stats.columns:
        ax9.boxplot([test_stats['age_mean'].values], labels=['Age'])
        ax9.set_ylabel('Years')
        ax9.set_title('Age Distribution Across Folds', fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
    
    # Progression rate box plot
    ax10 = fig.add_subplot(gs[2, 1])
    if 'progression_rate' in test_stats.columns:
        ax10.boxplot([test_stats['progression_rate'].values * 100], 
                     labels=['Progression\nRate'])
        ax10.set_ylabel('Percentage (%)')
        ax10.set_title('Progression Rate Distribution', fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
    
    # Sample size box plot
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.boxplot([test_stats['n_patients'].values], labels=['Sample\nSize'])
    ax11.set_ylabel('Number of Patients')
    ax11.set_title('Sample Size Distribution', fontweight='bold')
    ax11.grid(True, alpha=0.3, axis='y')
    
    # Summary text
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    summary_text = "STRATIFICATION SUMMARY\n\n"
    summary_text += f"Number of folds: {len(test_stats)}\n\n"
    
    if 'progression_rate' in test_stats.columns:
        prog_mean = test_stats['progression_rate'].mean()
        prog_std = test_stats['progression_rate'].std()
        prog_cv = prog_std / prog_mean if prog_mean > 0 else 0
        summary_text += f"Progression Rate:\n"
        summary_text += f"  {prog_mean:.1%} ± {prog_std:.1%}\n"
        summary_text += f"  CV: {prog_cv:.1%}\n\n"
    
    if 'age_mean' in test_stats.columns:
        age_mean_of_means = test_stats['age_mean'].mean()
        age_std_of_means = test_stats['age_mean'].std()
        age_cv = age_std_of_means / age_mean_of_means
        summary_text += f"Age:\n"
        summary_text += f"  {age_mean_of_means:.1f} ± {age_std_of_means:.1f} years\n"
        summary_text += f"  CV: {age_cv:.1%}\n\n"
    
    summary_text += f"Sample Size:\n"
    summary_text += f"  {test_stats['n_patients'].mean():.1f} ± {test_stats['n_patients'].std():.1f}\n"
    summary_text += f"  Range: {test_stats['n_patients'].min()}-{test_stats['n_patients'].max()}\n"
    
    ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('K-Fold Stratification Quality Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = save_dir / 'stratification_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization: {save_path}")
    
    return save_path


def generate_detailed_report(stats_df: pd.DataFrame, test_results: dict, save_dir: Path):
    """
    Generate detailed text report
    """
    
    report_path = save_dir / 'stratification_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("K-FOLD CROSS-VALIDATION STRATIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall summary
        test_stats = stats_df[stats_df['split'] == 'test']
        
        f.write(f"Number of folds: {len(test_stats)}\n")
        f.write(f"Total patients: {test_stats['n_patients'].sum()}\n")
        f.write(f"Average test set size: {test_stats['n_patients'].mean():.1f} ± {test_stats['n_patients'].std():.1f}\n\n")
        
        # Per-fold details
        f.write("="*70 + "\n")
        f.write("FOLD-BY-FOLD BREAKDOWN\n")
        f.write("="*70 + "\n\n")
        
        for _, row in test_stats.iterrows():
            f.write(f"Fold {row['fold']}:\n")
            f.write(f"  Patients: {row['n_patients']}\n")
            
            if 'n_progressors' in row:
                f.write(f"  Progressors: {row['n_progressors']:.0f} ({row['progression_rate']:.1%})\n")
            
            if 'age_mean' in row:
                f.write(f"  Age: {row['age_mean']:.1f} ± {row['age_std']:.1f} years\n")
            
            if 'pct_male' in row:
                f.write(f"  Male: {row['pct_male']:.1%}\n")
            
            if 'tissue_mean' in row:
                f.write(f"  Tissue: {row['tissue_mean']:.1f} ± {row['tissue_std']:.1f}\n")
            
            f.write("\n")
        
        # Statistical tests
        f.write("="*70 + "\n")
        f.write("STATISTICAL TESTS\n")
        f.write("="*70 + "\n\n")
        
        for key, result in test_results.items():
            f.write(f"{result['metric']}:\n")
            f.write(f"  Mean: {result['mean']:.3f}\n")
            f.write(f"  Std: {result['std']:.3f}\n")
            f.write(f"  CV: {result['cv']:.2%}\n")
            
            if 'p_value' in result:
                f.write(f"  p-value: {result['p_value']:.3f}\n")
            
            status = "✅ BALANCED" if result['balanced'] else "⚠️ IMBALANCED"
            f.write(f"  Status: {status}\n\n")
        
        # Recommendations
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        balanced_count = sum(1 for r in test_results.values() if r.get('balanced', False))
        total_tests = len(test_results)
        
        if balanced_count == total_tests:
            f.write("✅ Excellent stratification. No action needed.\n")
        elif balanced_count >= total_tests * 0.8:
            f.write("✅ Good stratification. Minor imbalances are acceptable.\n")
        else:
            f.write("⚠️ Consider improving stratification:\n")
            f.write("  1. Use multi-variate stratification (age bins + progression)\n")
            f.write("  2. Increase number of patients if possible\n")
            f.write("  3. Use repeated k-fold CV to reduce variance\n")
    
    print(f"✓ Saved report: {report_path}")
    
    return report_path


def main():
    """
    Main verification script
    """
    
    # Configuration - UPDATE THESE PATHS
    BASE_DIR = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training")
    
    KFOLD_PATH = BASE_DIR / "Dataset" / "kfold_splits_3fold_stratified.pkl"
    PATIENT_FEATURES_PATH = BASE_DIR / "Dataset" / "patient_features_30_60.csv"
    TRAIN_CSV_PATH = BASE_DIR / "Dataset" / "train.csv"
    
    RESULTS_DIR = BASE_DIR / "stratification_verification"
    
    print("\n" + "="*70)
    print("K-FOLD STRATIFICATION VERIFICATION")
    print("="*70)
    
    # Load data
    kfold_splits, patient_features = load_data(
        KFOLD_PATH, 
        PATIENT_FEATURES_PATH,
        TRAIN_CSV_PATH
    )
    
    # Compute statistics
    stats_df = compute_fold_statistics(kfold_splits, patient_features)
    
    # Save statistics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(RESULTS_DIR / "fold_statistics.csv", index=False)
    print(f"\n✓ Saved statistics: {RESULTS_DIR / 'fold_statistics.csv'}")
    
    # Statistical tests
    test_results = test_stratification_quality(stats_df)
    
    # Visualizations
    visualize_stratification(stats_df, RESULTS_DIR)
    
    # Generate report
    generate_detailed_report(stats_df, test_results, RESULTS_DIR)
    
    print("\n" + "="*70)
    print("✓ VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - fold_statistics.csv")
    print(f"  - stratification_analysis.png")
    print(f"  - stratification_report.txt")
    

if __name__ == "__main__":
    main()