"""
Compare All 4 Approaches - Comprehensive Analysis
==================================================

This script creates visualizations and statistical comparisons of all 4 approaches:
1. CNN Only
2. CNN + Handcrafted
3. CNN + Demographics
4. CNN + Handcrafted + Demographics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'results_dir': Path('Training/Retraining_cross_validation/results'),
    'plots_dir': Path('Training/Retraining_cross_validation/plots'),
}

APPROACHES = {
    'cnn_only': 'CNN Only',
    'cnn_handcrafted': 'CNN + Handcrafted',
    'cnn_demographics': 'CNN + Demographics',
    'cnn_full': 'CNN + HF + Demo'
}

COLORS = {
    'cnn_only': '#3498db',
    'cnn_handcrafted': '#e74c3c',
    'cnn_demographics': '#f39c12',
    'cnn_full': '#2ecc71'
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_predictions():
    """Load predictions from all approaches"""
    predictions = {}
    
    for approach_key in APPROACHES.keys():
        file_path = CONFIG['results_dir'] / f'{approach_key}_all_folds_fvc52_predictions.csv'
        if file_path.exists():
            predictions[approach_key] = pd.read_csv(file_path)
        else:
            print(f"⚠️  Warning: {file_path} not found")
    
    return predictions


def load_summary():
    """Load summary statistics"""
    summary_path = CONFIG['results_dir'] / 'fvc52_summary.csv'
    if summary_path.exists():
        return pd.read_csv(summary_path)
    else:
        print(f"⚠️  Warning: {summary_path} not found")
        return None


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def perform_statistical_tests(predictions_dict):
    """Perform pairwise statistical comparisons"""
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS (Paired t-tests)")
    print("="*80)
    
    # Get absolute errors for each approach
    errors = {}
    for key, df in predictions_dict.items():
        df_valid = df[df['has_true_fvc52']].copy()
        df_valid['abs_error'] = np.abs(df_valid['fvc52_true'] - df_valid['fvc52_predicted'])
        errors[key] = df_valid
    
    # Pairwise comparisons
    approach_keys = list(APPROACHES.keys())
    results = []
    
    for i in range(len(approach_keys)):
        for j in range(i + 1, len(approach_keys)):
            key1 = approach_keys[i]
            key2 = approach_keys[j]
            
            # Get common patients
            patients1 = set(errors[key1]['patient_id'])
            patients2 = set(errors[key2]['patient_id'])
            common_patients = patients1.intersection(patients2)
            
            if len(common_patients) > 0:
                # Extract errors for common patients
                err1_dict = dict(zip(errors[key1]['patient_id'], errors[key1]['abs_error']))
                err2_dict = dict(zip(errors[key2]['patient_id'], errors[key2]['abs_error']))
                
                err1_vals = [err1_dict[p] for p in common_patients]
                err2_vals = [err2_dict[p] for p in common_patients]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(err1_vals, err2_vals)
                
                mean_diff = np.mean(err1_vals) - np.mean(err2_vals)
                
                results.append({
                    'Comparison': f'{APPROACHES[key1]} vs {APPROACHES[key2]}',
                    'N_Common_Patients': len(common_patients),
                    'Mean_MAE_1': np.mean(err1_vals),
                    'Mean_MAE_2': np.mean(err2_vals),
                    'Mean_Difference': mean_diff,
                    'T_Statistic': t_stat,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
                
                print(f"\n{APPROACHES[key1]} vs {APPROACHES[key2]}:")
                print(f"  Common patients: {len(common_patients)}")
                print(f"  MAE: {np.mean(err1_vals):.2f} vs {np.mean(err2_vals):.2f}")
                print(f"  Difference: {mean_diff:.2f}")
                print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(CONFIG['results_dir'] / 'statistical_comparisons.csv', index=False)
    print(f"\n✓ Saved: {CONFIG['results_dir'] / 'statistical_comparisons.csv'}")
    
    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_comparison(predictions_dict, summary_df):
    """Create comprehensive comparison dashboard"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # 1. Mean Performance Metrics (Bar Chart)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = ['MAE_Mean', 'RMSE_Mean']
    x = np.arange(len(APPROACHES))
    width = 0.35
    
    mae_vals = summary_df['MAE_Mean'].values
    rmse_vals = summary_df['RMSE_Mean'].values
    
    ax1.bar(x - width/2, mae_vals, width, label='MAE', alpha=0.8, color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, rmse_vals, width, label='RMSE', alpha=0.8, color='coral', edgecolor='black')
    
    ax1.set_ylabel('Error (ml)', fontsize=11)
    ax1.set_title('Mean Errors Across Folds', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([APPROACHES[k] for k in APPROACHES.keys()], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # 2. R² Comparison (Bar Chart)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    r2_means = summary_df['R2_Mean'].values
    r2_stds = summary_df['R2_Std'].values
    
    bars = ax2.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.8,
                   color=[COLORS[k] for k in APPROACHES.keys()],
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('R² Performance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([APPROACHES[k] for k in APPROACHES.keys()], rotation=15, ha='right')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, r2_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # 3. Percentage Error (Bar Chart)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    pct_means = summary_df['PctError_Mean'].values
    pct_stds = summary_df['PctError_Std'].values
    
    bars = ax3.bar(x, pct_means, yerr=pct_stds, capsize=5, alpha=0.8,
                   color=[COLORS[k] for k in APPROACHES.keys()],
                   edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Mean % Error', fontsize=11)
    ax3.set_title('Percentage Error', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([APPROACHES[k] for k in APPROACHES.keys()], rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, pct_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # 4. Scatter: Predicted vs True (All Approaches)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, :])
    
    for approach_key, df in predictions_dict.items():
        df_valid = df[df['has_true_fvc52']].copy()
        
        if len(df_valid) > 0:
            ax4.scatter(df_valid['fvc52_true'], df_valid['fvc52_predicted'],
                       alpha=0.5, s=30, label=APPROACHES[approach_key],
                       color=COLORS[approach_key], edgecolors='black', linewidth=0.5)
    
    # Identity line
    min_val = min([df[df['has_true_fvc52']]['fvc52_true'].min() for df in predictions_dict.values()])
    max_val = max([df[df['has_true_fvc52']]['fvc52_true'].max() for df in predictions_dict.values()])
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    ax4.set_xlabel('True FVC@52 (ml)', fontsize=11)
    ax4.set_ylabel('Predicted FVC@52 (ml)', fontsize=11)
    ax4.set_title('Predicted vs True FVC@52 - All Approaches', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. Error Distribution (Violin Plot)
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    error_data = []
    labels = []
    
    for approach_key in APPROACHES.keys():
        df = predictions_dict[approach_key]
        df_valid = df[df['has_true_fvc52']].copy()
        errors = df_valid['fvc52_predicted'] - df_valid['fvc52_true']
        error_data.append(errors)
        labels.append(APPROACHES[approach_key])
    
    parts = ax5.violinplot(error_data, positions=range(len(APPROACHES)), 
                           showmeans=True, showmedians=True)
    
    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(COLORS.values())[i])
        pc.set_alpha(0.7)
    
    ax5.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Error')
    ax5.set_ylabel('Prediction Error (ml)', fontsize=11)
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(APPROACHES)))
    ax5.set_xticklabels(labels, rotation=15, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend()
    
    # =========================================================================
    # 6. Absolute Error Distribution (Box Plot)
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    abs_error_data = []
    
    for approach_key in APPROACHES.keys():
        df = predictions_dict[approach_key]
        df_valid = df[df['has_true_fvc52']].copy()
        abs_errors = np.abs(df_valid['fvc52_predicted'] - df_valid['fvc52_true'])
        abs_error_data.append(abs_errors)
    
    bp = ax6.boxplot(abs_error_data, labels=labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(alpha=0.7))
    
    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(list(COLORS.values())[i])
    
    ax6.set_ylabel('Absolute Error (ml)', fontsize=11)
    ax6.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(labels, rotation=15, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # 7. Cumulative Error Distribution
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 2])
    
    for approach_key in APPROACHES.keys():
        df = predictions_dict[approach_key]
        df_valid = df[df['has_true_fvc52']].copy()
        abs_errors = np.abs(df_valid['fvc52_predicted'] - df_valid['fvc52_true'])
        
        sorted_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        ax7.plot(sorted_errors, cumulative, linewidth=2.5, 
                label=APPROACHES[approach_key], color=COLORS[approach_key])
    
    ax7.set_xlabel('Absolute Error (ml)', fontsize=11)
    ax7.set_ylabel('Cumulative % of Patients', fontsize=11)
    ax7.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax7.legend(loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(left=0)
    ax7.set_ylim([0, 100])
    
    # Add reference lines
    for pct in [50, 75, 90]:
        ax7.axhline(pct, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    plot_path = CONFIG['plots_dir'] / 'comprehensive_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    
    return fig


def create_fold_consistency_plot(predictions_dict):
    """Plot performance consistency across folds"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['MAE', 'RMSE', 'R²', '% Error']
    
    for idx, (metric_name, ax) in enumerate(zip(metrics, axes)):
        for approach_key in APPROACHES.keys():
            df = predictions_dict[approach_key]
            
            # Compute metric per fold
            fold_metrics = []
            for fold in df['fold'].unique():
                fold_df = df[(df['fold'] == fold) & (df['has_true_fvc52'])].copy()
                
                if len(fold_df) > 0:
                    if metric_name == 'MAE':
                        val = np.mean(np.abs(fold_df['fvc52_predicted'] - fold_df['fvc52_true']))
                    elif metric_name == 'RMSE':
                        val = np.sqrt(np.mean((fold_df['fvc52_predicted'] - fold_df['fvc52_true'])**2))
                    elif metric_name == 'R²':
                        from sklearn.metrics import r2_score
                        val = r2_score(fold_df['fvc52_true'], fold_df['fvc52_predicted'])
                    else:  # % Error
                        val = np.mean(np.abs(fold_df['fvc52_predicted'] - fold_df['fvc52_true']) / fold_df['fvc52_true'] * 100)
                    
                    fold_metrics.append(val)
            
            folds = list(range(1, len(fold_metrics) + 1))
            ax.plot(folds, fold_metrics, marker='o', linewidth=2, markersize=8,
                   label=APPROACHES[approach_key], color=COLORS[approach_key])
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Across Folds', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 6))
    
    plt.tight_layout()
    
    plot_path = CONFIG['plots_dir'] / 'fold_consistency.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE COMPARISON OF ALL APPROACHES")
    print("="*80)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    predictions_dict = load_all_predictions()
    summary_df = load_summary()
    
    print(f"✓ Loaded predictions for {len(predictions_dict)} approaches")
    
    # Display summary
    if summary_df is not None:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary_df.to_string(index=False))
    
    # Statistical tests
    stats_df = perform_statistical_tests(predictions_dict)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig1 = create_comprehensive_comparison(predictions_dict, summary_df)
    plt.show()
    
    fig2 = create_fold_consistency_plot(predictions_dict)
    plt.show()
    
    # Final summary
    print("\n" + "="*80)
    print("BEST APPROACH ANALYSIS")
    print("="*80)
    
    if summary_df is not None:
        best_mae = summary_df.loc[summary_df['MAE_Mean'].idxmin()]
        best_r2 = summary_df.loc[summary_df['R2_Mean'].idxmax()]
        
        print(f"\n🏆 Best MAE: {best_mae['Approach']}")
        print(f"   MAE: {best_mae['MAE_Mean']:.2f} ± {best_mae['MAE_Std']:.2f} ml")
        
        print(f"\n🏆 Best R²: {best_r2['Approach']}")
        print(f"   R²: {best_r2['R2_Mean']:.4f} ± {best_r2['R2_Std']:.4f}")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
    print(f"\nPlots saved in: {CONFIG['plots_dir']}")
    print(f"Results saved in: {CONFIG['results_dir']}")


if __name__ == "__main__":
    main()
