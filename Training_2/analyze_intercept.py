"""
Analyze Relationship Between Regression Intercept and Baseline FVC
===================================================================

This script investigates:
1. Whether intercept from linear regression equals baseline FVC
2. How to handle cases where baseline (week 0) measurements are missing
3. Whether we can reconstruct baseline FVC from intercept
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train.csv',
    'output_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\CNN_Training\Cyclic_kfold\baseline_analysis')
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("ANALYZING INTERCEPT vs BASELINE FVC")
print("="*80)

print(f"\n📂 Loading data from:")
print(f"   CSV: {CONFIG['csv_path']}")
print(f"   Features: {CONFIG['features_path']}")

# Load coefficients (slope, intercept)
df_coefs = pd.read_csv(CONFIG['csv_path'])
print(f"\n✓ Loaded {len(df_coefs)} patients with slope/intercept")
print(f"  Columns: {df_coefs.columns.tolist()}")

# Load features (longitudinal FVC data)
df_features = pd.read_csv(CONFIG['features_path'])
print(f"\n✓ Loaded {len(df_features)} feature records")
print(f"  Columns: {df_features.columns.tolist()}")

# =============================================================================
# ANALYZE TIME ENCODING
# =============================================================================

print("\n" + "="*80)
print("STEP 1: ANALYZE TIME ENCODING")
print("="*80)

# Check what time column exists
time_cols = [col for col in df_features.columns if 'week' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
print(f"\n🕐 Time columns found: {time_cols}")

if 'Weeks' in df_features.columns:
    time_col = 'Weeks'
elif 'Week' in df_features.columns:
    time_col = 'Week'
elif 'week' in df_features.columns:
    time_col = 'week'
else:
    # Try to find any time-related column
    if time_cols:
        time_col = time_cols[0]
    else:
        print("❌ No time column found! Please check your data.")
        time_col = None

if time_col:
    print(f"\n✓ Using time column: '{time_col}'")
    print(f"  Unique time values: {sorted(df_features[time_col].unique())}")
    print(f"  Range: {df_features[time_col].min()} to {df_features[time_col].max()}")

# =============================================================================
# FIND BASELINE FVC
# =============================================================================

print("\n" + "="*80)
print("STEP 2: IDENTIFY BASELINE FVC")
print("="*80)

# Find FVC column
fvc_cols = [col for col in df_features.columns if 'fvc' in col.lower() or 'FVC' in col]
print(f"\n📊 FVC columns found: {fvc_cols}")

if fvc_cols:
    fvc_col = fvc_cols[0]
    print(f"✓ Using FVC column: '{fvc_col}'")
else:
    print("❌ No FVC column found!")
    fvc_col = None

if time_col and fvc_col:
    # Group by patient and find earliest measurement
    baseline_fvc = []
    
    for patient_id in df_coefs['Patient'].unique():
        patient_measurements = df_features[df_features['Patient'] == patient_id].copy()
        
        if len(patient_measurements) == 0:
            baseline_fvc.append({
                'Patient': patient_id,
                'has_baseline': False,
                'earliest_week': None,
                'baseline_fvc': None,
                'n_measurements': 0
            })
            continue
        
        # Find earliest measurement
        earliest_idx = patient_measurements[time_col].idxmin()
        earliest_week = patient_measurements.loc[earliest_idx, time_col]
        earliest_fvc = patient_measurements.loc[earliest_idx, fvc_col]
        
        baseline_fvc.append({
            'Patient': patient_id,
            'has_baseline': earliest_week == 0,  # True baseline at week 0
            'earliest_week': earliest_week,
            'baseline_fvc': earliest_fvc,
            'n_measurements': len(patient_measurements)
        })
    
    df_baseline = pd.DataFrame(baseline_fvc)
    
    # Merge with coefficients
    df_analysis = df_coefs.merge(df_baseline, on='Patient', how='left')

    # Rinomina per coerenza semantica
    df_analysis = df_analysis.rename(columns={
        'fvc_intercept0': 'intercept',
        'fvc_slope': 'slope'
    })
    
    # Calcola distanza assoluta da settimana 0
    df_analysis['abs_week'] = df_analysis['Weeks'].abs()

    # Per ogni paziente, prendi la riga con settimana più vicina a 0
    df_baseline_closest = (
        df_analysis
        .loc[df_analysis.groupby('Patient')['abs_week'].idxmin()]
        .drop(columns='abs_week')
        .reset_index(drop=True)
    )

    print(f"\n✓ Extracted closest-to-baseline measurements for each patient.")
    print(f"  Total patients: {len(df_baseline_closest)}")
    print(f"\n📋 Sample data:"
          f"\n{df_baseline_closest.head()}")
    
    print(f"\n📈 Baseline FVC Summary:")
    print(f"  Total patients: {len(df_baseline_closest)}")
    print(f"  Patients with week 0 measurement: {df_baseline_closest['has_baseline'].sum()}")
    print(f"  Patients with shifted baseline: {(~df_baseline_closest['has_baseline'] & df_baseline_closest['baseline_fvc'].notna()).sum()}")
    print(f"  Patients with no FVC data: {df_baseline_closest['baseline_fvc'].isna().sum()}")
    
    # Show earliest week distribution
    print(f"\n🕐 Earliest measurement distribution:")
    earliest_week_counts = df_baseline_closest['earliest_week'].value_counts().sort_index()
    for week, count in earliest_week_counts.items():
        if pd.notna(week):
            print(f"  Week {int(week):3d}: {count:3d} patients ({count/len(df_baseline_closest)*100:.1f}%)")

# =============================================================================
# COMPARE INTERCEPT vs BASELINE FVC
# =============================================================================

print("\n" + "="*80)
print("STEP 3: COMPARE INTERCEPT vs BASELINE FVC")
print("="*80)

if 'intercept' in df_baseline_closest.columns and 'baseline_fvc' in df_baseline_closest.columns:
    # Filter to patients with both values
    df_compare = df_baseline_closest[df_baseline_closest['baseline_fvc'].notna() & df_baseline_closest['intercept'].notna()].copy()
    
    print(f"\n📊 Patients with both intercept and baseline FVC: {len(df_compare)}")
    
    if len(df_compare) > 0:
        # Calculate differences
        df_compare['difference'] = df_compare['intercept'] - df_compare['baseline_fvc']
        df_compare['abs_difference'] = df_compare['difference'].abs()
        df_compare['percent_difference'] = (df_compare['difference'] / df_compare['baseline_fvc'] * 100)
        
        print(f"\n📏 Difference Statistics (Intercept - Baseline FVC):")
        print(f"  Mean difference: {df_compare['difference'].mean():.2f} ± {df_compare['difference'].std():.2f}")
        print(f"  Median difference: {df_compare['difference'].median():.2f}")
        print(f"  Mean absolute difference: {df_compare['abs_difference'].mean():.2f}")
        print(f"  Mean percent difference: {df_compare['percent_difference'].mean():.2f}%")
        
        # Correlation
        corr, p_value = stats.pearsonr(df_compare['intercept'], df_compare['baseline_fvc'])
        print(f"\n🔗 Correlation:")
        print(f"  Pearson r = {corr:.4f} (p < {p_value:.2e})")
        
        # Linear regression
        slope, intercept_reg, r_value, p_value_reg, std_err = stats.linregress(
            df_compare['baseline_fvc'], df_compare['intercept']
        )
        print(f"\n📈 Linear Regression (intercept ~ baseline_fvc):")
        print(f"  intercept = {slope:.4f} × baseline_fvc + {intercept_reg:.2f}")
        print(f"  R² = {r_value**2:.4f}")
        
        # Check if intercept ≈ baseline_fvc (slope ≈ 1, intercept ≈ 0)
        if abs(slope - 1.0) < 0.05 and abs(intercept_reg) < 50:
            print(f"\n✅ CONCLUSION: Intercept ≈ Baseline FVC")
            print(f"   The regression intercept is essentially the baseline FVC measurement!")
        else:
            print(f"\n⚠️  CONCLUSION: Intercept ≠ Baseline FVC")
            print(f"   There's a systematic difference - check time encoding in original regression!")
        
        # =============================================================================
        # VISUALIZATIONS
        # =============================================================================
        
        print(f"\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scatter plot: Intercept vs Baseline FVC
        ax1 = axes[0, 0]
        ax1.scatter(df_compare['baseline_fvc'], df_compare['intercept'], alpha=0.5, s=30)
        ax1.plot([df_compare['baseline_fvc'].min(), df_compare['baseline_fvc'].max()],
                 [df_compare['baseline_fvc'].min(), df_compare['baseline_fvc'].max()],
                 'r--', label='y=x (perfect match)', linewidth=2)
        
        # Add regression line
        x_range = np.array([df_compare['baseline_fvc'].min(), df_compare['baseline_fvc'].max()])
        y_pred = slope * x_range + intercept_reg
        ax1.plot(x_range, y_pred, 'b-', label=f'Fit: y={slope:.3f}x+{intercept_reg:.1f}', linewidth=2)
        
        ax1.set_xlabel('Baseline FVC (earliest measurement)', fontsize=11)
        ax1.set_ylabel('Regression Intercept', fontsize=11)
        ax1.set_title(f'Intercept vs Baseline FVC (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Difference distribution
        ax2 = axes[0, 1]
        ax2.hist(df_compare['difference'], bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax2.axvline(df_compare['difference'].mean(), color='blue', linestyle='-', 
                    linewidth=2, label=f'Mean: {df_compare["difference"].mean():.1f}')
        ax2.set_xlabel('Difference (Intercept - Baseline FVC)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Differences', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Bland-Altman plot
        ax3 = axes[1, 0]
        mean_values = (df_compare['intercept'] + df_compare['baseline_fvc']) / 2
        differences = df_compare['difference']
        mean_diff = differences.mean()
        std_diff = differences.std()
        
        ax3.scatter(mean_values, differences, alpha=0.5, s=30)
        ax3.axhline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.1f}')
        ax3.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5, 
                    label=f'±1.96 SD: [{mean_diff-1.96*std_diff:.1f}, {mean_diff+1.96*std_diff:.1f}]')
        ax3.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5)
        ax3.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax3.set_xlabel('Mean of Intercept and Baseline FVC', fontsize=11)
        ax3.set_ylabel('Difference (Intercept - Baseline)', fontsize=11)
        ax3.set_title('Bland-Altman Plot', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Breakdown by earliest week
        ax4 = axes[1, 1]
        week_groups = df_compare.groupby('earliest_week')['difference'].agg(['mean', 'std', 'count'])
        week_groups = week_groups[week_groups['count'] >= 5]  # Only show weeks with >=5 patients
        
        x_pos = np.arange(len(week_groups))
        ax4.bar(x_pos, week_groups['mean'], yerr=week_groups['std'], 
                capsize=5, alpha=0.7, edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Wk {int(w)}' for w in week_groups.index], rotation=45)
        ax4.axhline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Earliest Measurement Week', fontsize=11)
        ax4.set_ylabel('Mean Difference (Intercept - Baseline)', fontsize=11)
        ax4.set_title('Difference by Earliest Measurement Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = CONFIG['output_dir'] / 'intercept_vs_baseline_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {plot_path}")
        
        # =============================================================================
        # SAVE RESULTS
        # =============================================================================
        
        # Save detailed comparison
        comparison_cols = ['Patient', 'intercept', 'baseline_fvc', 'earliest_week', 
                          'difference', 'percent_difference', 'has_baseline', 'slope']
        df_compare[comparison_cols].to_csv(
            CONFIG['output_dir'] / 'intercept_baseline_comparison.csv', 
            index=False
        )
        print(f"✓ Saved detailed comparison to: intercept_baseline_comparison.csv")
        
        # Save summary statistics
        summary = {
            'total_patients': int(len(df_baseline_closest)),
            'patients_with_both_values': int(len(df_compare)),
            'patients_with_week_0': int(df_baseline_closest['has_baseline'].sum()),
            'correlation': float(corr),
            'correlation_pvalue': float(p_value),
            'mean_difference': float(df_compare['difference'].mean()),
            'std_difference': float(df_compare['difference'].std()),
            'median_difference': float(df_compare['difference'].median()),
            'mean_abs_difference': float(df_compare['abs_difference'].mean()),
            'mean_percent_difference': float(df_compare['percent_difference'].mean()),
            'regression_slope': float(slope),
            'regression_intercept': float(intercept_reg),
            'regression_r2': float(r_value**2)
        }

        
        import json
        with open(CONFIG['output_dir'] / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to: analysis_summary.json")

# =============================================================================
# RECONSTRUCTION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("STEP 4: CAN WE RECONSTRUCT BASELINE FVC FROM INTERCEPT?")
print("="*80)

if 'intercept' in df_baseline_closest.columns and 'baseline_fvc' in df_baseline_closest.columns and 'slope' in df_baseline_closest.columns:
    # For patients WITHOUT week 0, try to reconstruct
    df_shifted = df_baseline_closest[~df_baseline_closest['has_baseline'] & df_baseline_closest['baseline_fvc'].notna()].copy()
    
    print(f"\n📊 Patients with shifted baseline (no week 0): {len(df_shifted)}")
    
    if len(df_shifted) > 0:
        # Reconstruct week 0 from: FVC(week_n) = intercept + slope × week_n
        # Therefore: intercept = FVC(week_n) - slope × week_n
        df_shifted['reconstructed_intercept'] = df_shifted['baseline_fvc'] - df_shifted['slope'] * df_shifted['earliest_week']
        df_shifted['reconstruction_error'] = df_shifted['intercept'] - df_shifted['reconstructed_intercept']
        
        print(f"\n📏 Reconstruction Error (Actual Intercept - Reconstructed):")
        print(f"  Mean error: {df_shifted['reconstruction_error'].mean():.2f} ± {df_shifted['reconstruction_error'].std():.2f}")
        print(f"  Median error: {df_shifted['reconstruction_error'].median():.2f}")
        print(f"  Mean absolute error: {df_shifted['reconstruction_error'].abs().mean():.2f}")
        
        if df_shifted['reconstruction_error'].abs().mean() < 50:
            print(f"\n✅ GOOD NEWS: We can reconstruct baseline FVC from intercept!")
            print(f"   For shifted patients: baseline_fvc ≈ intercept (directly)")
        else:
            print(f"\n⚠️  WARNING: Reconstruction has significant error")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 Results saved to: {CONFIG['output_dir']}")