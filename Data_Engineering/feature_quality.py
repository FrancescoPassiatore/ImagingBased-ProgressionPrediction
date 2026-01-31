"""
Data Quality Check Script for CT Feature Extraction
====================================================

This script performs comprehensive validation and quality checks on your 
extracted CT features to identify issues before modeling.

Usage:
    python check_feature_quality.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = "C:\\Users\\frank\\OneDrive\\Desktop\\ImagingBased-ProgressionPrediction\\Data_Engineering"
FEATURES_FILE = "patient_features_30_60.csv"

# Feature definitions for validation
EXPECTED_FEATURES = {
    'Patient': 'string',
    'Data': 'string',
    'SliceThickness': 'float',
    'PixelSpacing': 'float',
    'NumImgBw5Prec': 'int',
    'ApproxVol_30_60': 'float',
    'Avg_NumTissuePixel_30_60': 'float',
    'Avg_Tissue_30_60': 'float',
    'Avg_Tissue_thickness_30_60': 'float',
    'Avg_TissueByTotal_30_60': 'float',
    'Avg_TissueByLung_30_60': 'float',
    'Mean_30_60': 'float',
    'Skew_30_60': 'float',
    'Kurtosis_30_60': 'float'
}

# Expected value ranges (based on clinical knowledge)
EXPECTED_RANGES = {
    'SliceThickness': (0.5, 10.0),         # mm
    'PixelSpacing': (0.3, 1.0),            # mm
    'NumImgBw5Prec': (1, 50),              # slices between percentiles
    'ApproxVol_30_60': (100000, 15000000), # mm³ (lung volume estimate)
    'Avg_NumTissuePixel_30_60': (1000, 50000),  # pixels
    'Avg_Tissue_30_60': (500, 30000),      # mm²
    'Avg_Tissue_thickness_30_60': (500, 100000),  # mm³
    'Avg_TissueByTotal_30_60': (0.005, 0.15),  # ratio
    'Avg_TissueByLung_30_60': (0.03, 0.5),  # ratio (key fibrosis metric)
    'Mean_30_60': (-1000, 1000),           # Hounsfield Units
    'Skew_30_60': (-2, 5),                 # skewness
    'Kurtosis_30_60': (-2, 10)             # kurtosis
}


# ============================================================================
# BASIC DATA CHECKS
# ============================================================================

def load_and_validate_structure(filepath):
    """Load data and check basic structure"""
    print("="*80)
    print("STEP 1: LOADING DATA AND CHECKING STRUCTURE")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded {filepath}")
        print(f"  Shape: {df.shape[0]} patients × {df.shape[1]} features")
    except Exception as e:
        print(f"✗ ERROR loading file: {e}")
        return None
    
    # Check expected columns
    print("\n--- Column Check ---")
    missing_cols = set(EXPECTED_FEATURES.keys()) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_FEATURES.keys())
    
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
    else:
        print("✓ All expected columns present")
    
    if extra_cols:
        print(f"⚠ Extra columns: {extra_cols}")
    
    # Check for duplicate patients
    print("\n--- Duplicate Check ---")
    duplicates = df[df.duplicated(subset=['Patient'], keep=False)]
    if len(duplicates) > 0:
        print(f"✗ Found {len(duplicates)} duplicate patient entries:")
        print(duplicates[['Patient', 'Data']])
    else:
        print("✓ No duplicate patients found")
    
    return df


def check_missing_values(df):
    """Check for missing or invalid values"""
    print("\n" + "="*80)
    print("STEP 2: MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing_summary = []
    
    for col in df.columns:
        if col in ['Patient', 'Data']:
            continue
            
        n_missing = df[col].isna().sum()
        n_zero = (df[col] == 0).sum()
        n_negative = (df[col] < 0).sum() if col not in ['Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'] else 0
        
        missing_summary.append({
            'Feature': col,
            'Missing': n_missing,
            'Missing %': f"{100*n_missing/len(df):.1f}%",
            'Zero': n_zero,
            'Zero %': f"{100*n_zero/len(df):.1f}%",
            'Negative': n_negative if n_negative > 0 else '-'
        })
    
    missing_df = pd.DataFrame(missing_summary)
    print(missing_df.to_string(index=False))
    
    # Flag issues
    print("\n--- Issues Detected ---")
    issues_found = False
    
    for _, row in missing_df.iterrows():
        if row['Missing'] > 0:
            print(f"⚠ {row['Feature']}: {row['Missing']} missing values ({row['Missing %']})")
            issues_found = True
        
        # Zero values in key features are suspicious
        if row['Feature'] in ['ApproxVol_30_60', 'Avg_NumTissuePixel_30_60'] and row['Zero'] > 0:
            print(f"⚠ {row['Feature']}: {row['Zero']} zero values - likely extraction failures")
            issues_found = True
    
    if not issues_found:
        print("✓ No critical missing value issues detected")
    
    return missing_df


def check_value_ranges(df):
    """Check if values fall within expected ranges"""
    print("\n" + "="*80)
    print("STEP 3: VALUE RANGE VALIDATION")
    print("="*80)
    
    range_issues = []
    
    for feature, (min_val, max_val) in EXPECTED_RANGES.items():
        if feature not in df.columns:
            continue
        
        data = df[feature].dropna()
        
        below_min = (data < min_val).sum()
        above_max = (data > max_val).sum()
        
        if below_min > 0 or above_max > 0:
            range_issues.append({
                'Feature': feature,
                'Expected Range': f"[{min_val}, {max_val}]",
                'Actual Min': f"{data.min():.4f}",
                'Actual Max': f"{data.max():.4f}",
                'Below Min': below_min,
                'Above Max': above_max,
                'Total Out': below_min + above_max
            })
    
    if range_issues:
        issues_df = pd.DataFrame(range_issues)
        print(issues_df.to_string(index=False))
        
        print("\n⚠ Warning: Values outside expected ranges detected")
        print("This may indicate:")
        print("  - Extraction errors")
        print("  - Different scanner settings")
        print("  - Edge cases in the data")
    else:
        print("✓ All values within expected ranges")
    
    return range_issues


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def generate_statistical_summary(df):
    """Generate comprehensive statistical summary"""
    print("\n" + "="*80)
    print("STEP 4: STATISTICAL SUMMARY")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary_stats = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        summary_stats.append({
            'Feature': col,
            'Count': len(data),
            'Mean': f"{data.mean():.4f}",
            'Std': f"{data.std():.4f}",
            'Min': f"{data.min():.4f}",
            '25%': f"{data.quantile(0.25):.4f}",
            'Median': f"{data.median():.4f}",
            '75%': f"{data.quantile(0.75):.4f}",
            'Max': f"{data.max():.4f}"
        })
    
    stats_df = pd.DataFrame(summary_stats)
    print(stats_df.to_string(index=False))
    
    return stats_df


def check_outliers(df):
    """Detect outliers using IQR method"""
    print("\n" + "="*80)
    print("STEP 5: OUTLIER DETECTION")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = 100 * outliers / len(data)
        
        if outliers > 0:
            outlier_summary.append({
                'Feature': col,
                'N_Outliers': outliers,
                'Percent': f"{outlier_pct:.1f}%",
                'Lower Bound': f"{lower_bound:.4f}",
                'Upper Bound': f"{upper_bound:.4f}"
            })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.to_string(index=False))
        
        print("\n⚠ Note: Outliers are not necessarily errors")
        print("  They may represent true biological variation or extreme cases")
    else:
        print("✓ No significant outliers detected (3×IQR method)")
    
    return outlier_summary


# ============================================================================
# CORRELATION & RELATIONSHIP CHECKS
# ============================================================================

def check_feature_correlations(df):
    """Check correlations between features to identify potential issues"""
    print("\n" + "="*80)
    print("STEP 6: FEATURE CORRELATIONS")
    print("="*80)
    
    # Select numeric features
    numeric_cols = [col for col in df.columns if col not in ['Patient', 'Data']]
    numeric_df = df[numeric_cols].dropna()
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find highly correlated pairs (excluding self-correlation)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.95:  # Very high correlation threshold
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.4f}"
                })
    
    if high_corr:
        print("⚠ Highly correlated feature pairs (|r| > 0.95):")
        high_corr_df = pd.DataFrame(high_corr)
        print(high_corr_df.to_string(index=False))
        print("\nConsider removing one feature from each pair to reduce multicollinearity")
    else:
        print("✓ No extremely high correlations detected")
    
    # Check expected relationships
    print("\n--- Expected Relationship Checks ---")
    
    # Check 1: Volume should relate to tissue measures
    if 'ApproxVol_30_60' in df.columns and 'Avg_NumTissuePixel_30_60' in df.columns:
        valid_data = df[['ApproxVol_30_60', 'Avg_NumTissuePixel_30_60']].dropna()
        valid_data = valid_data[(valid_data > 0).all(axis=1)]
        
        if len(valid_data) > 10:
            corr, _ = stats.pearsonr(valid_data['ApproxVol_30_60'], 
                                     valid_data['Avg_NumTissuePixel_30_60'])
            print(f"Volume vs Tissue Pixels: r = {corr:.3f}")
            if abs(corr) < 0.3:
                print("  ⚠ Warning: Weak correlation - check volume calculation")
    
    # Check 2: Tissue ratios should be related
    if 'Avg_TissueByTotal_30_60' in df.columns and 'Avg_TissueByLung_30_60' in df.columns:
        valid_data = df[['Avg_TissueByTotal_30_60', 'Avg_TissueByLung_30_60']].dropna()
        valid_data = valid_data[(valid_data > 0).all(axis=1)]
        
        if len(valid_data) > 10:
            corr, _ = stats.pearsonr(valid_data['Avg_TissueByTotal_30_60'],
                                     valid_data['Avg_TissueByLung_30_60'])
            print(f"Tissue-by-Total vs Tissue-by-Lung: r = {corr:.3f}")
            if corr < 0.5:
                print("  ⚠ Warning: Lower correlation than expected")
    
    return corr_matrix


# ============================================================================
# SPECIFIC QUALITY CHECKS
# ============================================================================

def check_fibrosis_metrics(df):
    """Check key fibrosis-related metrics"""
    print("\n" + "="*80)
    print("STEP 7: FIBROSIS METRIC VALIDATION")
    print("="*80)
    
    key_metric = 'Avg_TissueByLung_30_60'
    
    if key_metric not in df.columns:
        print(f"✗ Key metric '{key_metric}' not found")
        return
    
    data = df[key_metric].dropna()
    
    print(f"Analyzing: {key_metric} (primary fibrosis indicator)")
    print(f"  N = {len(data)}")
    print(f"  Mean = {data.mean():.4f}")
    print(f"  Median = {data.median():.4f}")
    print(f"  Std = {data.std():.4f}")
    print(f"  Range = [{data.min():.4f}, {data.max():.4f}]")
    
    # Clinical interpretation
    print("\n--- Clinical Interpretation ---")
    low_fibrosis = (data < 0.10).sum()
    moderate_fibrosis = ((data >= 0.10) & (data < 0.25)).sum()
    high_fibrosis = (data >= 0.25).sum()
    
    print(f"Low fibrosis (<10%):      {low_fibrosis} patients ({100*low_fibrosis/len(data):.1f}%)")
    print(f"Moderate fibrosis (10-25%): {moderate_fibrosis} patients ({100*moderate_fibrosis/len(data):.1f}%)")
    print(f"High fibrosis (>25%):     {high_fibrosis} patients ({100*high_fibrosis/len(data):.1f}%)")
    
    # Check for suspiciously low values
    very_low = (data < 0.03).sum()
    if very_low > 0:
        print(f"\n⚠ Warning: {very_low} patients with very low tissue-by-lung (<3%)")
        print("  This may indicate segmentation issues or early disease")


def check_histogram_features(df):
    """Validate histogram-derived features"""
    print("\n" + "="*80)
    print("STEP 8: HISTOGRAM FEATURE VALIDATION")
    print("="*80)
    
    histogram_features = ['Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60']
    
    for feature in histogram_features:
        if feature not in df.columns:
            print(f"⚠ {feature} not found")
            continue
        
        data = df[feature].dropna()
        
        print(f"\n{feature}:")
        print(f"  N = {len(data)}")
        print(f"  Mean = {data.mean():.4f}")
        print(f"  Median = {data.median():.4f}")
        print(f"  Range = [{data.min():.4f}, {data.max():.4f}]")
        
        # Check for suspicious patterns
        if feature == 'Mean_30_60':
            # Lung tissue should be negative HU (air) to slightly positive (tissue)
            very_positive = (data > 500).sum()
            if very_positive > 0:
                print(f"  ⚠ {very_positive} patients with mean > 500 HU (very dense)")
            
            very_negative = (data < -800).sum()
            if very_negative > 0:
                print(f"  ⚠ {very_negative} patients with mean < -800 HU (very aerated)")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_diagnostic_plots(df, output_dir):
    """Create diagnostic visualizations"""
    print("\n" + "="*80)
    print("STEP 9: GENERATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Distribution of key features
    key_features = [
        'Avg_TissueByLung_30_60',
        'ApproxVol_30_60',
        'Avg_NumTissuePixel_30_60',
        'Mean_30_60'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            data = df[feature].dropna()
            
            axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].axvline(data.mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {data.mean():.2f}')
            axes[idx].axvline(data.median(), color='blue', linestyle='--', 
                            linewidth=2, label=f'Median: {data.median():.2f}')
            axes[idx].set_xlabel(feature.replace('_', ' '))
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'Distribution: {feature}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: Correlation heatmap
    numeric_cols = [col for col in df.columns if col not in ['Patient', 'Data']]
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot2_path}")
    plt.close()
    
    # Plot 3: Box plots for outlier visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            data = df[feature].dropna()
            
            axes[idx].boxplot(data, vert=True)
            axes[idx].set_ylabel(feature.replace('_', ' '))
            axes[idx].set_title(f'Box Plot: {feature}')
            axes[idx].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'boxplots_outliers.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot3_path}")
    plt.close()


# ============================================================================
# FINAL REPORT
# ============================================================================

def generate_quality_report(df, output_dir):
    """Generate comprehensive quality report"""
    print("\n" + "="*80)
    print("STEP 10: GENERATING QUALITY REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("DATA QUALITY ASSESSMENT REPORT")
    report.append("="*80)
    report.append(f"\nDataset: {FEATURES_FILE}")
    report.append(f"Total Patients: {len(df)}")
    report.append(f"Total Features: {len(df.columns)}")
    
    # Data completeness
    report.append("\n" + "="*80)
    report.append("DATA COMPLETENESS")
    report.append("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    complete_cases = df[numeric_cols].dropna().shape[0]
    report.append(f"Complete cases (no missing values): {complete_cases}/{len(df)} ({100*complete_cases/len(df):.1f}%)")
    
    # Feature quality summary
    report.append("\n" + "="*80)
    report.append("FEATURE QUALITY SUMMARY")
    report.append("="*80)
    
    for col in numeric_cols:
        data = df[col].dropna()
        zero_count = (data == 0).sum()
        
        report.append(f"\n{col}:")
        report.append(f"  Valid: {len(data)}/{len(df)} ({100*len(data)/len(df):.1f}%)")
        report.append(f"  Mean ± SD: {data.mean():.4f} ± {data.std():.4f}")
        report.append(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
        if zero_count > 0:
            report.append(f"  Zero values: {zero_count} ({100*zero_count/len(data):.1f}%)")
    
    # Recommendations
    report.append("\n" + "="*80)
    report.append("RECOMMENDATIONS")
    report.append("="*80)
    
    # Check for issues
    issues = []
    
    # Missing values
    if complete_cases < len(df):
        issues.append("1. MISSING VALUES: Consider imputation or patient exclusion")
    
    # Zero volumes
    if 'ApproxVol_30_60' in df.columns:
        zero_vol = (df['ApproxVol_30_60'] == 0).sum()
        if zero_vol > 0:
            issues.append(f"2. ZERO VOLUMES: {zero_vol} patients with zero volume - re-extract features")
    
    # Outliers in key metrics
    if 'Avg_TissueByLung_30_60' in df.columns:
        extreme_tissue = (df['Avg_TissueByLung_30_60'] > 0.5).sum()
        if extreme_tissue > 0:
            issues.append(f"3. EXTREME FIBROSIS: {extreme_tissue} patients with >50% tissue - verify")
    
    if issues:
        report.extend(issues)
    else:
        report.append("✓ No critical issues detected - data quality is acceptable")
    
    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'quality_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Saved: {report_path}")
    print("\n" + report_text)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("CT FEATURE QUALITY CHECK - STARTING ANALYSIS")
    print("="*80)
    
    # File path
    filepath = os.path.join(BASE_PATH, FEATURES_FILE)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return
    
    # Load and validate
    df = load_and_validate_structure(filepath)
    if df is None:
        return
    
    # Run all checks
    missing_df = check_missing_values(df)
    range_issues = check_value_ranges(df)
    stats_df = generate_statistical_summary(df)
    outlier_summary = check_outliers(df)
    corr_matrix = check_feature_correlations(df)
    check_fibrosis_metrics(df)
    check_histogram_features(df)
    
    # Generate visualizations
    output_dir = BASE_PATH
    create_diagnostic_plots(df, output_dir)
    
    # Generate final report
    generate_quality_report(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - quality_report.txt")
    print("  - feature_distributions.png")
    print("  - correlation_heatmap.png")
    print("  - boxplots_outliers.png")


if __name__ == "__main__":
    main()