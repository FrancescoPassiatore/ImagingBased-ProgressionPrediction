"""
Regional Feature Configurations for IPF Progression Prediction
Aligned with CNN Max Pooling Strategy

Use this file to replace your current HAND_FEATURE_COLS in ablation studies
"""

import pandas as pd
import numpy as np

# =============================================================================
# FEATURE SET CONFIGURATIONS
# =============================================================================

# ----------------------------------------------------------------------------
# STRATEGY 1: REGIONAL MAX FEATURES (RECOMMENDED - Aligns with max pooling)
# ----------------------------------------------------------------------------
HAND_FEATURE_COLS_REGIONAL_MAX = [
    # Lung volume baseline
    'approx_lung_volume',
    
    # Upper region - Maximum abnormality
    'upper_num_tissue_pixels_max',
    'upper_tissue_by_total_max',
    'upper_tissue_by_lung_max',
    'upper_mean_max',
    'upper_skewness_max',
    'upper_kurtosis_max',
    
    # Middle region - Maximum abnormality (your 30-60% sweet spot!)
    'middle_num_tissue_pixels_max',
    'middle_tissue_by_total_max',
    'middle_tissue_by_lung_max',
    'middle_mean_max',
    'middle_skewness_max',
    'middle_kurtosis_max',
    
    # Lower region - Maximum abnormality (CRITICAL for IPF - basal fibrosis)
    'lower_num_tissue_pixels_max',
    'lower_tissue_by_total_max',
    'lower_tissue_by_lung_max',
    'lower_mean_max',
    'lower_skewness_max',
    'lower_kurtosis_max',
    
    # Body composition (optional, comment out if not needed)
    'muscle_area_cm2',
    'total_fat_area_cm2',
    'muscle_to_fat_ratio',
]

# ----------------------------------------------------------------------------
# STRATEGY 2: LOWER LUNG FOCUSED (IPF-Specific - Basal Predominance)
# ----------------------------------------------------------------------------
HAND_FEATURE_COLS_LOWER_FOCUSED = [
    'approx_lung_volume',
    
    # Lower lung - comprehensive statistics
    'lower_num_tissue_pixels_mean',
    'lower_num_tissue_pixels_std',
    'lower_num_tissue_pixels_max',
    'lower_tissue_by_total_mean',
    'lower_tissue_by_total_std',
    'lower_tissue_by_total_max',
    'lower_tissue_by_lung_mean',
    'lower_tissue_by_lung_std',
    'lower_tissue_by_lung_max',
    'lower_mean_mean',
    'lower_mean_std',
    'lower_mean_max',
    'lower_skewness_mean',
    'lower_skewness_max',
    
    # Middle for comparison (your successful 30-60% range)
    'middle_tissue_by_lung_max',
    'middle_mean_max',
    
    # Body composition
    'muscle_area_cm2',
]

# ----------------------------------------------------------------------------
# STRATEGY 3: MINIMAL (Just the essentials - good for initial testing)
# ----------------------------------------------------------------------------
HAND_FEATURE_COLS_MINIMAL = [
    'approx_lung_volume',
    
    # One key feature per region (just MAX tissue ratio)
    'upper_tissue_by_lung_max',
    'middle_tissue_by_lung_max',
    'lower_tissue_by_lung_max',
    
    # Intensity
    'lower_mean_max',
    'middle_mean_max',
    
    # Body
    'muscle_area_cm2',
]

# ----------------------------------------------------------------------------
# STRATEGY 4: FULL COMPARISON (Original - Your baseline)
# ----------------------------------------------------------------------------
HAND_FEATURE_COLS_FULL_BASELINE = [
    'approx_lung_volume',
    'full_num_tissue_pixels_mean',
    'full_tissue_by_total_mean',
    'full_tissue_by_lung_mean',
    'full_mean_mean',
    'full_skewness_mean',
    'full_kurtosis_mean',
]


# =============================================================================
# COMPUTED FEATURES (Add to your dataframe BEFORE training)
# =============================================================================

def add_progression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute disease progression features from regional data.
    These capture spatial patterns characteristic of IPF.
    
    Args:
        df: DataFrame with regional features
        
    Returns:
        DataFrame with added computed features
    """
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. BASAL PREDOMINANCE (IPF hallmark)
    # -------------------------------------------------------------------------
    # IPF typically shows more severe fibrosis in lower vs upper lung
    df['basal_predominance_tissue'] = (
        df['lower_tissue_by_lung_max'] / 
        (df['upper_tissue_by_lung_max'] + 1e-6)
    )
    
    df['basal_predominance_intensity'] = (
        df['lower_mean_max'] / 
        (df['upper_mean_max'] + 1e-6)
    )
    
    # -------------------------------------------------------------------------
    # 2. DISEASE SEVERITY RANGE (Heterogeneity)
    # -------------------------------------------------------------------------
    # How much variation between worst lower and best upper?
    df['tissue_range_lower_upper'] = (
        df['lower_tissue_by_lung_max'] - 
        df['upper_tissue_by_lung_max']
    )
    
    df['tissue_range_lower_middle'] = (
        df['lower_tissue_by_lung_max'] - 
        df['middle_tissue_by_lung_max']
    )
    
    # -------------------------------------------------------------------------
    # 3. MIDDLE LUNG DOMINANCE (Your 30-60% success factor)
    # -------------------------------------------------------------------------
    # How much does middle region contribute vs overall?
    df['middle_to_full_ratio'] = (
        df['middle_tissue_by_lung_max'] / 
        (df['full_tissue_by_lung_mean'] + 1e-6)
    )
    
    # -------------------------------------------------------------------------
    # 4. OVERALL HETEROGENEITY INDEX
    # -------------------------------------------------------------------------
    # Average variability across all regions
    df['heterogeneity_index'] = (
        df['lower_tissue_by_lung_std'] + 
        df['middle_tissue_by_lung_std'] + 
        df['upper_tissue_by_lung_std']
    ) / 3.0
    
    # -------------------------------------------------------------------------
    # 5. PROGRESSIVE SCORE (Custom IPF indicator)
    # -------------------------------------------------------------------------
    # Weighted combination: lower > middle > upper (matches IPF progression)
    df['ipf_progression_score'] = (
        0.5 * df['lower_tissue_by_lung_max'] +
        0.3 * df['middle_tissue_by_lung_max'] +
        0.2 * df['upper_tissue_by_lung_max']
    )
    
    # -------------------------------------------------------------------------
    # 6. TEXTURE HETEROGENEITY (Spatial texture variation)
    # -------------------------------------------------------------------------
    if 'lower_texture_entropy_max' in df.columns:
        df['texture_heterogeneity'] = (
            df['lower_texture_entropy_max'] - 
            df['upper_texture_entropy_max']
        )
    
    return df


# =============================================================================
# ENHANCED FEATURE SETS (Include computed features)
# =============================================================================

HAND_FEATURE_COLS_ENHANCED = HAND_FEATURE_COLS_REGIONAL_MAX + [
    # Computed progression features
    'basal_predominance_tissue',
    'basal_predominance_intensity',
    'tissue_range_lower_upper',
    'middle_to_full_ratio',
    'heterogeneity_index',
    'ipf_progression_score',
]

HAND_FEATURE_COLS_IPF_SIGNATURE = [
    'approx_lung_volume',
    
    # Core IPF indicators
    'lower_tissue_by_lung_max',      # Basal fibrosis
    'middle_tissue_by_lung_max',     # Your 30-60% sweet spot
    'basal_predominance_tissue',     # IPF hallmark
    'ipf_progression_score',         # Weighted progression
    
    # Supporting features
    'lower_mean_max',
    'middle_mean_max',
    'heterogeneity_index',
    'muscle_area_cm2',
]


# =============================================================================
# ABLATION STUDY CONFIGURATIONS
# =============================================================================

ABLATION_CONFIGS = {
    'baseline_full_mean': {
        'hand_features': HAND_FEATURE_COLS_FULL_BASELINE,
        'description': 'Full lung averages (original - poor alignment)',
        'expected_auc': 0.54,  # Based on your results
    },
    
    'regional_max': {
        'hand_features': HAND_FEATURE_COLS_REGIONAL_MAX,
        'description': 'Regional MAX features (aligns with pooling)',
        'expected_auc': 0.68,  # Predicted improvement
    },
    
    'lower_focused': {
        'hand_features': HAND_FEATURE_COLS_LOWER_FOCUSED,
        'description': 'Lower lung focus (IPF-specific)',
        'expected_auc': 0.66,
    },
    
    'minimal': {
        'hand_features': HAND_FEATURE_COLS_MINIMAL,
        'description': 'Minimal essential features',
        'expected_auc': 0.63,
    },
    
    'enhanced': {
        'hand_features': HAND_FEATURE_COLS_ENHANCED,
        'description': 'Regional MAX + computed progression features',
        'expected_auc': 0.70,  # Best expected performance
        'requires_computed': True,
    },
    
    'ipf_signature': {
        'hand_features': HAND_FEATURE_COLS_IPF_SIGNATURE,
        'description': 'Curated IPF-specific signature',
        'expected_auc': 0.69,
        'requires_computed': True,
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_features(df: pd.DataFrame, feature_list: list) -> tuple[list, list]:
    """
    Check which features are available in the dataframe.
    
    Args:
        df: DataFrame to check
        feature_list: List of requested features
        
    Returns:
        (available_features, missing_features)
    """
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    
    if missing:
        print(f"⚠️ Missing {len(missing)} features:")
        for f in missing:
            print(f"  - {f}")
    
    print(f"✅ Found {len(available)}/{len(feature_list)} features")
    
    return available, missing


def prepare_features_for_training(
    df: pd.DataFrame,
    config_name: str = 'regional_max',
    add_computed: bool = True
) -> tuple[pd.DataFrame, list]:
    """
    Prepare features for a specific configuration.
    
    Args:
        df: Raw dataframe with regional features
        config_name: Name of configuration from ABLATION_CONFIGS
        add_computed: Whether to compute progression features
        
    Returns:
        (prepared_df, feature_list)
    """
    df = df.copy()
    
    # Get configuration
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(ABLATION_CONFIGS.keys())}")
    
    config = ABLATION_CONFIGS[config_name]
    
    # Add computed features if needed
    if add_computed or config.get('requires_computed', False):
        print(f"Computing progression features...")
        df = add_progression_features(df)
    
    # Get feature list
    feature_list = config['hand_features']
    
    # Validate
    available, missing = validate_features(df, feature_list)
    
    if missing:
        print(f"\n⚠️ WARNING: Using only {len(available)} available features")
        feature_list = available
    
    return df, feature_list


def compare_feature_sets(df: pd.DataFrame):
    """
    Quick comparison of different feature sets.
    """
    print("\n" + "="*70)
    print("FEATURE SET COMPARISON")
    print("="*70)
    
    for config_name, config in ABLATION_CONFIGS.items():
        features = config['hand_features']
        available = [f for f in features if f in df.columns]
        
        print(f"\n{config_name.upper()}")
        print(f"  Description: {config['description']}")
        print(f"  Features: {len(available)}/{len(features)} available")
        print(f"  Expected AUC: ~{config['expected_auc']:.2f}")
        if config.get('requires_computed'):
            print(f"  ⚠️ Requires computed features")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Feature_extraction\patient_features_improved.csv")
    
    print(f"Loaded {len(df)} patients with {len(df.columns)} features")
    
    # Compare all configurations
    compare_feature_sets(df)
    
    # Prepare for specific experiment
    print("\n" + "="*70)
    print("PREPARING ENHANCED CONFIGURATION")
    print("="*70)
    
    df_prepared, features = prepare_features_for_training(
        df, 
        config_name='regional_max',
        add_computed=True
    )
    
    print(f"\n✅ Ready for training with {len(features)} features:")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")