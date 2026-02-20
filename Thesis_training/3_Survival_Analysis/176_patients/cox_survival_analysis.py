"""
Cox Proportional Hazards Survival Analysis for IPF Progression Prediction
Uses CNN embeddings + handcrafted features + demographics
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor

# Feature columns
HAND_FEATURE_COLS = [
    'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60', 
    'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
]

DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']

# Configuration
CONFIG = {
    "survival_data_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\patient_event_slice.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\Fold\survival_folds_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "output_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\results_cnn_univariate"),
    
    # Feature extraction
    'backbone': 'resnet50',
    'pooling_type': 'mean',
    'image_size': (224, 224),
    
    # Cox model parameters (FIXED - NO TUNING)
    'penalizer': 0.5,    # Fixed regularization
    'l1_ratio': 0.0,     # Pure Ridge (L2 only)
    'n_select': 50,      # Number of features to select via univariate Cox (30-50 recommended)
}


class SurvivalDataLoader:
    """Load and prepare data for Cox survival analysis"""
    
    def __init__(self, config):
        self.config = config
        self.cnn_extractor = None
        self.cnn_features_cache = None  # Cache for CNN features
        
    def load_survival_data(self):
        """Load patient survival data (event + time)"""
        df = pd.read_csv(self.config['survival_data_path'])
        print(f"\n{'='*70}")
        print(f"LOADED SURVIVAL DATA")
        print(f"{'='*70}")
        print(f"Total patients: {len(df)}")
        print(f"Events (progression): {df['event'].sum()}")
        print(f"Censored: {(df['event']==0).sum()}")
        print(f"Time range: {df['time'].min()}-{df['time'].max()} weeks")
        return df
    
    def load_handcrafted_features(self, patient_ids):
        """Load handcrafted radiological features"""
        df = pd.read_csv(self.config['patient_features_path'])
        df = df[df['Patient'].isin(patient_ids)]
        
        # Select only handcrafted features
        feature_cols = ['Patient'] + HAND_FEATURE_COLS
        df = df[feature_cols]
        
        print(f"\nLoaded handcrafted features: {len(HAND_FEATURE_COLS)} features")
        return df
    
    def load_demographics(self, patient_ids):
        """Load demographic features"""
        df = pd.read_csv(self.config['train_csv_path'])
        df = df[df['Patient'].isin(patient_ids)]
        
        # Select demographics
        demo_cols_available = ['Patient'] + [c for c in DEMO_FEATURE_COLS if c in df.columns]
        df = df[demo_cols_available].drop_duplicates(subset=['Patient'])
        
        # Encode categorical variables
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
        
        if 'SmokingStatus' in df.columns:
            # Ex-smoker: 0, Never smoked: 1, Currently smokes: 2
            smoking_map = {'Ex-smoker': 0, 'Never smoked': 1, 'Currently smokes': 2}
            df['SmokingStatus'] = df['SmokingStatus'].map(smoking_map)
        
        print(f"Loaded demographics: {', '.join([c for c in DEMO_FEATURE_COLS if c in df.columns])}")
        return df
    
    def extract_cnn_features(self, patient_ids):
        """Extract CNN embeddings from CT scans using efficient batch processing"""
        
        # Check cache first
        if self.cnn_features_cache is not None:
            print(f"\n{'='*70}")
            print(f"USING CACHED CNN FEATURES")
            print(f"{'='*70}")
            # Filter to requested patients
            cached_df = self.cnn_features_cache[self.cnn_features_cache['Patient'].isin(patient_ids)]
            print(f"✓ Retrieved {len(cached_df)} patients from cache")
            return cached_df
        
        print(f"\n{'='*70}")
        print(f"EXTRACTING CNN FEATURES")
        print(f"{'='*70}")
        
        # Build patient_data structure directly from npy files
        import glob
        import os
        
        patient_data = {}
        npy_dir = self.config['ct_scan_path']
        
        for patient_id in patient_ids:
            patient_npy_folder = os.path.join(npy_dir, patient_id)
            if not os.path.exists(patient_npy_folder):
                continue
                
            npy_files = sorted(glob.glob(os.path.join(patient_npy_folder, "*.npy")))
            if not npy_files:
                continue
            
            patient_data[patient_id] = {
                'slices': npy_files,
                'n_slices': len(npy_files),
                'gt_has_progressed': 0  # Dummy value - not used in survival analysis
            }
        
        print(f"Found {len(patient_data)} patients with CT scans")
        print(f"Extracting features for {len(patient_data)} patients")
        
        # Initialize CNN extractor
        if self.cnn_extractor is None:
            from utilities import CNNFeatureExtractor
            self.cnn_extractor = CNNFeatureExtractor(
                model_name=self.config['backbone']
            )
        
        # Extract slice-level features using batch processing
        slice_features_df = self.cnn_extractor.extract_features_patient_grouping(
            patient_data=patient_data,
            patients_per_batch=4,
            save_path=None  # Don't save intermediate results
        )
        
        # Aggregate to patient level using mean pooling
        print(f"\nAggregating slice features to patient level (mean pooling)...")
        cnn_feature_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
        
        patient_features = []
        valid_patients = []
        
        for patient_id in patient_ids:
            patient_slices = slice_features_df[slice_features_df['patient_id'] == patient_id]
            
            if len(patient_slices) > 0:
                # Mean pooling across slices
                mean_features = patient_slices[cnn_feature_cols].mean(axis=0).values
                patient_features.append(mean_features)
                valid_patients.append(patient_id)
            else:
                print(f"⚠️  No features for {patient_id}")
        
        # Create DataFrame
        cnn_features = np.array(patient_features)
        df = pd.DataFrame(cnn_features, columns=cnn_feature_cols)
        df.insert(0, 'Patient', valid_patients)
        
        print(f"✓ Extracted CNN features for {len(valid_patients)} patients")
        print(f"  Feature dimension: {cnn_features.shape[1]}")
        
        return df
            
    
    
    def prepare_full_dataset(self, use_cnn=True, use_hand=True, use_demo=True):
        """Prepare complete dataset with all features"""
        print(f"\n{'='*70}")
        print(f"PREPARING DATASET")
        print(f"{'='*70}")
        print(f"Features: CNN={use_cnn}, Handcrafted={use_hand}, Demographics={use_demo}")
        
        # Load survival data
        survival_df = self.load_survival_data()
        patient_ids = survival_df['Patient'].tolist()
        
        # Start with survival data
        df = survival_df.copy()
        
        # Add features based on configuration
        if use_cnn:
            cnn_df = self.extract_cnn_features(patient_ids)
            df = df.merge(cnn_df, on='Patient', how='left')
            print(f"✓ Added CNN features")
        
        if use_hand:
            hand_df = self.load_handcrafted_features(patient_ids)
            df = df.merge(hand_df, on='Patient', how='left')
            print(f"✓ Added handcrafted features")
        
        if use_demo:
            demo_df = self.load_demographics(patient_ids)
            df = df.merge(demo_df, on='Patient', how='left')
            print(f"✓ Added demographics")
        
        # Remove patients with missing data
        print(f"\nPatients before dropna: {len(df)}")
        n_before = len(df)
        df = df.dropna()
        print(f"Patients after dropna: {len(df)}")
        print(f"Dropped: {n_before - len(df)} patients with missing data")
        print(f"\n✓ Final dataset: {len(df)} patients")
        
        return df


class CoxSurvivalAnalyzer:
    """Cox Proportional Hazards model with cross-validation"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
    
    def select_features_univariate(self, train_features_scaled, train_df, n_select=50):
        """
        Univariate Cox feature selection on training data.
        Fits a Cox model for each feature individually and ranks by p-value.
        """
        from lifelines import CoxPHFitter
        
        print(f"\n{'='*70}")
        print(f"UNIVARIATE FEATURE SELECTION")
        print(f"{'='*70}")
        
        feature_cols = train_features_scaled.columns.tolist()
        feature_pvalues = []
        
        # Test each feature individually
        for i, feat in enumerate(feature_cols):
            if (i + 1) % 500 == 0:
                print(f"  Testing feature {i+1}/{len(feature_cols)}...")
            
            # Create temporary dataframe with one feature
            temp_df = pd.DataFrame({
                feat: train_features_scaled[feat].values,
                'time': train_df['time'].values,
                'event': train_df['event'].values
            })
            
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(temp_df, duration_col='time', event_col='event', show_progress=False)
                p_val = cph.summary.loc[feat, 'p']
                feature_pvalues.append({'feature': feat, 'p_value': p_val})
            except:
                # If fitting fails, assign worst p-value
                feature_pvalues.append({'feature': feat, 'p_value': 1.0})
        
        # Sort by p-value and select top features
        feature_df = pd.DataFrame(feature_pvalues).sort_values('p_value')
        selected_features = feature_df['feature'].head(n_select).tolist()
        
        print(f"✓ Selected {len(selected_features)} features with lowest p-values")
        print(f"  P-value range: [{feature_df['p_value'].iloc[0]:.2e}, {feature_df['p_value'].iloc[n_select-1]:.2e}]")
        print(f"  Features: {selected_features[:5]}... (showing first 5)")
        
        return selected_features, feature_df
        
    def fit_fold(self, train_df, val_df, test_df, fold_num):
        """Fit Cox model on one fold with train/val/test splits"""
        print(f"\n{'='*70}")
        print(f"FOLD {fold_num}")
        print(f"{'='*70}")
        print(f"Train: {len(train_df)} patients, Val: {len(val_df)} patients, Test: {len(test_df)} patients")
        
        # Prepare data
        feature_cols = [c for c in train_df.columns if c not in ['Patient', 'event', 'time']]
        
        # Normalize features using training set statistics
        scaler = StandardScaler()
        train_features = train_df[feature_cols].copy()
        val_features = val_df[feature_cols].copy()
        test_features = test_df[feature_cols].copy()
        
        train_features_scaled = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=feature_cols,
            index=train_df.index
        )
        
        val_features_scaled = pd.DataFrame(
            scaler.transform(val_features),
            columns=feature_cols,
            index=val_df.index
        )
        
        test_features_scaled = pd.DataFrame(
            scaler.transform(test_features),
            columns=feature_cols,
            index=test_df.index
        )
        
        # Apply univariate feature selection for CNN features (high-dimensional)
        # Keep all handcrafted + demographics features (low-dimensional)
        feature_cols_used = feature_cols
        selected_features = None
        
        if any('cnn_feature_' in col for col in feature_cols):
            # CNN features present - separate CNN from non-CNN features
            cnn_cols = [c for c in feature_cols if 'cnn_feature_' in c]
            non_cnn_cols = [c for c in feature_cols if 'cnn_feature_' not in c]
            
            n_select = self.config.get('n_select', 50)
            print(f"Found {len(cnn_cols)} CNN features and {len(non_cnn_cols)} non-CNN features")
            print(f"Applying univariate selection to CNN features only (top {n_select})...")
            print(f"Keeping all {len(non_cnn_cols)} non-CNN features (handcrafted + demographics)")
            
            # Apply univariate selection ONLY to CNN features
            cnn_features_scaled = train_features_scaled[cnn_cols]
            selected_cnn_features, feature_ranking = self.select_features_univariate(
                cnn_features_scaled, train_df, n_select=n_select
            )
            
            # Combine selected CNN + all non-CNN features
            selected_features = selected_cnn_features + non_cnn_cols
            print(f"Final feature set: {len(selected_cnn_features)} CNN + {len(non_cnn_cols)} non-CNN = {len(selected_features)} total")
            
            # Select these features for all sets
            train_cox = train_features_scaled[selected_features].copy()
            train_cox['time'] = train_df['time'].values
            train_cox['event'] = train_df['event'].values
            
            val_cox = val_features_scaled[selected_features].copy()
            val_cox['time'] = val_df['time'].values
            val_cox['event'] = val_df['event'].values
            
            test_cox = test_features_scaled[selected_features].copy()
            test_cox['time'] = test_df['time'].values
            test_cox['event'] = test_df['event'].values
            
            feature_cols_used = selected_features
        else:
            # No CNN features - use scaled features directly
            print(f"Using {len(feature_cols)} features directly (no selection needed)")
            
            train_cox = train_features_scaled.copy()
            train_cox['time'] = train_df['time'].values
            train_cox['event'] = train_df['event'].values
            
            val_cox = val_features_scaled.copy()
            val_cox['time'] = val_df['time'].values
            val_cox['event'] = val_df['event'].values
            
            test_cox = test_features_scaled.copy()
            test_cox['time'] = test_df['time'].values
            test_cox['event'] = test_df['event'].values
            
            feature_cols_used = feature_cols
        
        # Fit Cox model
        print(f"Fitting Cox model with {len(feature_cols_used)} features...")
        cph = CoxPHFitter(
            penalizer=self.config['penalizer'],
            l1_ratio=self.config['l1_ratio']
        )
        
        try:
            cph.fit(train_cox, duration_col='time', event_col='event', show_progress=True)
            
            # Check feature selection (non-zero coefficients)
            nonzero = (cph.params_.abs() > 1e-6).sum()
            print(f"Non-zero coefficients: {nonzero}/{len(cph.params_)}")
            
            # Evaluate on validation and test sets
            train_ci = cph.concordance_index_
            val_ci = concordance_index(
                val_cox['time'], 
                cph.predict_partial_hazard(val_cox.drop(['time', 'event'], axis=1)),
                val_cox['event']
            )
            test_ci = concordance_index(
                test_cox['time'], 
                cph.predict_partial_hazard(test_cox.drop(['time', 'event'], axis=1)),
                test_cox['event']
            )
            
            print(f"Train C-index: {train_ci:.4f}")
            print(f"Val C-index: {val_ci:.4f}")
            print(f"Test C-index: {test_ci:.4f}")
            
            # Get risk scores (use prepared dataframes without time/event columns)
            train_risk = cph.predict_partial_hazard(train_cox.drop(['time', 'event'], axis=1))
            val_risk = cph.predict_partial_hazard(val_cox.drop(['time', 'event'], axis=1))
            test_risk = cph.predict_partial_hazard(test_cox.drop(['time', 'event'], axis=1))
            
            # Store results
            fold_results = {
                'fold': fold_num,
                'model': cph,
                'scaler': scaler,
                'feature_ranking': feature_ranking if any('cnn_feature_' in col for col in feature_cols) else None,
                'selected_features': selected_features if any('cnn_feature_' in col for col in feature_cols) else feature_cols_used,
                'train_ci': train_ci,
                'val_ci': val_ci,
                'test_ci': test_ci,
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df,
                'train_risk': train_risk,
                'val_risk': val_risk,
                'test_risk': test_risk,
                'feature_cols': feature_cols_used,
                'original_feature_count': len(feature_cols)
            }
            
            return fold_results
            
        except Exception as e:
            print(f"❌ Error fitting Cox model: {e}")
            return None
    
    def plot_kaplan_meier(self, fold_results, output_dir):
        """Plot Kaplan-Meier curves stratified by risk score"""
        fold_num = fold_results['fold']
        val_df = fold_results['val_df']
        val_risk = fold_results['val_risk']
        
        # Split into high/low risk groups
        risk_median = val_risk.median()
        high_risk = val_risk >= risk_median
        low_risk = val_risk < risk_median
        
        # Fit Kaplan-Meier
        kmf = KaplanMeierFitter()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # High risk group
        kmf.fit(
            val_df.loc[high_risk, 'time'],
            val_df.loc[high_risk, 'event'],
            label='High Risk'
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color='red')
        
        # Low risk group
        kmf.fit(
            val_df.loc[low_risk, 'time'],
            val_df.loc[low_risk, 'event'],
            label='Low Risk'
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color='blue')
        
        ax.set_xlabel('Time (weeks)', fontsize=12)
        ax.set_ylabel('Progression-free probability', fontsize=12)
        ax.set_title(f'Fold {fold_num} - Kaplan-Meier Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fold_{fold_num}_kaplan_meier.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved Kaplan-Meier plot")
    
    def plot_hazard_ratios(self, fold_results, output_dir, top_n=20):
        """Plot hazard ratios for top features"""
        fold_num = fold_results['fold']
        cph = fold_results['model']
        
        # Get hazard ratios
        hr_summary = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
        hr_summary = hr_summary.sort_values('exp(coef)', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(hr_summary))
        hazard_ratios = hr_summary['exp(coef)'].values
        lower_ci = hr_summary['exp(coef) lower 95%'].values
        upper_ci = hr_summary['exp(coef) upper 95%'].values
        
        # Plot hazard ratios with CI
        ax.errorbar(
            hazard_ratios, y_pos,
            xerr=[hazard_ratios - lower_ci, upper_ci - hazard_ratios],
            fmt='o', color='steelblue', markersize=6, capsize=4
        )
        
        # Add vertical line at HR=1
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='HR = 1 (no effect)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hr_summary.index, fontsize=9)
        ax.set_xlabel('Hazard Ratio', fontsize=12)
        ax.set_title(f'Fold {fold_num} - Top {top_n} Hazard Ratios', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fold_{fold_num}_hazard_ratios.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved hazard ratios plot")
    
    def plot_risk_distribution(self, fold_results, output_dir):
        """Plot risk score distribution for events vs non-events"""
        fold_num = fold_results['fold']
        val_df = fold_results['val_df']
        val_risk = fold_results['val_risk']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        events = val_df['event'] == 1
        no_events = val_df['event'] == 0
        
        ax.hist(val_risk[events], bins=20, alpha=0.6, label='Progression', color='red')
        ax.hist(val_risk[no_events], bins=20, alpha=0.6, label='No progression', color='blue')
        
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Fold {fold_num} - Risk Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fold_{fold_num}_risk_distribution.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved risk distribution plot")
    
    def run_cross_validation(self, df, kfold_splits, use_cnn=True, use_hand=True, use_demo=True, experiment_name=None):
        """Run Cox analysis with K-fold cross-validation"""
        print(f"\n{'='*80}")
        print(f"COX SURVIVAL ANALYSIS - CROSS VALIDATION")
        print(f"{'='*80}")
        
        # Create output directory
        if experiment_name:
            config_name = experiment_name
        else:
            pen_str = f"pen{self.config['penalizer']:.3f}".replace('.', '_')
            l1_str = f"l1_{self.config['l1_ratio']:.2f}".replace('.', '_')
            config_name = f"cnn{int(use_cnn)}_hand{int(use_hand)}_demo{int(use_demo)}_{pen_str}_{l1_str}"
        
        output_dir = self.config['output_dir'] / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for fold_num in range(1, 6):
            train_ids = kfold_splits[f'fold_{fold_num}']['train']
            val_ids = kfold_splits[f'fold_{fold_num}']['val']
            test_ids = kfold_splits[f'fold_{fold_num}'].get('test', val_ids)  # Fallback to val if no test
            
            train_df = df[df['Patient'].isin(train_ids)].reset_index(drop=True)
            val_df = df[df['Patient'].isin(val_ids)].reset_index(drop=True)
            test_df = df[df['Patient'].isin(test_ids)].reset_index(drop=True)
            
            # Fit model
            fold_results = self.fit_fold(train_df, val_df, test_df, fold_num)
            
            if fold_results is not None:
                # Generate plots
                fold_dir = output_dir / f'fold_{fold_num}'
                fold_dir.mkdir(exist_ok=True)
                
                self.plot_kaplan_meier(fold_results, fold_dir)
                self.plot_hazard_ratios(fold_results, fold_dir)
                self.plot_risk_distribution(fold_results, fold_dir)
                
                all_results.append(fold_results)
        
        # Aggregate results
        self.save_summary(all_results, output_dir)
        
        return all_results
    
    def save_summary(self, all_results, output_dir):
        """Save summary of all folds"""
        if not all_results:
            print("\n⚠️  No successful folds to summarize")
            return
        
        summary_df = pd.DataFrame([
            {
                'Fold': r['fold'],
                'Train C-index': r['train_ci'],
                'Val C-index': r['val_ci'],
                'Test C-index': r['test_ci'],
                'N_train': len(r['train_df']),
                'N_val': len(r['val_df']),
                'N_test': len(r['test_df'])
            }
            for r in all_results
        ])
        
        # Add mean and std
        mean_row = {
            'Fold': 'Mean',
            'Train C-index': summary_df['Train C-index'].mean(),
            'Val C-index': summary_df['Val C-index'].mean(),
            'Test C-index': summary_df['Test C-index'].mean(),
            'N_train': summary_df['N_train'].mean(),
            'N_val': summary_df['N_val'].mean(),
            'N_test': summary_df['N_test'].mean()
        }
        
        std_row = {
            'Fold': 'Std',
            'Train C-index': summary_df['Train C-index'].std(),
            'Val C-index': summary_df['Val C-index'].std(),
            'Test C-index': summary_df['Test C-index'].std(),
            'N_train': summary_df['N_train'].std(),
            'N_val': summary_df['N_val'].std(),
            'N_test': summary_df['N_test'].std()
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        
        # Save to CSV
        summary_df.to_csv(output_dir / 'cross_validation_summary.csv', index=False)
        
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main execution"""
    print(f"\n{'='*80}")
    print(f"COX PROPORTIONAL HAZARDS SURVIVAL ANALYSIS")
    print(f"{'='*80}")
    
    # Load data
    data_loader = SurvivalDataLoader(CONFIG)
    
    # Load K-fold splits
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"✓ Loaded K-fold splits: {len(kfold_splits)} folds")
    
    # Prepare dataset
    df = data_loader.prepare_full_dataset(
        use_cnn=True,
        use_hand=True,
        use_demo=True
    )
    
    # Run Cox analysis
    analyzer = CoxSurvivalAnalyzer(CONFIG)
    results = analyzer.run_cross_validation(
        df, kfold_splits,
        use_cnn=True,
        use_hand=True,
        use_demo=True
    )
    print(f"\n{'='*80}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
