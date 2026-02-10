"""
Machine Learning Baseline Comparison for IPF Progression Prediction

Compares traditional ML models with deep learning approach using the same:
- Feature extractions (CNN features aggregated per patient)
- K-fold splits
- Evaluation metrics

Models tested:
1. Logistic Regression (L1, L2 regularization)
2. Random Forest
3. XGBoost
4. LightGBM
5. SVM (RBF kernel)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys
from typing import Dict, List, Tuple
import json

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader

# Configuration
BASE_CONFIG = {
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl"),
    "results_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\ml_baseline_results"),
    "backbone": 'resnet50',
}

# Hand-crafted features
HAND_FEATURE_COLS = [
    'ApproxVol_30_60',
    'Avg_NumTissuePixel_30_60',
    'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60',
    'Avg_TissueByTotal_30_60',
    'Avg_TissueByLung_30_60',
    'Mean_30_60',
    'Skew_30_60',
    'Kurtosis_30_60'
]

# Demographics
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']


def aggregate_cnn_features_per_patient(slice_features_df: pd.DataFrame, 
                                       aggregation: str = 'max') -> pd.DataFrame:
    """
    Aggregate CNN features from slices to patient level
    
    Args:
        slice_features_df: DataFrame with columns [patient_id, slice_path, gt_has_progressed, cnn_feature_0, ...]
        aggregation: 'max', 'mean', or 'both'
    
    Returns:
        DataFrame with one row per patient
    """
    print(f"\n{'='*70}")
    print(f"AGGREGATING CNN FEATURES PER PATIENT ({aggregation.upper()})")
    print("="*70)
    
    # Identify CNN feature columns
    cnn_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
    print(f"CNN features: {len(cnn_cols)} dimensions")
    
    # Group by patient
    grouped = slice_features_df.groupby('patient_id')
    
    if aggregation == 'max':
        agg_features = grouped[cnn_cols].max()
        agg_features.columns = [f'cnn_{c}_max' for c in cnn_cols]
    elif aggregation == 'mean':
        agg_features = grouped[cnn_cols].mean()
        agg_features.columns = [f'cnn_{c}_mean' for c in cnn_cols]
    elif aggregation == 'both':
        agg_max = grouped[cnn_cols].max()
        agg_max.columns = [f'cnn_{c}_max' for c in cnn_cols]
        agg_mean = grouped[cnn_cols].mean()
        agg_mean.columns = [f'cnn_{c}_mean' for c in cnn_cols]
        agg_features = pd.concat([agg_max, agg_mean], axis=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Add label
    labels = grouped['gt_has_progressed'].first()
    agg_features['label'] = labels
    
    # Reset index to have patient_id as column
    agg_features = agg_features.reset_index()
    
    print(f"Aggregated features shape: {agg_features.shape}")
    print(f"  Patients: {len(agg_features)}")
    print(f"  Features: {agg_features.shape[1] - 2}")  # Exclude patient_id and label
    
    return agg_features


def load_and_merge_all_features(cnn_features_df: pd.DataFrame,
                                patient_features_path: Path,
                                train_csv_path: Path) -> pd.DataFrame:
    """
    Merge CNN features with hand-crafted features and demographics
    """
    print(f"\n{'='*70}")
    print("MERGING ALL FEATURES")
    print("="*70)
    
    # Load patient features (hand-crafted)
    patient_features_df = pd.read_csv(patient_features_path)
    
    # Load demographics
    train_df = pd.read_csv(train_csv_path)
    demo_cols = ['Patient'] + [c for c in DEMO_FEATURE_COLS if c in train_df.columns]
    demographics_df = train_df[demo_cols].copy()
    
    # Merge everything
    merged_df = cnn_features_df.copy()
    
    # Merge hand-crafted features
    available_hand_cols = [c for c in HAND_FEATURE_COLS if c in patient_features_df.columns]
    if available_hand_cols:
        merged_df = merged_df.merge(
            patient_features_df[['Patient'] + available_hand_cols],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        merged_df.drop('Patient', axis=1, inplace=True)
        print(f"✓ Added {len(available_hand_cols)} hand-crafted features")
    
    # Merge demographics
    available_demo_cols = [c for c in demo_cols[1:] if c in demographics_df.columns]
    if available_demo_cols:
        merged_df = merged_df.merge(
            demographics_df,
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        merged_df.drop('Patient', axis=1, inplace=True)
        print(f"✓ Added {len(available_demo_cols)} demographic features")
    
    # Encode categorical features
    print(f"\nEncoding categorical features...")
    if 'Sex' in merged_df.columns:
        merged_df['Sex'] = merged_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"  ✓ Sex encoded (Male=1, Female=0)")
    
    if 'SmokingStatus' in merged_df.columns:
        # Ex-smoker=0, Never smoked=1, Currently smokes=2
        smoking_map = {'Ex-smoker': 0, 'Never smoked': 1, 'Currently smokes': 2}
        merged_df['SmokingStatus'] = merged_df['SmokingStatus'].map(smoking_map)
        print(f"  ✓ SmokingStatus encoded (Ex-smoker=0, Never=1, Current=2)")
    
    # Handle missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'label']
    
    missing_counts = merged_df[numeric_cols].isnull().sum()
    if missing_counts.any():
        print(f"\n⚠ Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count}")
        print(f"Filling with median...")
        for col in numeric_cols:
            if merged_df[col].isnull().any():
                merged_df[col].fillna(merged_df[col].median(), inplace=True)
    
    print(f"\nFinal merged features: {merged_df.shape}")
    print(f"  Total features: {len(numeric_cols)}")
    
    return merged_df


def get_ml_models() -> Dict:
    """
    Define ML models to test
    """
    models = {
        'LogisticRegression_L2': LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'LogisticRegression_L1': LogisticRegression(
            penalty='l1',
            C=0.1,
            solver='liblinear',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
    }
    
    return models


def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """Evaluate model predictions"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_ml_models_kfold(features_df: pd.DataFrame, 
                          kfold_splits: Dict,
                          results_dir: Path):
    """
    Train all ML models using k-fold cross-validation
    """
    print("\n" + "="*70)
    print("TRAINING ML MODELS - K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models = get_ml_models()
    all_results = []
    
    # Feature columns (exclude patient_id and label)
    feature_cols = [c for c in features_df.columns if c not in ['patient_id', 'label']]
    
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Models to test: {len(models)}")
    print(f"Folds: {len(kfold_splits)}")
    
    # Train each model on each fold
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print("="*70)
        
        fold_results = []
        
        for fold_idx in sorted(kfold_splits.keys()):
            fold_data = kfold_splits[fold_idx]
            
            print(f"\n--- Fold {fold_idx} ---")
            
            # Get train, val, test sets
            train_ids = fold_data['train']
            val_ids = fold_data['val']
            test_ids = fold_data['test']
            
            train_df = features_df[features_df['patient_id'].isin(train_ids)]
            val_df = features_df[features_df['patient_id'].isin(val_ids)]
            test_df = features_df[features_df['patient_id'].isin(test_ids)]
            
            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values
            X_val = val_df[feature_cols].values
            y_val = val_df['label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['label'].values
            
            # Standardize features (EXCLUDING categorical: Sex, SmokingStatus)
            # Identify categorical columns (Sex and SmokingStatus are already encoded as integers)
            categorical_cols = []
            continuous_cols_idx = []
            for i, col in enumerate(feature_cols):
                if col in ['Sex', 'SmokingStatus']:
                    categorical_cols.append(i)
                else:
                    continuous_cols_idx.append(i)
            
            # Scale only continuous features
            scaler = StandardScaler()
            if continuous_cols_idx:
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_test_scaled = X_test.copy()
                
                X_train_scaled[:, continuous_cols_idx] = scaler.fit_transform(X_train[:, continuous_cols_idx])
                X_val_scaled[:, continuous_cols_idx] = scaler.transform(X_val[:, continuous_cols_idx])
                X_test_scaled[:, continuous_cols_idx] = scaler.transform(X_test[:, continuous_cols_idx])
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Find optimal threshold on validation set
            optimal_threshold = find_optimal_threshold(y_val, y_val_proba)
            
            # Evaluate on validation with default threshold
            val_metrics_default = evaluate_model(y_val, y_val_proba, threshold=0.5)
            
            # Evaluate on test with both thresholds
            test_metrics_default = evaluate_model(y_test, y_test_proba, threshold=0.5)
            test_metrics_optimal = evaluate_model(y_test, y_test_proba, threshold=optimal_threshold)
            
            print(f"  Val AUC: {val_metrics_default['auc']:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  Test AUC (default): {test_metrics_default['auc']:.4f}")
            print(f"  Test AUC (optimal): {test_metrics_optimal['auc']:.4f}")
            
            # Store results
            fold_result = {
                'model': model_name,
                'fold': fold_idx,
                'val_auc': val_metrics_default['auc'],
                'optimal_threshold': optimal_threshold,
                'test_auc_default': test_metrics_default['auc'],
                'test_acc_default': test_metrics_default['accuracy'],
                'test_prec_default': test_metrics_default['precision'],
                'test_rec_default': test_metrics_default['recall'],
                'test_f1_default': test_metrics_default['f1'],
                'test_auc_optimal': test_metrics_optimal['auc'],
                'test_acc_optimal': test_metrics_optimal['accuracy'],
                'test_prec_optimal': test_metrics_optimal['precision'],
                'test_rec_optimal': test_metrics_optimal['recall'],
                'test_f1_optimal': test_metrics_optimal['f1'],
            }
            fold_results.append(fold_result)
        
        # Aggregate results for this model
        fold_df = pd.DataFrame(fold_results)
        
        print(f"\n{model_name} - Summary across folds:")
        print(f"  Val AUC: {fold_df['val_auc'].mean():.4f} ± {fold_df['val_auc'].std():.4f}")
        print(f"  Test AUC (optimal): {fold_df['test_auc_optimal'].mean():.4f} ± {fold_df['test_auc_optimal'].std():.4f}")
        print(f"  Test F1 (optimal): {fold_df['test_f1_optimal'].mean():.4f} ± {fold_df['test_f1_optimal'].std():.4f}")
        
        all_results.extend(fold_results)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "ml_models_all_folds.csv", index=False)
    
    # Create summary by model
    summary_results = []
    for model_name in models.keys():
        model_df = results_df[results_df['model'] == model_name]
        
        summary_results.append({
            'model': model_name,
            'val_auc_mean': model_df['val_auc'].mean(),
            'val_auc_std': model_df['val_auc'].std(),
            'test_auc_optimal_mean': model_df['test_auc_optimal'].mean(),
            'test_auc_optimal_std': model_df['test_auc_optimal'].std(),
            'test_acc_optimal_mean': model_df['test_acc_optimal'].mean(),
            'test_acc_optimal_std': model_df['test_acc_optimal'].std(),
            'test_f1_optimal_mean': model_df['test_f1_optimal'].mean(),
            'test_f1_optimal_std': model_df['test_f1_optimal'].std(),
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values('test_auc_optimal_mean', ascending=False)
    summary_df.to_csv(results_dir / "ml_models_summary.csv", index=False)
    
    print("\n" + "="*70)
    print("ML MODELS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Plot comparison
    plot_ml_comparison(summary_df, results_dir)
    
    return results_df, summary_df


def plot_ml_comparison(summary_df: pd.DataFrame, save_dir: Path):
    """
    Plot comparison of ML models
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = summary_df['model'].values
    x = np.arange(len(models))
    
    # AUC
    auc_means = summary_df['test_auc_optimal_mean'].values
    auc_stds = summary_df['test_auc_optimal_std'].values
    axes[0].bar(x, auc_means, yerr=auc_stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Test AUC', fontsize=12)
    axes[0].set_title('AUC Comparison (Optimal Threshold)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1)
    
    # Accuracy
    acc_means = summary_df['test_acc_optimal_mean'].values
    acc_stds = summary_df['test_acc_optimal_std'].values
    axes[1].bar(x, acc_means, yerr=acc_stds, capsize=5, alpha=0.7, color='seagreen')
    axes[1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1)
    
    # F1-Score
    f1_means = summary_df['test_f1_optimal_mean'].values
    f1_stds = summary_df['test_f1_optimal_std'].values
    axes[2].bar(x, f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color='coral')
    axes[2].set_ylabel('Test F1-Score', fontsize=12)
    axes[2].set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0, 1)
    
    plt.suptitle('ML Models Performance Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "ml_models_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved: {save_dir / 'ml_models_comparison.png'}")


def test_feature_combinations(slice_features_df: pd.DataFrame,
                              patient_features_path: Path,
                              train_csv_path: Path,
                              kfold_splits: Dict,
                              results_dir: Path):
    """
    Test different feature combinations (ablation study for ML)
    """
    print("\n" + "="*70)
    print("FEATURE COMBINATION ABLATION STUDY")
    print("="*70)
    
    feature_configs = {
        'cnn_only_max': {
            'use_cnn': True,
            'cnn_agg': 'max',
            'use_hand': False,
            'use_demo': False
        },
        'cnn_only_mean': {
            'use_cnn': True,
            'cnn_agg': 'mean',
            'use_hand': False,
            'use_demo': False
        },
        'hand_demo': {
            'use_cnn': False,
            'cnn_agg': None,
            'use_hand': True,
            'use_demo': True
        },
        'cnn_max_hand_demo': {
            'use_cnn': True,
            'cnn_agg': 'max',
            'use_hand': True,
            'use_demo': True
        },
    }
    
    all_results = []
    
    for config_name, config in feature_configs.items():
        print(f"\n{'='*70}")
        print(f"Configuration: {config_name}")
        print("="*70)
        
        # Prepare features based on config
        if config['use_cnn']:
            cnn_agg_df = aggregate_cnn_features_per_patient(slice_features_df, config['cnn_agg'])
        else:
            # Just patient_id and label
            cnn_agg_df = slice_features_df.groupby('patient_id')['gt_has_progressed'].first().reset_index()
            cnn_agg_df.columns = ['patient_id', 'label']
        
        if config['use_hand'] or config['use_demo']:
            features_df = load_and_merge_all_features(cnn_agg_df, patient_features_path, train_csv_path)
            
            # Remove features not in config
            cols_to_keep = ['patient_id', 'label']
            if config['use_cnn']:
                cols_to_keep.extend([c for c in features_df.columns if c.startswith('cnn_')])
            if config['use_hand']:
                cols_to_keep.extend([c for c in HAND_FEATURE_COLS if c in features_df.columns])
            if config['use_demo']:
                cols_to_keep.extend([c for c in DEMO_FEATURE_COLS if c in features_df.columns])
            
            features_df = features_df[cols_to_keep]
        else:
            features_df = cnn_agg_df
        
        # Use only best 2 models for speed
        best_models = {
            'XGBoost': get_ml_models()['XGBoost'],
            'RandomForest': get_ml_models()['RandomForest']
        }
        
        # Train models
        config_results_dir = results_dir / f"ablation_{config_name}"
        config_results_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in best_models.items():
            print(f"\n  Training {model_name}...")
            
            feature_cols = [c for c in features_df.columns if c not in ['patient_id', 'label']]
            
            for fold_key in sorted(kfold_splits.keys()):
                # Handle both string ('fold_0') and integer (0) keys
                if isinstance(fold_key, str):
                    fold_idx = int(fold_key.split('_')[1])
                else:
                    fold_idx = fold_key
                
                fold_data = kfold_splits[fold_key]
                
                train_ids = fold_data['train']
                val_ids = fold_data['val']
                test_ids = fold_data['test']
                
                train_df = features_df[features_df['patient_id'].isin(train_ids)]
                val_df = features_df[features_df['patient_id'].isin(val_ids)]
                test_df = features_df[features_df['patient_id'].isin(test_ids)]
                
                X_train = train_df[feature_cols].values
                y_train = train_df['label'].values
                X_val = val_df[feature_cols].values
                y_val = val_df['label'].values
                X_test = test_df[feature_cols].values
                y_test = test_df['label'].values
                
                # Standardize features (EXCLUDING categorical: Sex, SmokingStatus)
                categorical_cols = []
                continuous_cols_idx = []
                for i, col in enumerate(feature_cols):
                    if col in ['Sex', 'SmokingStatus']:
                        categorical_cols.append(i)
                    else:
                        continuous_cols_idx.append(i)
                
                scaler = StandardScaler()
                if continuous_cols_idx:
                    X_train_scaled = X_train.copy()
                    X_val_scaled = X_val.copy()
                    X_test_scaled = X_test.copy()
                    
                    X_train_scaled[:, continuous_cols_idx] = scaler.fit_transform(X_train[:, continuous_cols_idx])
                    X_val_scaled[:, continuous_cols_idx] = scaler.transform(X_val[:, continuous_cols_idx])
                    X_test_scaled[:, continuous_cols_idx] = scaler.transform(X_test[:, continuous_cols_idx])
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                    X_test_scaled = X_test
                
                model.fit(X_train_scaled, y_train)
                
                y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                optimal_threshold = find_optimal_threshold(y_val, y_val_proba)
                val_metrics = evaluate_model(y_val, y_val_proba, 0.5)
                test_metrics = evaluate_model(y_test, y_test_proba, optimal_threshold)
                
                all_results.append({
                    'config': config_name,
                    'model': model_name,
                    'fold': fold_idx,
                    'n_features': len(feature_cols),
                    'val_auc': val_metrics['auc'],
                    'test_auc': test_metrics['auc'],
                    'test_f1': test_metrics['f1'],
                })
    
    # Save and summarize
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "feature_ablation_results.csv", index=False)
    
    # Summary
    summary = results_df.groupby(['config', 'model']).agg({
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'n_features': 'first'
    }).reset_index()
    
    print("\n" + "="*70)
    print("FEATURE ABLATION SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
    
    summary.to_csv(results_dir / "feature_ablation_summary.csv", index=False)
    
    return results_df


def main():
    """Main pipeline"""
    
    print("\n" + "="*70)
    print("ML BASELINE COMPARISON FOR IPF PROGRESSION PREDICTION")
    print("="*70)
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path']
    )
    
    patient_data, _ = data_loader.get_patient_data()
    
    # Step 2: Extract CNN features
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CNN FEATURES")
    print("="*70)
    
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device='cuda' if __name__ == '__main__' else 'cpu'
    )
    
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=None
    )
    
    # Step 3: Load K-Fold splits
    print("\n" + "="*70)
    print("STEP 3: LOADING K-FOLD SPLITS")
    print("="*70)
    
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"Loaded {len(kfold_splits)} folds")
    
    # Step 4: Aggregate CNN features and merge with other features
    print("\n" + "="*70)
    print("STEP 4: PREPARING FEATURES")
    print("="*70)
    
    # Use max aggregation (matches DL approach better)
    cnn_agg_df = aggregate_cnn_features_per_patient(slice_features_df, aggregation='max')
    
    # Merge all features
    full_features_df = load_and_merge_all_features(
        cnn_agg_df,
        BASE_CONFIG['patient_features_path'],
        BASE_CONFIG['train_csv_path']
    )
    
    # Step 5: Train ML models
    print("\n" + "="*70)
    print("STEP 5: TRAINING ML MODELS")
    print("="*70)
    
    results_df, summary_df = train_ml_models_kfold(
        full_features_df,
        kfold_splits,
        BASE_CONFIG['results_dir']
    )
    
    # Step 6: Feature ablation study
    print("\n" + "="*70)
    print("STEP 6: FEATURE ABLATION STUDY")
    print("="*70)
    
    ablation_results = test_feature_combinations(
        slice_features_df,
        BASE_CONFIG['patient_features_path'],
        BASE_CONFIG['train_csv_path'],
        kfold_splits,
        BASE_CONFIG['results_dir']
    )
    
    print("\n" + "="*70)
    print("✓ ML BASELINE COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {BASE_CONFIG['results_dir']}")
    print("\nFiles generated:")
    print(f"  - ml_models_all_folds.csv")
    print(f"  - ml_models_summary.csv")
    print(f"  - ml_models_comparison.png")
    print(f"  - feature_ablation_results.csv")
    print(f"  - feature_ablation_summary.csv")


if __name__ == "__main__":
    main()
