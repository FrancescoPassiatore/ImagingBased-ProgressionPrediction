"""
ML Baseline with PCA Feature Reduction

Addresses the curse of dimensionality by reducing CNN features 
from 2048 dimensions to a manageable number using PCA.

Compares performance with original high-dimensional features.
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
from sklearn.decomposition import PCA
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
    "results_dir": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\ml_baseline_pca_results"),
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
    """Aggregate CNN features from slices to patient level"""
    print(f"\n{'='*70}")
    print(f"AGGREGATING CNN FEATURES PER PATIENT ({aggregation.upper()})")
    print("="*70)
    
    cnn_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
    print(f"CNN features: {len(cnn_cols)} dimensions")
    
    grouped = slice_features_df.groupby('patient_id')
    
    if aggregation == 'max':
        agg_features = grouped[cnn_cols].max()
        agg_features.columns = [f'{c}_max' for c in cnn_cols]
    elif aggregation == 'mean':
        agg_features = grouped[cnn_cols].mean()
        agg_features.columns = [f'{c}_mean' for c in cnn_cols]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    labels = grouped['gt_has_progressed'].first()
    agg_features['label'] = labels
    agg_features = agg_features.reset_index()
    
    print(f"Aggregated features shape: {agg_features.shape}")
    
    return agg_features


def load_and_merge_all_features(cnn_features_df: pd.DataFrame,
                                patient_features_path: Path,
                                train_csv_path: Path,
                                include_hand: bool = True,
                                include_demo: bool = True) -> pd.DataFrame:
    """Merge CNN features with hand-crafted features and demographics"""
    print(f"\n{'='*70}")
    print("MERGING FEATURES")
    print("="*70)
    
    merged_df = cnn_features_df.copy()
    
    if include_hand:
        patient_features_df = pd.read_csv(patient_features_path)
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
    
    if include_demo:
        train_df = pd.read_csv(train_csv_path)
        demo_cols = ['Patient'] + [c for c in DEMO_FEATURE_COLS if c in train_df.columns]
        demographics_df = train_df[demo_cols].copy()
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
        print(f"  ✓ Sex encoded")
    
    if 'SmokingStatus' in merged_df.columns:
        smoking_map = {'Ex-smoker': 0, 'Never smoked': 1, 'Currently smokes': 2}
        merged_df['SmokingStatus'] = merged_df['SmokingStatus'].map(smoking_map)
        print(f"  ✓ SmokingStatus encoded")
    
    # Handle missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'label']
    
    for col in numeric_cols:
        if merged_df[col].isnull().any():
            merged_df[col].fillna(merged_df[col].median(), inplace=True)
    
    print(f"\nFinal merged features: {merged_df.shape}")
    
    return merged_df


def apply_pca_to_cnn_features(features_df: pd.DataFrame, 
                              n_components: int,
                              train_ids: List[str]) -> Tuple[pd.DataFrame, PCA, List[str], List[str]]:
    """
    Apply PCA to CNN features, fit on training set only
    
    Returns:
        - Transformed dataframe
        - Fitted PCA object
        - List of CNN feature columns
        - List of non-CNN feature columns
    """
    # Identify CNN and non-CNN columns
    cnn_cols = [c for c in features_df.columns if 'cnn_feature_' in c]
    non_cnn_cols = [c for c in features_df.columns if c not in cnn_cols and c not in ['patient_id', 'label']]
    
    # Separate training data for fitting PCA
    train_df = features_df[features_df['patient_id'].isin(train_ids)]
    
    # Fit PCA on training CNN features only
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_df[cnn_cols].values)
    
    # Transform all CNN features
    cnn_transformed = pca.transform(features_df[cnn_cols].values)
    
    # Create new dataframe with PCA features
    pca_cols = [f'pca_{i}' for i in range(n_components)]
    pca_df = pd.DataFrame(cnn_transformed, columns=pca_cols, index=features_df.index)
    
    # Combine with non-CNN features
    result_df = pd.concat([
        features_df[['patient_id', 'label']],
        pca_df,
        features_df[non_cnn_cols]
    ], axis=1)
    
    explained_var = pca.explained_variance_ratio_.sum()
    
    print(f"\n  PCA: {len(cnn_cols)} → {n_components} components")
    print(f"  Explained variance: {explained_var:.2%}")
    
    return result_df, pca, cnn_cols, non_cnn_cols


def get_ml_models() -> Dict:
    """Define ML models to test"""
    models = {
        'LogisticRegression_L2': LogisticRegression(
            penalty='l2',
            C=1.0,
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


def train_with_pca_variants(features_df: pd.DataFrame,
                           kfold_splits: Dict,
                           results_dir: Path):
    """
    Train ML models with different PCA configurations
    """
    print("\n" + "="*70)
    print("TRAINING ML MODELS WITH PCA FEATURE REDUCTION")
    print("="*70)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test different numbers of PCA components
    pca_configs = [
        {'n_components': 30, 'name': 'pca_30'},
        {'n_components': 50, 'name': 'pca_50'},
        {'n_components': 100, 'name': 'pca_100'},
        {'n_components': 150, 'name': 'pca_150'},
    ]
    
    all_results = []
    
    models = get_ml_models()
    
    for pca_config in pca_configs:
        n_components = pca_config['n_components']
        config_name = pca_config['name']
        
        print(f"\n{'='*70}")
        print(f"PCA CONFIGURATION: {n_components} components")
        print("="*70)
        
        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            
            for fold_key in sorted(kfold_splits.keys()):
                if isinstance(fold_key, str):
                    fold_idx = int(fold_key.split('_')[1])
                else:
                    fold_idx = fold_key
                
                fold_data = kfold_splits[fold_key]
                
                train_ids = fold_data['train']
                val_ids = fold_data['val']
                test_ids = fold_data['test']
                
                # Apply PCA (fit on training set only)
                pca_features_df, pca_obj, _, _ = apply_pca_to_cnn_features(
                    features_df, 
                    n_components=n_components,
                    train_ids=train_ids
                )
                
                # Prepare datasets
                train_df = pca_features_df[pca_features_df['patient_id'].isin(train_ids)]
                val_df = pca_features_df[pca_features_df['patient_id'].isin(val_ids)]
                test_df = pca_features_df[pca_features_df['patient_id'].isin(test_ids)]
                
                feature_cols = [c for c in pca_features_df.columns if c not in ['patient_id', 'label']]
                
                X_train = train_df[feature_cols].values
                y_train = train_df['label'].values
                X_val = val_df[feature_cols].values
                y_val = val_df['label'].values
                X_test = test_df[feature_cols].values
                y_test = test_df['label'].values
                
                # Standardize features (EXCLUDING categorical: Sex, SmokingStatus)
                # Match normalization strategy from progression_ablation
                categorical_cols = []
                continuous_cols_idx = []
                for i, col in enumerate(feature_cols):
                    if col in ['Sex', 'SmokingStatus']:
                        categorical_cols.append(i)
                    else:
                        continuous_cols_idx.append(i)
                
                # Scale only continuous features (PCA features + Age + hand-crafted)
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
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Find optimal threshold
                optimal_threshold = find_optimal_threshold(y_val, y_val_proba)
                
                # Evaluate
                val_metrics = evaluate_model(y_val, y_val_proba, 0.5)
                test_metrics = evaluate_model(y_test, y_test_proba, optimal_threshold)
                
                all_results.append({
                    'config': config_name,
                    'n_pca_components': n_components,
                    'model': model_name,
                    'fold': fold_idx,
                    'n_features': len(feature_cols),
                    'val_auc': val_metrics['auc'],
                    'optimal_threshold': optimal_threshold,
                    'test_auc': test_metrics['auc'],
                    'test_acc': test_metrics['accuracy'],
                    'test_prec': test_metrics['precision'],
                    'test_rec': test_metrics['recall'],
                    'test_f1': test_metrics['f1'],
                })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "pca_all_results.csv", index=False)
    
    # Create summary
    summary = results_df.groupby(['config', 'n_pca_components', 'model']).agg({
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'n_features': 'first'
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.sort_values('test_auc_mean', ascending=False)
    
    summary.to_csv(results_dir / "pca_summary.csv", index=False)
    
    print("\n" + "="*70)
    print("PCA RESULTS SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
    
    # Plot comparison
    plot_pca_comparison(results_df, results_dir)
    
    return results_df, summary


def plot_pca_comparison(results_df: pd.DataFrame, save_dir: Path):
    """Plot comparison across different PCA configurations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC by PCA components for each model
    ax = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        summary = model_data.groupby('n_pca_components').agg({
            'test_auc': ['mean', 'std']
        }).reset_index()
        
        ax.errorbar(summary['n_pca_components'], 
                   summary['test_auc']['mean'],
                   yerr=summary['test_auc']['std'],
                   marker='o', label=model, linewidth=2, capsize=5)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title('AUC vs PCA Components', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.8)
    
    # 2. F1 Score by PCA components
    ax = axes[0, 1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        summary = model_data.groupby('n_pca_components').agg({
            'test_f1': ['mean', 'std']
        }).reset_index()
        
        ax.errorbar(summary['n_pca_components'], 
                   summary['test_f1']['mean'],
                   yerr=summary['test_f1']['std'],
                   marker='s', label=model, linewidth=2, capsize=5)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Test F1-Score', fontsize=12)
    ax.set_title('F1-Score vs PCA Components', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Best configuration comparison
    ax = axes[1, 0]
    best_configs = results_df.groupby(['model', 'n_pca_components'])['test_auc'].mean().reset_index()
    best_by_model = best_configs.loc[best_configs.groupby('model')['test_auc'].idxmax()]
    
    x = np.arange(len(best_by_model))
    bars = ax.bar(x, best_by_model['test_auc'], alpha=0.7, color='steelblue')
    
    # Add component count on bars
    for i, (bar, row) in enumerate(zip(bars, best_by_model.iterrows())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'n={int(row[1]["n_pca_components"])}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Best Test AUC', fontsize=12)
    ax.set_title('Best PCA Configuration per Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(best_by_model['model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # 4. Variance across folds
    ax = axes[1, 1]
    variance_data = results_df.groupby(['model', 'n_pca_components'])['test_auc'].std().reset_index()
    
    for model in results_df['model'].unique():
        model_var = variance_data[variance_data['model'] == model]
        ax.plot(model_var['n_pca_components'], model_var['test_auc'], 
               marker='o', label=model, linewidth=2)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('AUC Std Dev Across Folds', fontsize=12)
    ax.set_title('Model Stability vs PCA Components', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('PCA Feature Reduction Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "pca_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ PCA comparison plot saved")


def compare_with_without_pca(original_results_path: Path,
                             pca_results_df: pd.DataFrame,
                             save_dir: Path):
    """Compare performance with and without PCA"""
    
    print("\n" + "="*70)
    print("COMPARING WITH/WITHOUT PCA")
    print("="*70)
    
    # Load original results
    if original_results_path.exists():
        original_df = pd.read_csv(original_results_path)
        
        # Filter to CNN-only configs for fair comparison
        original_cnn = original_df[original_df['config'].str.contains('cnn_only_max')]
        
        # Create comparison dataframe
        comparison = []
        
        # Get unique models from PCA results
        for model in pca_results_df['model'].unique():
            orig_model = original_cnn[original_cnn['model'] == model]
            pca_model = pca_results_df[pca_results_df['model'] == model]
            
            if len(orig_model) > 0:
                comparison.append({
                    'model': model,
                    'original_auc_mean': orig_model['test_auc'].mean(),
                    'original_auc_std': orig_model['test_auc'].std(),
                    'pca_auc_mean': pca_model['test_auc'].mean(),
                    'pca_auc_std': pca_model['test_auc'].std(),
                    'improvement': pca_model['test_auc'].mean() - orig_model['test_auc'].mean(),
                    'best_n_components': pca_model.groupby('n_pca_components')['test_auc'].mean().idxmax()
                })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('improvement', ascending=False)
        
        print("\nPerformance Comparison (Original vs PCA):")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv(save_dir / "pca_vs_original_comparison.csv", index=False)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['original_auc_mean'], width,
              yerr=comparison_df['original_auc_std'], 
              label='Original (2048 features)', alpha=0.7, capsize=5)
        ax.bar(x + width/2, comparison_df['pca_auc_mean'], width,
              yerr=comparison_df['pca_auc_std'],
              label='PCA (best n_components)', alpha=0.7, capsize=5)
        
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Original vs PCA-Reduced Features', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_dir / "original_vs_pca.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison plot saved")
    else:
        print(f"\n⚠ Original results not found at {original_results_path}")


def main():
    """Main pipeline"""
    
    print("\n" + "="*70)
    print("ML MODELS WITH PCA FEATURE REDUCTION")
    print("="*70)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path']
    )
    
    patient_data, _ = data_loader.get_patient_data()
    
    # Extract CNN features
    print("\n" + "="*70)
    print("EXTRACTING CNN FEATURES")
    print("="*70)
    
    import torch
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=None
    )
    
    # Load K-Fold splits
    print("\n" + "="*70)
    print("LOADING K-FOLD SPLITS")
    print("="*70)
    
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"Loaded {len(kfold_splits)} folds")
    
    # Aggregate and merge features
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    cnn_agg_df = aggregate_cnn_features_per_patient(slice_features_df, aggregation='max')
    
    # Add hand-crafted and demographics
    full_features_df = load_and_merge_all_features(
        cnn_agg_df,
        BASE_CONFIG['patient_features_path'],
        BASE_CONFIG['train_csv_path'],
        include_hand=True,
        include_demo=True
    )
    
    # Train with PCA
    print("\n" + "="*70)
    print("TRAINING WITH PCA")
    print("="*70)
    
    pca_results_df, pca_summary = train_with_pca_variants(
        full_features_df,
        kfold_splits,
        BASE_CONFIG['results_dir']
    )
    
    # Compare with original results
    original_results_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\ml_baseline_results\feature_ablation_results.csv")
    
    compare_with_without_pca(
        original_results_path,
        pca_results_df,
        BASE_CONFIG['results_dir']
    )
    
    print("\n" + "="*70)
    print("✓ PCA ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {BASE_CONFIG['results_dir']}")


if __name__ == "__main__":
    main()
