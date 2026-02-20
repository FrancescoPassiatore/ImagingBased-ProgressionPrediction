"""
Utility Functions for Expert Ensemble System
Handles data preparation, normalization, and result aggregation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import json


def normalize_features_per_fold(
    features_df: pd.DataFrame,
    features_data: Dict,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize hand-crafted and demographic features per fold
    Prevents data leakage by fitting scaler only on training data
    
    Args:
        features_df: DataFrame with CNN features and metadata
        features_data: Dict with hand-crafted and demographic features per patient
        train_ids: Training patient IDs
        val_ids: Validation patient IDs
        test_ids: Test patient IDs
    
    Returns:
        features_df_normalized: DataFrame with normalized features added
        encoding_info: Dict with normalization parameters
    """
    # Make a copy
    df = features_df.copy()
    
    # Add hand-crafted features to dataframe
    hand_feature_names = [
        'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
        'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60',
        'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
    ]
    
    for feature in hand_feature_names:
        # Map from features_data to dataframe
        feature_map = {
            'ApproxVol_30_60': 'approx_vol',
            'Avg_NumTissuePixel_30_60': 'avg_num_tissue_pixel',
            'Avg_Tissue_30_60': 'avg_tissue',
            'Avg_Tissue_thickness_30_60': 'avg_tissue_thickness',
            'Avg_TissueByTotal_30_60': 'avg_tissue_by_total',
            'Avg_TissueByLung_30_60': 'avg_tissue_by_lung',
            'Mean_30_60': 'mean',
            'Skew_30_60': 'skew',
            'Kurtosis_30_60': 'kurtosis'
        }
        
        source_feature = feature_map[feature]
        df[feature] = df['patient_id'].map(
            lambda pid: features_data.get(pid, {}).get(source_feature, 0.0)
        )
    
    # Add demographic features
    df['Age'] = df['patient_id'].map(
        lambda pid: features_data.get(pid, {}).get('age', 65.0)
    )
    df['Sex'] = df['patient_id'].map(
        lambda pid: features_data.get(pid, {}).get('sex', 0)
    )
    df['SmokingStatus'] = df['patient_id'].map(
        lambda pid: features_data.get(pid, {}).get('smoking_status', 0)
    )
    
    # Normalize hand-crafted features (continuous)
    scaler_hand = StandardScaler()
    
    # Fit on training data only
    train_mask = df['patient_id'].isin(train_ids)
    scaler_hand.fit(df.loc[train_mask, hand_feature_names])
    
    # Transform all splits
    df_norm = df.copy()
    df_norm[hand_feature_names] = scaler_hand.transform(df[hand_feature_names])
    
    # Normalize demographic features
    # Age: continuous -> standardize
    scaler_age = StandardScaler()
    scaler_age.fit(df.loc[train_mask, ['Age']])
    df_norm['Age_normalized'] = scaler_age.transform(df[['Age']])
    
    # Sex: binary -> encode as -1, +1 (centered)
    df_norm['Sex_encoded'] = df['Sex'].map({0: -1, 1: 1})
    
    # SmokingStatus: categorical -> one-hot encode (then center)
    # Assuming: 0=Never, 1=Former, 2=Current
    smoking_dummies = pd.get_dummies(df['SmokingStatus'], prefix='SmokingStatus')
    
    # Center the one-hot encoding (subtract mean from training set)
    for col in smoking_dummies.columns:
        train_mean = smoking_dummies.loc[train_mask, col].mean()
        smoking_dummies[col] = smoking_dummies[col] - train_mean
    
    # Add to dataframe
    df_norm = pd.concat([df_norm, smoking_dummies], axis=1)
    
    # Store encoding info
    encoding_info = {
        'hand_scaler_mean': scaler_hand.mean_.tolist(),
        'hand_scaler_std': scaler_hand.scale_.tolist(),
        'hand_feature_names': hand_feature_names,
        'age_scaler_mean': float(scaler_age.mean_[0]),
        'age_scaler_std': float(scaler_age.scale_[0]),
        'smoking_columns': list(smoking_dummies.columns)
    }
    
    return df_norm, encoding_info


def prepare_lgbm_features(
    features_df: pd.DataFrame,
    patient_ids: List[str],
    hand_feature_cols: List[str],
    demo_feature_cols: List[str],
    encoding_info: Dict
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix for LightGBM
    
    Args:
        features_df: DataFrame with all features
        patient_ids: List of patient IDs to include
        hand_feature_cols: Hand-crafted feature column names
        demo_feature_cols: Demographic feature column names (original names)
        encoding_info: Dict with encoding information
    
    Returns:
        X: Feature matrix (N_patients, n_features)
        y: Labels (N_patients,)
        feature_names: List of feature names
    """
    # Filter to specified patients
    patient_df = features_df[features_df['patient_id'].isin(patient_ids)].copy()
    
    # Get one row per patient
    patient_df = patient_df.groupby('patient_id').first().reset_index()
    
    # Build feature list
    feature_names = []
    feature_values = []
    
    # Add hand-crafted features (already normalized in dataframe)
    for col in hand_feature_cols:
        if col in patient_df.columns:
            feature_values.append(patient_df[col].values)
            feature_names.append(col)
    
    # Add demographic features (use preprocessed versions)
    if 'Age' in demo_feature_cols and 'Age_normalized' in patient_df.columns:
        feature_values.append(patient_df['Age_normalized'].values)
        feature_names.append('Age')
    
    if 'Sex' in demo_feature_cols and 'Sex_encoded' in patient_df.columns:
        feature_values.append(patient_df['Sex_encoded'].values)
        feature_names.append('Sex')
    
    if 'SmokingStatus' in demo_feature_cols:
        smoking_cols = encoding_info.get('smoking_columns', [])
        for col in smoking_cols:
            if col in patient_df.columns:
                feature_values.append(patient_df[col].values)
                feature_names.append(col)
    
    # Stack features
    X = np.column_stack(feature_values)
    
    # Get labels
    y = patient_df['gt_has_progressed'].values
    
    print(f"\nPrepared LightGBM features:")
    print(f"  Patients: {len(patient_df)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Feature names: {feature_names}")
    
    return X, y, feature_names


def aggregate_results(fold_results: List[Dict]) -> Dict:
    """
    Aggregate results across folds
    
    Args:
        fold_results: List of result dicts from each fold
    
    Returns:
        summary: Dict with mean and std for all metrics
    """
    n_folds = len(fold_results)
    
    # Initialize accumulators
    metrics_to_aggregate = [
        'auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity'
    ]
    
    summary = {}
    
    # Aggregate metrics for each model
    for model_name in ['cnn', 'lgbm', 'fused']:
        for metric in metrics_to_aggregate:
            values = [fold[f'{model_name}_metrics'][metric] for fold in fold_results]
            summary[f'{model_name}_{metric}_mean'] = float(np.mean(values))
            summary[f'{model_name}_{metric}_std'] = float(np.std(values))
    
    # Aggregate threshold
    threshold_values = [fold['threshold'] for fold in fold_results]
    summary['threshold_mean'] = float(np.mean(threshold_values))
    summary['threshold_std'] = float(np.std(threshold_values))
    
    # Aggregate optimal weight (if using weighted fusion)
    weight_values = [fold['optimal_weight'] for fold in fold_results if fold.get('optimal_weight') is not None]
    if weight_values:
        summary['optimal_weight_mean'] = float(np.mean(weight_values))
        summary['optimal_weight_std'] = float(np.std(weight_values))
    
    # Aggregate correlation
    pearson_values = [fold['correlation']['pearson_r'] for fold in fold_results]
    spearman_values = [fold['correlation']['spearman_r'] for fold in fold_results]
    summary['pearson_correlation_mean'] = float(np.mean(pearson_values))
    summary['pearson_correlation_std'] = float(np.std(pearson_values))
    summary['spearman_correlation_mean'] = float(np.mean(spearman_values))
    summary['spearman_correlation_std'] = float(np.std(spearman_values))
    
    # Store per-fold results
    summary['per_fold_results'] = []
    for fold_idx, fold in enumerate(fold_results):
        fold_summary = {
            'fold': fold_idx,
            'threshold': fold['threshold'],
            'correlation_pearson': fold['correlation']['pearson_r'],
            'optimal_weight': fold.get('optimal_weight'),
            'cnn_auc': fold['cnn_metrics']['auc'],
            'cnn_f1': fold['cnn_metrics']['f1'],
            'lgbm_auc': fold['lgbm_metrics']['auc'],
            'lgbm_f1': fold['lgbm_metrics']['f1'],
            'fused_auc': fold['fused_metrics']['auc'],
            'fused_f1': fold['fused_metrics']['f1']
        }
        summary['per_fold_results'].append(fold_summary)
    
    return summary


def save_fold_results(fold_results: Dict, save_path: str):
    """
    Save fold results to JSON file
    
    Args:
        fold_results: Results dict for one fold
        save_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for key, value in fold_results.items():
        if key == 'test_predictions':
            # Save predictions separately as CSV
            pred_df = pd.DataFrame({
                'patient_id': value['patient_ids'],
                'y_true': value['y_true'],
                'p_cnn': value['p_cnn'],
                'p_lgbm': value['p_lgbm'],
                'p_fused': value['p_fused']
            })
            pred_path = str(save_path).replace('.json', '_predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            serializable_results['test_predictions_file'] = pred_path
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved fold results to: {save_path}")


def print_fold_summary(fold_results: Dict, fold_idx: int):
    """
    Print summary of fold results
    
    Args:
        fold_results: Results dict for one fold
        fold_idx: Fold index
    """
    print("\n" + "="*80)
    print(f"FOLD {fold_idx} SUMMARY")
    print("="*80)
    
    print(f"\nCorrelation (Pearson): {fold_results['correlation']['pearson_r']:.4f}")
    print(f"Optimal Threshold: {fold_results['threshold']:.4f}")
    
    print("\nTest Set Performance:")
    print(f"{'Model':<12} {'AUC':<8} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 80)
    
    for model_name, metrics_key in [('CNN', 'cnn_metrics'), ('LightGBM', 'lgbm_metrics'), ('Fusion', 'fused_metrics')]:
        metrics = fold_results[metrics_key]
        print(f"{model_name:<12} {metrics['auc']:<8.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")
    
    # Show improvement
    cnn_auc = fold_results['cnn_metrics']['auc']
    lgbm_auc = fold_results['lgbm_metrics']['auc']
    fused_auc = fold_results['fused_metrics']['auc']
    best_single = max(cnn_auc, lgbm_auc)
    improvement = ((fused_auc - best_single) / best_single) * 100
    
    print(f"\nFusion Improvement: {improvement:+.2f}% over best single expert")


def create_results_table(summary: Dict) -> pd.DataFrame:
    """
    Create a formatted results table
    
    Args:
        summary: Summary dict from aggregate_results
    
    Returns:
        df: DataFrame with formatted results
    """
    rows = []
    
    for model_name in ['CNN', 'LightGBM', 'Fusion']:
        model_key = model_name.lower()
        
        row = {
            'Model': model_name,
            'AUC': f"{summary[f'{model_key}_auc_mean']:.4f} ± {summary[f'{model_key}_auc_std']:.4f}",
            'Accuracy': f"{summary[f'{model_key}_accuracy_mean']:.4f} ± {summary[f'{model_key}_accuracy_std']:.4f}",
            'Precision': f"{summary[f'{model_key}_precision_mean']:.4f} ± {summary[f'{model_key}_precision_std']:.4f}",
            'Recall': f"{summary[f'{model_key}_recall_mean']:.4f} ± {summary[f'{model_key}_recall_std']:.4f}",
            'F1': f"{summary[f'{model_key}_f1_mean']:.4f} ± {summary[f'{model_key}_f1_std']:.4f}",
            'Specificity': f"{summary[f'{model_key}_specificity_mean']:.4f} ± {summary[f'{model_key}_specificity_std']:.4f}"
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def save_results_table(summary: Dict, save_path: str):
    """
    Save results table to CSV and LaTeX
    
    Args:
        summary: Summary dict from aggregate_results
        save_path: Base path for saving (without extension)
    """
    df = create_results_table(summary)
    
    # Save CSV
    csv_path = save_path + '.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to: {csv_path}")
    
    # Save LaTeX
    latex_path = save_path + '.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False))
    print(f"Saved LaTeX table to: {latex_path}")
    
    return df
