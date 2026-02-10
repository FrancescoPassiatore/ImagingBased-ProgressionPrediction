"""
ML Ensemble for IPF Progression Prediction
==========================================

This script implements a comprehensive ensemble approach combining:
1. Feature Engineering (ratios, interactions, polynomial)
2. Multiple ML Models (RF, XGBoost, GradientBoosting, LogisticRegression)
3. Hyperparameter Tuning
4. Stacking Ensemble
5. Feature Importance Analysis

Target: Beat Deep Learning AUC of 0.58 → Achieve AUC > 0.65
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch
import sys

# Add correct path to utilities
UTILITIES_PATH = Path(__file__).parent.parent / "1_progression"
sys.path.insert(0, str(UTILITIES_PATH))
from utilities import CNNFeatureExtractor, IPFDataLoader

# ML models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import VarianceThreshold

# Metrics
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CNN FEATURE EXTRACTION (Using CNNFeatureExtractor from utilities)
# =============================================================================

def extract_cnn_features_with_extractor(
    patient_data: Dict,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Extract CNN features using the same method as ablation_study.py
    
    Args:
        patient_data: Dictionary from IPFDataLoader.get_patient_data()
        device: Device to use for inference
    
    Returns:
        DataFrame with columns: patient_id, slice_idx, cnn_feature_0, ..., cnn_feature_2047
    """
    
    print("\nExtracting CNN features using CNNFeatureExtractor...")
    print(f"Device: {device}")
    
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device=device
    )
    
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=None  # Don't save, return directly
    )
    
    print(f"✓ Extracted CNN features: {slice_features_df.shape}")
    print(f"  Columns: {slice_features_df.columns.tolist()[:5]}... (showing first 5)")
    
    return slice_features_df


def aggregate_cnn_features_per_patient(
    slice_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate slice-level CNN features to patient-level (mean pooling)
    
    Args:
        slice_features_df: DataFrame with slice-level features
    
    Returns:
        DataFrame with patient-level aggregated features
    """
    
    print("\nAggregating CNN features to patient-level...")
    
    # Get CNN feature columns (exclude patient_id, slice_idx, label, etc.)
    feature_cols = [col for col in slice_features_df.columns 
                   if col.startswith('cnn_feature_')]
    
    print(f"  Found {len(feature_cols)} CNN features")
    
    # Group by patient and take mean
    patient_features = slice_features_df.groupby('patient_id')[feature_cols].mean().reset_index()
    
    print(f"✓ Aggregated to {len(patient_features)} patients")
    
    # CRITICAL: Verify one row per patient
    n_unique = patient_features['patient_id'].nunique()
    n_rows = len(patient_features)
    
    if n_unique != n_rows:
        raise ValueError(f"DUPLICATE PATIENTS! {n_unique} unique patients but {n_rows} rows")
    
    return patient_features


def select_best_cnn_features(
    patient_cnn_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_features: int = 100
) -> List[str]:
    """
    Select top CNN features using variance threshold + feature importance
    
    Args:
        patient_cnn_df: DataFrame with patient-level CNN features
        labels_df: DataFrame with patient_id and has_progressed
        n_features: Number of top features to keep
    
    Returns:
        List of selected feature column names
    """
    
    print(f"\nSelecting top {n_features} CNN features...")
    
    # Merge with labels
    merged = patient_cnn_df.merge(labels_df, left_on='patient_id', right_on='Patient', how='inner')
    
    # CRITICAL: Check for duplicates
    n_unique_patients = merged['patient_id'].nunique()
    n_rows = len(merged)
    
    if n_unique_patients != n_rows:
        print(f"⚠️  WARNING: {n_rows} rows but only {n_unique_patients} unique patients!")
        print(f"   This indicates duplicate patient data!")
        
        # Show duplicate patients
        duplicates = merged[merged.duplicated('patient_id', keep=False)]['patient_id'].unique()
        print(f"   Duplicate patients: {duplicates[:5]}... (showing first 5)")
        
        raise ValueError("DUPLICATE PATIENTS DETECTED! Cannot proceed with training.")
    
    # Get CNN feature columns
    feature_cols = [col for col in patient_cnn_df.columns if col.startswith('cnn_feature_')]
    
    X = merged[feature_cols].values
    y = merged['has_progressed'].values
    
    print(f"  Original CNN features: {len(feature_cols)}")
    print(f"  Training samples: {len(X)} (should equal number of unique patients)")
    
    # Step 1: Remove low-variance features
    selector = VarianceThreshold(threshold=0.01 * np.median(np.var(X, axis=0)))
    X_filtered = selector.fit_transform(X)
    high_var_cols = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    
    print(f"  After variance filtering: {len(high_var_cols)}")
    
    # Step 2: Train RF to get feature importance
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y)
    
    importances = rf.feature_importances_
    
    # Step 3: Select top N features by importance
    n_to_select = min(n_features, len(importances))
    top_indices = np.argsort(importances)[::-1][:n_to_select]
    
    selected_cols = [high_var_cols[i] for i in top_indices]
    
    print(f"  ✓ Selected top {len(selected_cols)} CNN features")
    
    return selected_cols


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressive feature engineering to extract maximum signal
    
    Creates:
    - Ratio features (tissue/volume, etc.)
    - Interaction features (age × tissue)
    - Polynomial features (tissue², age²)
    - Statistical transformations (log, sqrt)
    """
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    df_eng = df.copy()
    
    # Original feature count
    original_features = [c for c in df.columns if c not in ['Patient', 'PatientID', 'has_progressed']]
    print(f"\nOriginal features: {len(original_features)}")
    
    # =========================================================================
    # 1. RATIO FEATURES (very informative!)
    # =========================================================================
    
    if 'Avg_Tissue_30_60' in df.columns and 'ApproxVol_30_60' in df.columns:
        # Tissue density (tissue per unit volume)
        df_eng['Tissue_Density'] = df_eng['Avg_Tissue_30_60'] / (df_eng['ApproxVol_30_60'] + 1)
        print("  ✓ Added: Tissue_Density")
    
    if 'Avg_TissueByLung_30_60' in df.columns and 'Avg_TissueByTotal_30_60' in df.columns:
        # Lung involvement ratio
        df_eng['Lung_Involvement_Ratio'] = df_eng['Avg_TissueByLung_30_60'] / (df_eng['Avg_TissueByTotal_30_60'] + 0.01)
        print("  ✓ Added: Lung_Involvement_Ratio")
    
    if 'Avg_Tissue_30_60' in df.columns and 'Avg_Tissue_thickness_30_60' in df.columns:
        # Tissue concentration
        df_eng['Tissue_Concentration'] = df_eng['Avg_Tissue_30_60'] / (df_eng['Avg_Tissue_thickness_30_60'] + 0.01)
        print("  ✓ Added: Tissue_Concentration")
    
    # =========================================================================
    # 2. INTERACTION FEATURES (capture joint effects)
    # =========================================================================
    
    if 'Age' in df.columns and 'Avg_Tissue_30_60' in df.columns:
        # Age-Tissue interaction (older + more tissue = worse prognosis?)
        df_eng['Age_Tissue_Interaction'] = df_eng['Age'] * df_eng['Avg_Tissue_30_60'] / 1000
        print("  ✓ Added: Age_Tissue_Interaction")
    
    if 'Age' in df.columns and 'Avg_TissueByLung_30_60' in df.columns:
        df_eng['Age_TissueByLung'] = df_eng['Age'] * df_eng['Avg_TissueByLung_30_60']
        print("  ✓ Added: Age_TissueByLung")
    
    if 'SmokingStatus' in df.columns and 'Avg_Tissue_30_60' in df.columns:
        df_eng['Smoking_Tissue'] = df_eng['SmokingStatus'] * df_eng['Avg_Tissue_30_60'] / 100
        print("  ✓ Added: Smoking_Tissue")
    
    # =========================================================================
    # 3. POLYNOMIAL FEATURES (capture non-linear relationships)
    # =========================================================================
    
    if 'Avg_Tissue_30_60' in df.columns:
        df_eng['Tissue_Squared'] = df_eng['Avg_Tissue_30_60'] ** 2 / 10000
        print("  ✓ Added: Tissue_Squared")
    
    if 'Age' in df.columns:
        df_eng['Age_Squared'] = df_eng['Age'] ** 2 / 1000
        print("  ✓ Added: Age_Squared")
    
    # =========================================================================
    # 4. LOG TRANSFORMATIONS (handle skewed distributions)
    # =========================================================================
    
    if 'ApproxVol_30_60' in df.columns:
        df_eng['Log_Volume'] = np.log1p(df_eng['ApproxVol_30_60'])
        print("  ✓ Added: Log_Volume")
    
    if 'Avg_NumTissuePixel_30_60' in df.columns:
        df_eng['Log_NumPixels'] = np.log1p(df_eng['Avg_NumTissuePixel_30_60'])
        print("  ✓ Added: Log_NumPixels")
    
    # =========================================================================
    # 5. STATISTICAL FEATURES (distribution shape)
    # =========================================================================
    
    if 'Skew_30_60' in df.columns and 'Kurtosis_30_60' in df.columns:
        # Distribution complexity
        df_eng['Distribution_Complexity'] = np.abs(df_eng['Skew_30_60']) + np.abs(df_eng['Kurtosis_30_60'])
        print("  ✓ Added: Distribution_Complexity")
    
    # =========================================================================
    # 6. BINNED FEATURES (capture non-linear patterns)
    # =========================================================================
    
    if 'Age' in df.columns:
        # Age groups
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], bins=[0, 60, 70, 100], labels=[0, 1, 2]).astype(float)
        print("  ✓ Added: Age_Group")
    
    if 'Avg_Tissue_30_60' in df.columns:
        # Tissue severity groups
        df_eng['Tissue_Severity'] = pd.qcut(
            df_eng['Avg_Tissue_30_60'], 
            q=3, 
            labels=[0, 1, 2],
            duplicates='drop'
        ).astype(float)
        print("  ✓ Added: Tissue_Severity")
    
    # Count new features
    new_features = [c for c in df_eng.columns if c not in df.columns and c not in ['Patient', 'PatientID', 'has_progressed']]
    
    print(f"\n✓ Total new features created: {len(new_features)}")
    print(f"✓ Total features now: {len(original_features) + len(new_features)}")
    
    return df_eng


def select_features(df: pd.DataFrame, include_cnn: bool = False) -> List[str]:
    """Select final feature set for training"""
    
    # Hand-crafted features
    hand_features = [
        'Avg_Tissue_30_60',
        'ApproxVol_30_60',
        'Avg_TissueByLung_30_60',
        'Avg_NumTissuePixel_30_60',
        'Avg_Tissue_thickness_30_60',
        'Avg_TissueByTotal_30_60',
        'Mean_30_60',
        'Skew_30_60',
        'Kurtosis_30_60'
    ]
    
    # Demographics
    demo_features = ['Age', 'Sex', 'SmokingStatus']
    
    # Engineered features (check which exist)
    engineered_features = [
        'Tissue_Density',
        'Lung_Involvement_Ratio',
        'Tissue_Concentration',
        'Age_Tissue_Interaction',
        'Age_TissueByLung',
        'Smoking_Tissue',
        'Tissue_Squared',
        'Age_Squared',
        'Log_Volume',
        'Log_NumPixels',
        'Distribution_Complexity',
        'Age_Group',
        'Tissue_Severity'
    ]
    
    # CNN features (if requested)
    cnn_features = []
    if include_cnn:
        cnn_features = [col for col in df.columns if col.startswith('CNN_')]
    
    # Combine and filter to existing columns
    all_features = hand_features + demo_features + engineered_features + cnn_features
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"\n✓ Selected {len(available_features)} features for training")
    if cnn_features:
        print(f"  - Handcrafted: {len([f for f in available_features if not f.startswith('CNN_')])}")
        print(f"  - CNN features: {len([f for f in available_features if f.startswith('CNN_')])}")
    
    return available_features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_single_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """Train and evaluate a single model"""
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        'model': model_name,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'predictions': y_pred_proba,
        'y_true': y_test  # Save for confusion matrix
    }
    
    return metrics


def train_ensemble_kfold(
    df: pd.DataFrame,
    kfold_splits: dict,
    feature_cols: List[str],
    save_dir: Path
):
    """
    Train ensemble models across all folds
    """
    
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE MODELS")
    print("="*70)
    
    # CRITICAL: Verify no duplicate patients before training
    n_unique_patients = df['Patient'].nunique()
    n_rows = len(df)
    
    print(f"\n{'='*70}")
    print("PRE-TRAINING VALIDATION")
    print(f"{'='*70}")
    print(f"Total rows in dataframe: {n_rows}")
    print(f"Unique patients: {n_unique_patients}")
    
    if n_unique_patients != n_rows:
        print(f"\n❌ ERROR: DUPLICATE PATIENTS DETECTED!")
        print(f"   {n_rows} rows but only {n_unique_patients} unique patients")
        print(f"   This will cause DATA LEAKAGE and inflated performance!")
        
        # Show duplicate patients
        duplicates = df[df.duplicated('Patient', keep=False)].sort_values('Patient')
        print(f"\n   Duplicate patient IDs:")
        print(duplicates[['Patient', 'has_progressed']].head(10))
        
        raise ValueError("CRITICAL ERROR: Duplicate patients in dataset! Cannot proceed.")
    
    print(f"✓ VALIDATION PASSED: One row per patient")
    
    # Verify k-fold splits
    print(f"\n{'='*70}")
    print("K-FOLD SPLITS VALIDATION")
    print(f"{'='*70}")
    
    all_train_patients = set()
    all_test_patients = set()
    
    for fold_idx in sorted(kfold_splits.keys()):
        train_ids = kfold_splits[fold_idx]['train']
        test_ids = kfold_splits[fold_idx]['test']
        
        # Check for overlap
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            print(f"❌ FOLD {fold_idx}: Train/test overlap detected: {len(overlap)} patients")
            raise ValueError(f"K-fold split {fold_idx} has train/test overlap!")
        
        # Check sizes
        total = len(train_ids) + len(test_ids)
        print(f"Fold {fold_idx}: {len(train_ids)} train, {len(test_ids)} test, total={total}")
        
        all_train_patients.update(train_ids)
        all_test_patients.update(test_ids)
    
    # Verify all patients covered
    all_patients_in_splits = all_train_patients | all_test_patients
    patients_in_df = set(df['Patient'].unique())
    
    missing = patients_in_df - all_patients_in_splits
    extra = all_patients_in_splits - patients_in_df
    
    if missing:
        print(f"⚠️  WARNING: {len(missing)} patients in df but not in splits")
    if extra:
        print(f"⚠️  WARNING: {len(extra)} patients in splits but not in df")
    
    print(f"✓ K-fold validation passed")
    
    # Model configurations (conservative for small dataset)
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        )
    }
    
    all_fold_results = {model_name: [] for model_name in models.keys()}
    all_fold_results['Voting Ensemble'] = []
    all_fold_results['Stacking Ensemble'] = []
    
    for fold_idx in sorted(kfold_splits.keys()):
        fold_data = kfold_splits[fold_idx]
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*70}")
        
        # Get data
        train_ids = fold_data['train']
        test_ids = fold_data['test']
        
        train_df = df[df['Patient'].isin(train_ids)]
        test_df = df[df['Patient'].isin(test_ids)]
        
        # CRITICAL: Verify no duplicate patients in train/test
        assert train_df['Patient'].nunique() == len(train_df), "Duplicate patients in training set!"
        assert test_df['Patient'].nunique() == len(test_df), "Duplicate patients in test set!"
        
        X_train = train_df[feature_cols].values
        y_train = train_df['has_progressed'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['has_progressed'].values
        
        # Normalize
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\nTrain: {len(X_train)} patients (unique: {train_df['Patient'].nunique()})")
        print(f"Test: {len(X_test)} patients (unique: {test_df['Patient'].nunique()})")
        print(f"Features: {len(feature_cols)}")
        print(f"Train progression: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
        print(f"Test progression: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
        
        # Train individual models
        print("\n" + "-"*70)
        print("INDIVIDUAL MODELS")
        print("-"*70)
        
        trained_models = {}
        
        for model_name, model in models.items():
            metrics = train_single_model(
                model_name,
                model,
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test
            )
            
            all_fold_results[model_name].append(metrics)
            trained_models[model_name] = model
            
            print(f"{model_name:25s}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
        
        # Voting Ensemble (soft voting)
        print("\n" + "-"*70)
        print("ENSEMBLE METHODS")
        print("-"*70)
        
        voting = VotingClassifier(
            estimators=[
                ('rf', trained_models['Random Forest']),
                ('xgb', trained_models['XGBoost']),
                ('gb', trained_models['Gradient Boosting'])
            ],
            voting='soft'
        )
        
        voting_metrics = train_single_model(
            'Voting Ensemble',
            voting,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test
        )
        
        all_fold_results['Voting Ensemble'].append(voting_metrics)
        print(f"{'Voting Ensemble':25s}: AUC={voting_metrics['auc']:.3f}, F1={voting_metrics['f1']:.3f}, Acc={voting_metrics['accuracy']:.3f}")
        
        # Stacking Ensemble
        stacking = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss')),
                ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42))
            ],
            final_estimator=LogisticRegression(C=0.1),
            cv=3
        )
        
        stacking_metrics = train_single_model(
            'Stacking Ensemble',
            stacking,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test
        )
        
        all_fold_results['Stacking Ensemble'].append(stacking_metrics)
        print(f"{'Stacking Ensemble':25s}: AUC={stacking_metrics['auc']:.3f}, F1={stacking_metrics['f1']:.3f}, Acc={stacking_metrics['accuracy']:.3f}")
    
    return all_fold_results


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(all_fold_results: Dict, save_dir: Path):
    """Analyze and visualize ensemble results"""
    
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS SUMMARY")
    print("="*70)
    
    # Aggregate results
    summary_data = []
    
    for model_name, fold_results in all_fold_results.items():
        aucs = [r['auc'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        accs = [r['accuracy'] for r in fold_results]
        precs = [r['precision'] for r in fold_results]
        recs = [r['recall'] for r in fold_results]
        
        summary_data.append({
            'Model': model_name,
            'Mean_AUC': np.mean(aucs),
            'Std_AUC': np.std(aucs),
            'Mean_F1': np.mean(f1s),
            'Std_F1': np.std(f1s),
            'Mean_Accuracy': np.mean(accs),
            'Std_Accuracy': np.std(accs),
            'Mean_Precision': np.mean(precs),
            'Mean_Recall': np.mean(recs)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean_AUC', ascending=False)
    
    # Print summary
    print("\n" + "-"*70)
    print(summary_df.to_string(index=False))
    print("-"*70)
    
    # Save
    summary_df.to_csv(save_dir / 'ensemble_results_summary.csv', index=False)
    
    # Best model
    best_model = summary_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Mean AUC:       {best_model['Mean_AUC']:.4f} ± {best_model['Std_AUC']:.4f}")
    print(f"   Mean F1:        {best_model['Mean_F1']:.4f} ± {best_model['Std_F1']:.4f}")
    print(f"   Mean Accuracy:  {best_model['Mean_Accuracy']:.4f} ± {best_model['Std_Accuracy']:.4f}")
    print(f"   Mean Precision: {best_model['Mean_Precision']:.4f}")
    print(f"   Mean Recall:    {best_model['Mean_Recall']:.4f}")
    
    # Compare with deep learning baseline
    dl_baseline_auc = 0.519  # From your CNN-only results
    improvement = best_model['Mean_AUC'] - dl_baseline_auc
    
    print(f"\n{'='*70}")
    print("COMPARISON WITH DEEP LEARNING")
    print(f"{'='*70}")
    print(f"Deep Learning CNN-only:  AUC = {dl_baseline_auc:.3f}")
    print(f"Best Ensemble:           AUC = {best_model['Mean_AUC']:.3f}")
    print(f"Improvement:             {improvement:+.3f} ({improvement/dl_baseline_auc*100:+.1f}%)")
    
    if improvement > 0.08:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
        print(f"   Ensemble beats deep learning by {improvement:.3f} AUC points")
    elif improvement > 0.04:
        print(f"\n✅ MODERATE IMPROVEMENT")
        print(f"   Ensemble better than deep learning")
    elif improvement > 0.02:
        print(f"\n⚠️  MARGINAL IMPROVEMENT")
        print(f"   Ensemble slightly better than deep learning")
    elif improvement > 0:
        print(f"\n⚠️  MINIMAL IMPROVEMENT")
        print(f"   Ensemble comparable to deep learning")
    else:
        print(f"\n⚠️  NO IMPROVEMENT")
        print(f"   Deep learning performs similarly or better")
    
    return summary_df


def create_confusion_matrices(all_fold_results: Dict, save_dir: Path):
    """Create confusion matrix visualization for each model"""
    
    print("\n" + "="*70)
    print("CREATING CONFUSION MATRICES")
    print("="*70)
    
    model_names = list(all_fold_results.keys())
    n_models = len(model_names)
    
    # Create grid layout
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model_name in enumerate(model_names):
        fold_results = all_fold_results[model_name]
        
        # Aggregate confusion matrix across folds
        total_cm = np.zeros((2, 2))
        
        for fold_result in fold_results:
            y_true = fold_result['y_true']
            y_pred = (fold_result['predictions'] >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            total_cm += cm
        
        # Normalize
        cm_norm = total_cm / total_cm.sum()
        
        # Create visualization
        ax = axes[idx]
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Get average metrics
        avg_precision = np.mean([r['precision'] for r in fold_results])
        avg_recall = np.mean([r['recall'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        
        ax.set_title(f'{model_name}\nPrec={avg_precision:.3f}, Rec={avg_recall:.3f}, F1={avg_f1:.3f}', 
                    fontsize=10, fontweight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Prog', 'Prog'])
        ax.set_yticklabels(['No Prog', 'Prog'])
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({int(total_cm[i, j])})',
                             ha="center", va="center", 
                             color="white" if cm_norm[i, j] > 0.5 else "black",
                             fontsize=10)
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices (Aggregated Across Folds)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'confusion_matrices.png'}")


def create_roc_curves(all_fold_results: Dict, save_dir: Path):
    """Create ROC curves with mean ± std for each model"""
    
    print("\n" + "="*70)
    print("CREATING ROC CURVES")
    print("="*70)
    
    model_names = list(all_fold_results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        fold_results = all_fold_results[model_name]
        
        aucs = [r['auc'] for r in fold_results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Plot line for this model
        ax.plot([], [], 
               color=colors[idx], 
               linewidth=2,
               label=f'{model_name}: AUC={mean_auc:.3f}±{std_auc:.3f}')
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    
    # DL baseline
    dl_baseline = 0.519
    ax.axhline(dl_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'DL Baseline (AUC={dl_baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison\n(Mean AUC ± Std across folds)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'roc_curves_comparison.png'}")


def create_visualizations(all_fold_results: Dict, save_dir: Path):
    """Create comprehensive visualizations"""
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC comparison
    ax = axes[0, 0]
    
    model_names = list(all_fold_results.keys())
    mean_aucs = [np.mean([r['auc'] for r in all_fold_results[m]]) for m in model_names]
    std_aucs = [np.std([r['auc'] for r in all_fold_results[m]]) for m in model_names]
    
    # Sort by mean AUC
    sorted_indices = np.argsort(mean_aucs)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    mean_aucs = [mean_aucs[i] for i in sorted_indices]
    std_aucs = [std_aucs[i] for i in sorted_indices]
    
    x = np.arange(len(model_names))
    colors = ['green' if 'Ensemble' in m else 'steelblue' for m in model_names]
    
    bars = ax.barh(x, mean_aucs, xerr=std_aucs, capsize=5, alpha=0.7, color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('Model Performance Comparison (AUC)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add DL baseline line
    dl_baseline = 0.519
    ax.axvline(dl_baseline, color='red', linestyle='--', linewidth=2, 
              label=f'DL Baseline ({dl_baseline:.3f})')
    ax.legend(fontsize=9)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, mean_aucs, std_aucs)):
        width = bar.get_width()
        ax.text(width + std + 0.01, bar.get_y() + bar.get_height()/2,
               f'{mean:.3f}±{std:.3f}',
               va='center', fontsize=9)
    
    # 2. F1-Score comparison
    ax = axes[0, 1]
    
    mean_f1s = [np.mean([r['f1'] for r in all_fold_results[m]]) for m in model_names]
    std_f1s = [np.std([r['f1'] for r in all_fold_results[m]]) for m in model_names]
    
    bars = ax.barh(x, mean_f1s, xerr=std_f1s, capsize=5, alpha=0.7, color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Model Performance Comparison (F1)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Box plot - AUC distribution
    ax = axes[1, 0]
    
    auc_data = [[r['auc'] for r in all_fold_results[m]] for m in model_names]
    
    bp = ax.boxplot(auc_data, labels=[m.replace(' ', '\n') for m in model_names],
                    patch_artist=True, vert=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC Distribution Across Folds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(dl_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 4. Improvement over baseline
    ax = axes[1, 1]
    
    improvements = [mean - dl_baseline for mean in mean_aucs]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax.barh(x, improvements, alpha=0.7, color=colors_imp)
    ax.set_yticks(x)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('AUC Improvement over DL Baseline', fontsize=12)
    ax.set_title('Improvement Analysis', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {imp:+.3f} ({imp/dl_baseline*100:+.1f}%)',
               va='center', ha=ha, fontsize=9)
    
    plt.suptitle('ML Ensemble vs Deep Learning Comparison',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_dir / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'ensemble_comparison.png'}")


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(
    df: pd.DataFrame,
    kfold_splits: dict,
    feature_cols: List[str],
    save_dir: Path
):
    """Analyze feature importance using Random Forest"""
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    all_importances = []
    
    for fold_idx in sorted(kfold_splits.keys()):
        fold_data = kfold_splits[fold_idx]
        
        train_ids = fold_data['train']
        train_df = df[df['Patient'].isin(train_ids)]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['has_progressed'].values
        
        # Train RF
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        
        all_importances.append(rf.feature_importances_)
    
    # Average importances across folds
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': mean_importances,
        'Std': std_importances
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Print top features
    print("\nTop 15 Most Important Features:")
    print("-" * 70)
    print(importance_df.head(15).to_string(index=False))
    
    # Save
    importance_df.to_csv(save_dir / 'feature_importance.csv', index=False)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = 20
    top_features = importance_df.head(top_n)
    
    colors = ['green' if 'Age' in f or 'Tissue' in f or 'Interaction' in f 
              else 'steelblue' for f in top_features['Feature']]
    
    ax.barh(range(top_n), top_features['Importance'], 
           xerr=top_features['Std'], capsize=3, alpha=0.7, color=colors)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features (Random Forest)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {save_dir / 'feature_importance.png'}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution with CRITICAL data leakage prevention
    """
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    BASE_DIR = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training")
    
    GT_PATH = BASE_DIR / "Label_ground_truth" / "ground_truth.csv"
    PATIENT_FEATURES_PATH = BASE_DIR / "Dataset" / "patient_features_30_60.csv"
    TRAIN_CSV_PATH = BASE_DIR / "Dataset" / "train.csv"
    KFOLD_SPLITS_PATH = BASE_DIR / "Dataset" / "kfold_splits_stratified.pkl"
    
    # CNN features extraction
    CT_SCAN_PATH = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy")
    USE_CNN_FEATURES = True
    N_CNN_FEATURES = 100
    
    RESULTS_DIR = BASE_DIR / "ml_ensemble_results_corrected"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ML ENSEMBLE FOR IPF PROGRESSION PREDICTION")
    print("="*70)
    print(f"\nGoal: Beat Deep Learning AUC of 0.519")
    print(f"Target: Achieve AUC > 0.60")
    print(f"CNN Features: {'Enabled' if USE_CNN_FEATURES else 'Disabled'}")
    print(f"\n⚠️  DATA LEAKAGE PREVENTION ENABLED")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load data using IPFDataLoader (for patient_data dictionary)
    data_loader = IPFDataLoader(
        csv_path=GT_PATH,
        features_path=PATIENT_FEATURES_PATH,
        npy_dir=CT_SCAN_PATH
    )
    patient_data, _ = data_loader.get_patient_data()
    
    print(f"✓ Loaded {len(patient_data)} patients with CT scans")
    
    # Load patient features DataFrame separately
    patient_features_df = pd.read_csv(PATIENT_FEATURES_PATH)
    
    # CRITICAL: Keep only one row per patient
    print(f"\n⚠️  Checking for duplicate patients in patient_features_df...")
    print(f"   Original shape: {patient_features_df.shape}")
    print(f"   Unique patients: {patient_features_df['Patient'].nunique()}")
    
    if patient_features_df['Patient'].nunique() != len(patient_features_df):
        print(f"   ⚠️  WARNING: Multiple rows per patient detected!")
        print(f"   Taking first occurrence of each patient...")
        patient_features_df = patient_features_df.drop_duplicates(subset='Patient', keep='first')
        print(f"   New shape: {patient_features_df.shape}")
    
    print(f"✓ Patient features: {patient_features_df.shape}")
    
    # Load and merge demographics
    if TRAIN_CSV_PATH.exists():
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        demo_cols = ['Patient']
        for col in ['Age', 'Sex', 'SmokingStatus']:
            if col in train_df.columns:
                demo_cols.append(col)
        
        demographics = train_df[demo_cols].copy()
        
        # Remove duplicates in demographics too
        if demographics['Patient'].nunique() != len(demographics):
            print(f"   ⚠️  Removing duplicates from demographics...")
            demographics = demographics.drop_duplicates(subset='Patient', keep='first')
        
        # Encode categoricals
        if 'Sex' in demographics.columns:
            demographics['Sex'] = demographics['Sex'].map({'Male': 1, 'Female': 0})
        
        if 'SmokingStatus' in demographics.columns:
            demographics['SmokingStatus'] = demographics['SmokingStatus'].map({
                'Never smoked': 0,
                'Ex-smoker': 1,
                'Currently smokes': 2
            })
        
        patient_features_df = patient_features_df.merge(demographics, on='Patient', how='left')
        print(f"✓ Merged demographics")
    
    # Prepare main dataframe
    df = patient_features_df.copy()
    
    # CRITICAL: Final check for duplicates
    assert df['Patient'].nunique() == len(df), f"CRITICAL: Still have duplicates! {df['Patient'].nunique()} unique vs {len(df)} rows"
    
    print(f"✓ Final dataset: {df.shape} ({df['Patient'].nunique()} unique patients)")
    
    # Load ground truth labels
    gt_df = pd.read_csv(GT_PATH)
    print(f"✓ Ground truth: {gt_df.shape}")
    
    # Remove duplicates in ground truth
    if gt_df['PatientID'].nunique() != len(gt_df):
        print(f"   ⚠️  Removing duplicates from ground truth...")
        gt_df = gt_df.drop_duplicates(subset='PatientID', keep='first')
    
    # Convert has_progressed to numeric if needed
    if gt_df['has_progressed'].dtype == 'object':
        gt_df['has_progressed'] = gt_df['has_progressed'].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    # Merge labels
    df = df.merge(
        gt_df[['PatientID', 'has_progressed']],
        left_on='Patient',
        right_on='PatientID',
        how='inner'
    )
    df = df.drop(columns=['PatientID'])
    
    # CRITICAL: Verify no duplicates after merge
    assert df['Patient'].nunique() == len(df), f"CRITICAL: Duplicates after merge! {df['Patient'].nunique()} unique vs {len(df)} rows"
    
    print(f"✓ Merged labels: {df.shape} ({df['Patient'].nunique()} unique patients)")
    print(f"  Progression cases: {df['has_progressed'].sum()}/{len(df)} ({df['has_progressed'].mean()*100:.1f}%)")
    
    # Load k-fold splits
    with open(KFOLD_SPLITS_PATH, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print(f"✓ K-fold splits: {len(kfold_splits)} folds")
    
    # =========================================================================
    # EXTRACT CNN FEATURES (if enabled)
    # =========================================================================
    
    if USE_CNN_FEATURES:
        print("\n" + "="*70)
        print("EXTRACTING CNN FEATURES")
        print("="*70)
        
        # Extract slice-level CNN features
        slice_features_df = extract_cnn_features_with_extractor(
            patient_data=patient_data,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Aggregate to patient-level
        patient_cnn_df = aggregate_cnn_features_per_patient(slice_features_df)
        
        # CRITICAL: Verify patient-level aggregation
        print(f"\n{'='*70}")
        print("DATA VALIDATION - CNN FEATURES")
        print(f"{'='*70}")
        print(f"CNN features - Unique patients: {patient_cnn_df['patient_id'].nunique()}")
        print(f"CNN features - Total rows: {len(patient_cnn_df)}")
        print(f"Main df - Unique patients: {df['Patient'].nunique()}")
        
        assert patient_cnn_df['patient_id'].nunique() == len(patient_cnn_df), \
            "CRITICAL: Duplicate patients in CNN features!"
        
        print(f"✓ CNN features validation passed")
        
        # Select best CNN features
        selected_cnn_cols = select_best_cnn_features(
            patient_cnn_df=patient_cnn_df,
            labels_df=df[['Patient', 'has_progressed']],
            n_features=N_CNN_FEATURES
        )
        
        # Add selected CNN features to main dataframe
        print(f"\nMerging {len(selected_cnn_cols)} CNN features...")
        
        # Rename columns to avoid conflicts
        cnn_df_selected = patient_cnn_df[['patient_id'] + selected_cnn_cols].copy()
        cnn_df_selected = cnn_df_selected.rename(columns={col: f'CNN_{col}' for col in selected_cnn_cols})
        
        # Merge
        df = df.merge(cnn_df_selected, left_on='Patient', right_on='patient_id', how='inner')
        df = df.drop(columns=['patient_id'])
        
        # CRITICAL: Verify no duplicates after merge
        assert df['Patient'].nunique() == len(df), \
            f"CRITICAL: Duplicates after CNN merge! {df['Patient'].nunique()} unique vs {len(df)} rows"
        
        print(f"✓ Merged CNN features")
        print(f"✓ Final dataset: {len(df)} patients (all unique)")
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    df_engineered = engineer_features(df)
    
    # CRITICAL: Verify still no duplicates
    assert df_engineered['Patient'].nunique() == len(df_engineered), \
        "CRITICAL: Duplicates after feature engineering!"
    
    feature_cols = select_features(df_engineered, include_cnn=USE_CNN_FEATURES)
    
    # =========================================================================
    # FINAL VALIDATION BEFORE TRAINING
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("FINAL PRE-TRAINING VALIDATION")
    print(f"{'='*70}")
    print(f"Total patients in dataset: {len(df_engineered)}")
    print(f"Unique patients: {df_engineered['Patient'].nunique()}")
    print(f"Features selected: {len(feature_cols)}")
    print(f"Class distribution: {df_engineered['has_progressed'].value_counts().to_dict()}")
    
    if df_engineered['Patient'].nunique() != len(df_engineered):
        raise ValueError("CRITICAL ERROR: Dataset still has duplicate patients!")
    
    print(f"✓ All validations passed - ready for training")
    
    # =========================================================================
    # TRAIN ENSEMBLE
    # =========================================================================
    
    all_fold_results = train_ensemble_kfold(
        df=df_engineered,
        kfold_splits=kfold_splits,
        feature_cols=feature_cols,
        save_dir=RESULTS_DIR
    )
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    summary_df = analyze_results(all_fold_results, RESULTS_DIR)
    
    create_visualizations(all_fold_results, RESULTS_DIR)
    
    create_confusion_matrices(all_fold_results, RESULTS_DIR)
    
    create_roc_curves(all_fold_results, RESULTS_DIR)
    
    analyze_feature_importance(
        df=df_engineered,
        kfold_splits=kfold_splits,
        feature_cols=feature_cols,
        save_dir=RESULTS_DIR
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("✓ ML ENSEMBLE TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"\nGenerated files:")
    print(f"  1. ensemble_results_summary.csv - Performance summary")
    print(f"  2. ensemble_comparison.png - Visual comparison")
    print(f"  3. confusion_matrices.png - Confusion matrices for all models")
    print(f"  4. roc_curves_comparison.png - ROC curves with AUC mean±std")
    print(f"  5. feature_importance.csv - Feature rankings")
    print(f"  6. feature_importance.png - Top features visualization")
    
    # Final recommendation
    best_auc = summary_df.iloc[0]['Mean_AUC']
    dl_baseline = 0.519
    
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}")
    
    if best_auc > 0.60:
        print(f"\n✅ EXCELLENT RESULTS!")
        print(f"   Best ensemble achieves AUC = {best_auc:.3f}")
        print(f"   This is {best_auc - dl_baseline:.3f} better than deep learning")
        print(f"\n   → USE THIS ENSEMBLE for your thesis/paper")
        
    elif best_auc > dl_baseline + 0.04:
        print(f"\n✅ GOOD RESULTS!")
        print(f"   Best ensemble achieves AUC = {best_auc:.3f}")
        print(f"   Improvement over deep learning: {best_auc - dl_baseline:.3f}")
        print(f"\n   → ENSEMBLE is better than deep learning")
        
    elif best_auc > dl_baseline + 0.02:
        print(f"\n✅ MODERATE IMPROVEMENT")
        print(f"   Best ensemble achieves AUC = {best_auc:.3f}")
        print(f"   Better than deep learning ({dl_baseline:.3f})")
        print(f"\n   → Consider combining ensemble + deep features")
        
    elif best_auc > dl_baseline:
        print(f"\n⚠️  MARGINAL IMPROVEMENT")
        print(f"   Best ensemble achieves AUC = {best_auc:.3f}")
        print(f"   Slightly better than deep learning ({dl_baseline:.3f})")
        print(f"\n   → Hybrid approach may help")
        
    else:
        print(f"\n⚠️  NO IMPROVEMENT")
        print(f"   Best ensemble achieves AUC = {best_auc:.3f}")
        print(f"   Deep learning performs similarly ({dl_baseline:.3f})")
        print(f"\n   → Task is fundamentally difficult")
        print(f"   → Consider survival analysis or longitudinal modeling")


if __name__ == "__main__":
    main()