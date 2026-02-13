# PCA + Logistic Regression for IPF Progression Prediction
# Integrates with existing ablation study pipeline

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sys
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    CNNFeatureExtractor,
    IPFDataLoader,
    compute_class_weights
)


# Hand-crafted features to use
HAND_FEATURE_COLS = [
    'ApproxVol',
    'Avg_NumTissuePixel',
    'Avg_Tissue',
    'Avg_Tissue_thickness',
    'Avg_TissueByTotal',
    'Avg_TissueByLung',
    'Mean',
    'Skew',
    'Kurtosis'
]

# Demographic features
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']


BASE_CONFIG = {
    # Paths
    "gt_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_all_slices.csv"),
    "kfold_splits_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),
    
    # PCA parameters
    'n_pca_components': 0.95,  # 95% variance explained
    'pooling_type': 'max',  # 'max' or 'mean'
    
    # Logistic Regression parameters
    'cv_folds': 5,
    'penalty': 'l2',
    'Cs': np.logspace(-3, 3, 20),  # Regularization strengths to try
    
    # Normalization
    'normalization_type': 'standard',
}


def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load demographics from train.csv and merge with patient_features"""
    train_df = pd.read_csv(train_csv_path)
    
    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS FROM TRAIN.CSV")
    print("="*70)
    print(f"Train CSV shape: {train_df.shape}")
    print(f"Columns available: {train_df.columns.tolist()}")
    
    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  Found: {col}")
        else:
            print(f"  ⚠️  Missing: {col}")
    
    demographics_df = train_df[demo_cols].copy()
    enhanced_df = patient_features_df.merge(demographics_df, on='Patient', how='left')
    
    print(f"\nEnhanced features shape: {enhanced_df.shape}")
    
    # Encode categorical variables
    if 'Sex' in enhanced_df.columns:
        print(f"\nEncoding Sex (categorical -> numeric)")
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"  Encoded as: Male=1, Female=0")
    
    if 'SmokingStatus' in enhanced_df.columns:
        print(f"\nEncoding SmokingStatus (categorical -> numeric)")
        smoking_map = {
            'Never smoked': 0,
            'Ex-smoker': 1,
            'Currently smokes': 2
        }
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(smoking_map)
        print(f"  Encoded as: Never smoked=0, Ex-smoker=1, Currently smokes=2")
    
    return enhanced_df


def preprocess_demographics_improved(
    result_df: pd.DataFrame,
    train_patient_ids: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Improved demographics preprocessing with proper centering and encoding
    Same as ablation_study.py
    """
    encoding_info = {}
    
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)]
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    
    # === AGE (Continuous) ===
    if 'Age' in result_df.columns:
        print("\n=== PREPROCESSING AGE ===")
        
        print(f"  Pre-normalization:")
        print(f"    Mean: {train_patient_df['Age'].mean():.2f}")
        print(f"    Std: {train_patient_df['Age'].std():.2f}")
        
        age_scaler = StandardScaler()
        age_scaler.fit(train_patient_df[['Age']].values)
        result_df['Age_normalized'] = age_scaler.transform(result_df[['Age']].values)
        
        print(f"  Post-normalization:")
        train_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)]
        train_patient_normalized = train_normalized.groupby('patient_id').first().reset_index()
        print(f"    Mean: {train_patient_normalized['Age_normalized'].mean():.4f}")
        print(f"    Std: {train_patient_normalized['Age_normalized'].std():.4f}")
        
        encoding_info['age_scaler'] = age_scaler
    
    # === SEX (Binary Categorical) ===
    if 'Sex' in result_df.columns:
        print("\n=== PREPROCESSING SEX ===")
        result_df['Sex_encoded'] = result_df['Sex'].map({0: -1, 1: 1})
        print(f"  Encoded as: Female=-1, Male=1 (centered)")
        encoding_info['sex_encoding'] = {0: -1, 1: 1}
    
    # === SMOKING STATUS (Multi-class Categorical) ===
    if 'SmokingStatus' in result_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS ===")
        
        smoking_dummies = pd.get_dummies(
            result_df['SmokingStatus'], 
            prefix='Smoking',
            dtype=float
        )
        smoking_dummies = (smoking_dummies - 0.5)
        
        for col in smoking_dummies.columns:
            result_df[col] = smoking_dummies[col]
        
        smoking_cols = sorted(smoking_dummies.columns.tolist())
        encoding_info['smoking_columns'] = smoking_cols
        
        print(f"  One-hot encoded into {len(smoking_cols)} features")
        print(f"  Values centered to [-0.5, 0.5]")
    
    return result_df, encoding_info


def normalize_features_per_fold(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    hand_feature_cols: list,
    demo_feature_cols: list,
    normalization_type: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize features using ONLY training set statistics
    Same as ablation_study.py
    """
    result_df = features_df.copy()
    scalers = {}
    
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
    
    if not available_hand and not available_demo:
        return result_df, scalers
    
    print(f"\nNormalizing features ({normalization_type} scaler):")
    
    train_df = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
    train_patient_df = train_df.groupby('patient_id').first().reset_index()
    print(f"  Training set: {len(train_patient_df)} patients")
    
    # Hand-crafted features
    if available_hand:
        print(f"  Hand-crafted: {len(available_hand)} features")
        
        hand_scaler = StandardScaler()
        hand_scaler.fit(train_patient_df[available_hand].values)
        result_df[available_hand] = hand_scaler.transform(result_df[available_hand].values)
        
        train_df_normalized = result_df[result_df['patient_id'].isin(train_patient_ids)].copy()
        train_patient_normalized = train_df_normalized.groupby('patient_id').first().reset_index()
        
        print(f"\n  Post-normalization (Training Set):")
        for col in available_hand[:3]:  # Show first 3
            mean = train_patient_normalized[col].mean()
            std = train_patient_normalized[col].std()
            print(f"    {col:30s}: mean={mean:10.4f}, std={std:10.4f}")
        
        scalers['hand_scaler'] = hand_scaler
    
    # Demographics
    if available_demo:
        print(f"\n=== DEMOGRAPHIC FEATURES ===")
        print(f"  Found {len(available_demo)} demographic columns")
        
        result_df, encoding_info = preprocess_demographics_improved(
            result_df,
            train_patient_ids,
            normalization_type
        )
        
        scalers['demo_encoding'] = encoding_info
        print(f"  ✓ Demographics preprocessed")
    
    return result_df, scalers


class PCALogisticModel:
    """
    Complete pipeline: CNN features → Max/Mean Pool → PCA → + Hand/Demo → LogReg
    """
    
    def __init__(
        self,
        n_pca_components=0.95,
        pooling_type='max',
        cv_folds=5,
        penalty='l2',
        Cs=None
    ):
        self.n_pca_components = n_pca_components
        self.pooling_type = pooling_type
        self.cv_folds = cv_folds
        self.penalty = penalty
        self.Cs = Cs if Cs is not None else np.logspace(-3, 3, 20)
        
        # Components (fitted during training)
        self.cnn_scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components, svd_solver='full')
        
        self.classifier = LogisticRegressionCV(
            Cs=self.Cs,
            cv=cv_folds,
            penalty=penalty,
            solver='liblinear',
            scoring='roc_auc',
            max_iter=1000,
            random_state=42
        )
        
        self.is_fitted = False
    
    def _pool_cnn_features(self, slice_features_df, patient_ids):
        """
        Pool CNN features per patient
        
        Args:
            slice_features_df: DataFrame with CNN features per slice
            patient_ids: List of patient IDs
            
        Returns:
            pooled_features: (n_patients, cnn_dim)
            patient_order: List of patient IDs in same order
        """
        cnn_cols = [c for c in slice_features_df.columns if c.startswith('cnn_feature_')]
        
        pooled_list = []
        patient_order = []
        
        for patient_id in patient_ids:
            patient_slices = slice_features_df[
                slice_features_df['patient_id'] == patient_id
            ][cnn_cols].values
            
            if self.pooling_type == 'max':
                pooled = patient_slices.max(axis=0)
            elif self.pooling_type == 'mean':
                pooled = patient_slices.mean(axis=0)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling_type}")
            
            pooled_list.append(pooled)
            patient_order.append(patient_id)
        
        return np.array(pooled_list), patient_order
    
    def fit(
        self,
        slice_features_df: pd.DataFrame,
        patient_features_df: pd.DataFrame,
        train_patient_ids: list,
        verbose: bool = True
    ):
        """
        Fit the complete pipeline on training data
        """
        if verbose:
            print("\n" + "="*70)
            print("FITTING PCA + LOGISTIC REGRESSION PIPELINE")
            print("="*70)
            print(f"Training samples: {len(train_patient_ids)}")
        
        # Step 1: Pool CNN features
        if verbose:
            print("\n1. Pooling CNN features...")
        
        cnn_pooled, patient_order = self._pool_cnn_features(
            slice_features_df, train_patient_ids
        )
        
        if verbose:
            print(f"   Pooled shape: {cnn_pooled.shape}")
            print(f"   Pooling method: {self.pooling_type}")
        
        # Step 2: Standardize CNN features
        if verbose:
            print("\n2. Standardizing CNN features...")
        
        cnn_scaled = self.cnn_scaler.fit_transform(cnn_pooled)
        
        if verbose:
            print(f"   Mean: {cnn_scaled.mean():.6f}")
            print(f"   Std: {cnn_scaled.std():.6f}")
        
        # Step 3: PCA on CNN features
        if verbose:
            print("\n3. Applying PCA...")
        
        cnn_pca = self.pca.fit_transform(cnn_scaled)
        
        if verbose:
            n_components = self.pca.n_components_
            var_explained = self.pca.explained_variance_ratio_.sum()
            print(f"   Components selected: {n_components}")
            print(f"   Variance explained: {var_explained:.1%}")
            print(f"   Feature dimension: {cnn_pooled.shape[1]} → {n_components}")
        
        # Step 4: Get hand-crafted and demographic features (already normalized)
        if verbose:
            print("\n4. Extracting hand-crafted and demographic features...")
        
        # Get patient-level features
        train_patient_df = patient_features_df[
            patient_features_df['patient_id'].isin(train_patient_ids)
        ].set_index('patient_id').loc[patient_order].reset_index()
        
        # Hand-crafted features
        available_hand = [c for c in HAND_FEATURE_COLS if c in train_patient_df.columns]
        if available_hand:
            hand_features = train_patient_df[available_hand].values
        else:
            hand_features = np.zeros((len(patient_order), 0))
        
        # Demographic features (preprocessed)
        demo_features_list = []
        for _, row in train_patient_df.iterrows():
            demo_vals = []
            if 'Age_normalized' in row:
                demo_vals.append(row['Age_normalized'])
            if 'Sex_encoded' in row:
                demo_vals.append(row['Sex_encoded'])
            # Smoking one-hot columns (if exist)
            smoking_cols = [c for c in row.index if c.startswith('Smoking_')]
            for col in sorted(smoking_cols):
                demo_vals.append(row[col])
            demo_features_list.append(demo_vals)
        
        demo_features = np.array(demo_features_list) if demo_features_list[0] else np.zeros((len(patient_order), 0))
        
        if verbose:
            print(f"   Hand-crafted features: {hand_features.shape[1]}")
            print(f"   Demographic features: {demo_features.shape[1]}")
        
        # Step 5: Combine all features
        if verbose:
            print("\n5. Combining features...")
        
        X_combined = np.concatenate([
            cnn_pca,
            hand_features,
            demo_features
        ], axis=1)
        
        if verbose:
            print(f"   Total features: {X_combined.shape[1]}")
            print(f"     - CNN (PCA): {cnn_pca.shape[1]}")
            print(f"     - Hand-crafted: {hand_features.shape[1]}")
            print(f"     - Demographics: {demo_features.shape[1]}")
        
        # Get labels
        y_train = train_patient_df['gt_has_progressed'].values
        
        # Step 6: Fit Logistic Regression
        if verbose:
            print("\n6. Fitting Logistic Regression (with CV)...")
        
        self.classifier.fit(X_combined, y_train)
        
        if verbose:
            print(f"   Best C: {self.classifier.C_[0]:.4f}")
            print(f"   Cross-val AUC: {self.classifier.scores_[1].mean():.4f}")
        
        self.is_fitted = True
        
        # Training metrics
        train_pred = self.classifier.predict_proba(X_combined)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        if verbose:
            print(f"\n✓ Training AUC: {train_auc:.4f}")
        
        return {
            'train_auc': train_auc,
            'best_C': self.classifier.C_[0],
            'n_pca_components': self.pca.n_components_,
            'variance_explained': self.pca.explained_variance_ratio_.sum()
        }
    
    def predict_proba(
        self,
        slice_features_df: pd.DataFrame,
        patient_features_df: pd.DataFrame,
        patient_ids: list
    ) -> np.ndarray:
        """
        Predict probabilities for new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        # Pool CNN features
        cnn_pooled, patient_order = self._pool_cnn_features(
            slice_features_df, patient_ids
        )
        
        # Apply same transformations
        cnn_scaled = self.cnn_scaler.transform(cnn_pooled)
        cnn_pca = self.pca.transform(cnn_scaled)
        
        # Get hand-crafted and demographic features
        patient_df = patient_features_df[
            patient_features_df['patient_id'].isin(patient_ids)
        ].set_index('patient_id').loc[patient_order].reset_index()
        
        # Hand-crafted
        available_hand = [c for c in HAND_FEATURE_COLS if c in patient_df.columns]
        if available_hand:
            hand_features = patient_df[available_hand].values
        else:
            hand_features = np.zeros((len(patient_order), 0))
        
        # Demographics
        demo_features_list = []
        for _, row in patient_df.iterrows():
            demo_vals = []
            if 'Age_normalized' in row:
                demo_vals.append(row['Age_normalized'])
            if 'Sex_encoded' in row:
                demo_vals.append(row['Sex_encoded'])
            smoking_cols = [c for c in row.index if c.startswith('Smoking_')]
            for col in sorted(smoking_cols):
                demo_vals.append(row[col])
            demo_features_list.append(demo_vals)
        
        demo_features = np.array(demo_features_list) if demo_features_list[0] else np.zeros((len(patient_order), 0))
        
        # Combine
        X_combined = np.concatenate([
            cnn_pca,
            hand_features,
            demo_features
        ], axis=1)
        
        return self.classifier.predict_proba(X_combined)[:, 1], patient_order
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from logistic regression coefficients"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        
        coefficients = self.classifier.coef_[0]
        
        # Create feature names
        feature_names = []
        
        # PCA components
        for i in range(self.pca.n_components_):
            feature_names.append(f'PCA_{i+1}')
        
        # Hand-crafted
        available_hand = HAND_FEATURE_COLS
        feature_names.extend(available_hand)
        
        # Demographics
        feature_names.append('Age')
        feature_names.append('Sex')
        feature_names.extend(['Smoking_Never', 'Smoking_Ex', 'Smoking_Current'])
        
        # Truncate to actual number of features
        feature_names = feature_names[:len(coefficients)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df


def train_single_fold_pca_lr(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path
):
    """
    Train PCA + LogReg model on a single fold
    """
    
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_idx} - PCA + LOGISTIC REGRESSION")
    print("="*70)
    
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_ids = fold_data['train']
    val_ids = fold_data['val']
    test_ids = fold_data['test']
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    # Create model
    model = PCALogisticModel(
        n_pca_components=config['n_pca_components'],
        pooling_type=config['pooling_type'],
        cv_folds=config['cv_folds'],
        penalty=config['penalty'],
        Cs=config['Cs']
    )
    
    # Train
    train_metrics = model.fit(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        train_patient_ids=train_ids,
        verbose=True
    )
    
    # Evaluate on validation
    print(f"\n{'='*70}")
    print("VALIDATION SET EVALUATION")
    print("="*70)
    
    val_pred, val_order = model.predict_proba(
        slice_features_df, patient_features_df, val_ids
    )
    val_patient_df = patient_features_df[
        patient_features_df['patient_id'].isin(val_ids)
    ].set_index('patient_id').loc[val_order].reset_index()
    val_y = val_patient_df['gt_has_progressed'].values
    
    val_auc = roc_auc_score(val_y, val_pred)
    print(f"Val AUC: {val_auc:.4f}")
    
    # Evaluate on test
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_pred, test_order = model.predict_proba(
        slice_features_df, patient_features_df, test_ids
    )
    test_patient_df = patient_features_df[
        patient_features_df['patient_id'].isin(test_ids)
    ].set_index('patient_id').loc[test_order].reset_index()
    test_y = test_patient_df['gt_has_progressed'].values
    
    test_auc = roc_auc_score(test_y, test_pred)
    test_acc = accuracy_score(test_y, (test_pred >= 0.5).astype(int))
    test_f1 = f1_score(test_y, (test_pred >= 0.5).astype(int))
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': test_order,
        'true_label': test_y,
        'predicted_prob': test_pred,
        'predicted_label': (test_pred >= 0.5).astype(int)
    })
    predictions_df.to_csv(fold_dir / "test_predictions.csv", index=False)
    
    # Save feature importance
    feature_importance = model.get_feature_importance()
    feature_importance.to_csv(fold_dir / "feature_importance.csv", index=False)
    
    # Save model
    with open(fold_dir / "pca_logreg_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Fold {fold_idx} complete! Results saved to: {fold_dir}")
    
    return {
        'fold_idx': fold_idx,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'n_components': train_metrics['n_pca_components'],
        'best_C': train_metrics['best_C']
    }


def run_kfold_pca_lr(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    config: dict,
    results_base_dir: Path
):
    """
    Run K-fold cross-validation with PCA + LogReg
    """
    
    print("\n" + "="*70)
    print("PCA + LOGISTIC REGRESSION - K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = []
    fold_keys = sorted(kfold_splits.keys())
    
    for fold_idx in fold_keys:
        fold_data = kfold_splits[fold_idx]
        
        result = train_single_fold_pca_lr(
            slice_features_df=slice_features_df,
            patient_features_df=patient_features_df,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=config,
            results_dir=results_base_dir
        )
        
        fold_results.append(result)
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    print(f"Val AUC: {results_df['val_auc'].mean():.4f} ± {results_df['val_auc'].std():.4f}")
    print(f"Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"Test Acc: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")
    print(f"Test F1: {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")
    
    results_df.to_csv(results_base_dir / "kfold_summary.csv", index=False)
    
    return results_df


def main():
    results_base_dir = Path(
        r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training"
        r"\pca_logistic_regression"
    )
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=None
    )
    
    # Load patient features and demographics
    print("\n" + "="*70)
    print("LOADING PATIENT FEATURES")
    print("="*70)
    
    patient_features_df = pd.read_csv(BASE_CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=BASE_CONFIG['train_csv_path'],
        patient_features_df=patient_features_df
    )
    
    # Load K-Fold splits
    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"\nLoaded {len(kfold_splits)} folds")
    
    # CRITICAL: Normalize features per fold
    print("\n" + "="*70)
    print("NORMALIZING FEATURES PER FOLD")
    print("="*70)
    
    all_fold_results = []
    
    for fold_idx, fold_data in kfold_splits.items():
        print(f"\nProcessing fold {fold_idx}...")
        
        # Merge slice features with patient features
        merged_df = slice_features_df.merge(
            patient_features_df[['Patient'] + HAND_FEATURE_COLS + DEMO_FEATURE_COLS],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        merged_df.drop('Patient', axis=1, inplace=True)
        
        # Normalize using training set only
        normalized_df, scalers = normalize_features_per_fold(
            features_df=merged_df,
            train_patient_ids=fold_data['train'],
            hand_feature_cols=HAND_FEATURE_COLS,
            demo_feature_cols=DEMO_FEATURE_COLS,
            normalization_type=BASE_CONFIG['normalization_type']
        )
        
        # Split back into slice features and patient features
        cnn_cols = [c for c in normalized_df.columns if c.startswith('cnn_feature_')]
        slice_cols = ['patient_id', 'slice_path', 'slice_index', 'total_slices', 'gt_has_progressed'] + cnn_cols
        
        fold_slice_features = normalized_df[slice_cols]
        
        # Patient features (one row per patient)
        patient_cols = ['patient_id', 'gt_has_progressed'] + HAND_FEATURE_COLS
        if 'Age_normalized' in normalized_df.columns:
            patient_cols.append('Age_normalized')
        if 'Sex_encoded' in normalized_df.columns:
            patient_cols.append('Sex_encoded')
        smoking_cols = [c for c in normalized_df.columns if c.startswith('Smoking_')]
        patient_cols.extend(smoking_cols)
        
        fold_patient_features = normalized_df.groupby('patient_id').first()[
            [c for c in patient_cols if c != 'patient_id']
        ].reset_index()
        
        # Train fold
        result = train_single_fold_pca_lr(
            slice_features_df=fold_slice_features,
            patient_features_df=fold_patient_features,
            fold_data=fold_data,
            fold_idx=fold_idx,
            config=BASE_CONFIG,
            results_dir=results_base_dir
        )
        
        all_fold_results.append(result)
    
    # Final summary
    results_df = pd.DataFrame(all_fold_results)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"Results: {results_base_dir}")
    
    print("\nFinal Performance:")
    print(f"  Val AUC: {results_df['val_auc'].mean():.4f} ± {results_df['val_auc'].std():.4f}")
    print(f"  Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")


if __name__ == "__main__":
    main()