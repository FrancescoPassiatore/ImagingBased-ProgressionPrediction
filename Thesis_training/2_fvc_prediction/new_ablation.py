# FVC Ablation Study with Correct Normalization
# IMPORTANT: Each fold normalizes using ONLY its own training set

from pathlib import Path
import random
import pandas as pd
import pickle
import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.append(str(Path(__file__).parent.parent))
from utilities import CNNFeatureExtractor, IPFDataLoader, create_fvc_dataloaders
from fvc_prediction_model import (
    FVCPredictionModel,
    FVCModelTrainer,
    plot_evaluation_metrics
)


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================
# 4-block structure:
#   Block 1 – Clinical baseline  (hand-crafted + demographics, with FVC(0))
#   Block 2 – Imaging only       (pooling comparison, with FVC(0))
#   Block 3 – Full multimodal    (best pooling from Block 2, with FVC(0))
#   Block 4 – FVC(0) ablation    (mirrors key configs but WITHOUT FVC(0))

ABLATION_CONFIGS = {
    # ── BLOCK 1: Clinical Baseline (no imaging) ───────────────────────────────
    'hand_only': {
        'use_cnn_features':  False,
        'use_hand_features': True,
        'use_demographics':  False,
        'use_fvc_baseline':  True,
        'description': '[BLOCK 1] Hand-crafted features + FVC(0)',
    },
    'hand_demo': {
        'use_cnn_features':  False,
        'use_hand_features': True,
        'use_demographics':  True,
        'use_fvc_baseline':  True,
        'description': '[BLOCK 1] Hand-crafted + Demographics + FVC(0)',
    },

    # ── BLOCK 2: Imaging Only – pooling comparison ────────────────────────────
    'cnn_mean': {
        'use_cnn_features':  True,
        'use_hand_features': False,
        'use_demographics':  False,
        'use_fvc_baseline':  True,
        'pooling_type':      'mean',
        'description': '[BLOCK 2] CNN mean pooling + FVC(0)',
    },
    'cnn_max': {
        'use_cnn_features':  True,
        'use_hand_features': False,
        'use_demographics':  False,
        'use_fvc_baseline':  True,
        'pooling_type':      'max',
        'description': '[BLOCK 2] CNN max pooling + FVC(0)',
    },
    'cnn_max_mean': {
        'use_cnn_features':  True,
        'use_hand_features': False,
        'use_demographics':  False,
        'use_fvc_baseline':  True,
        'pooling_type':      'max_mean',
        'description': '[BLOCK 2] CNN max+mean pooling + FVC(0)',
    },

    # ── BLOCK 3: Full Multimodal ──────────────────────────────────────────────
    # Uses pooling_type from BASE_CONFIG — update after Block 2 results.
    'cnn_hand': {
        'use_cnn_features':  True,
        'use_hand_features': True,
        'use_demographics':  False,
        'use_fvc_baseline':  True,
        'description': '[BLOCK 3] CNN + Hand-crafted + FVC(0)',
    },
    'cnn_hand_demo': {
        'use_cnn_features':  True,
        'use_hand_features': True,
        'use_demographics':  True,
        'use_fvc_baseline':  True,
        'description': '[BLOCK 3] CNN + Hand-crafted + Demographics + FVC(0) [FULL]',
    },

    # ── BLOCK 4: FVC(0) Ablation ──────────────────────────────────────────────
    # Mirrors key configs from Blocks 1-3 but with FVC(0) zeroed out.
    # Answers: "how much does FVC(0) contribute vs imaging/handcrafted alone?"
    # Expected: significant MAE increase → FVC(0) is the dominant predictor.
    'hand_only_nofvc': {
        'use_cnn_features':  False,
        'use_hand_features': True,
        'use_demographics':  False,
        'use_fvc_baseline':  False,
        'description': '[BLOCK 4] Hand-crafted only – NO FVC(0)',
    },
    'cnn_only_nofvc': {
        'use_cnn_features':  True,
        'use_hand_features': False,
        'use_demographics':  False,
        'use_fvc_baseline':  False,
        'pooling_type':      'max_mean',   # best pooling from Block 2
        'description': '[BLOCK 4] CNN only – NO FVC(0)',
    },
    'cnn_hand_nofvc': {
        'use_cnn_features':  True,
        'use_hand_features': True,
        'use_demographics':  False,
        'use_fvc_baseline':  False,
        'pooling_type':      'max_mean',
        'description': '[BLOCK 4] CNN + Hand-crafted – NO FVC(0)',
    },
    'cnn_hand_demo_nofvc': {
        'use_cnn_features':  True,
        'use_hand_features': True,
        'use_demographics':  True,
        'use_fvc_baseline':  False,
        'pooling_type':      'max_mean',
        'description': '[BLOCK 4] Full model – NO FVC(0)',
    },
}


BASE_CONFIG = {
    # Paths
    "gt_path":               Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
    "ct_scan_path":          Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy"),
    "patient_features_path": Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv"),
    "kfold_splits_path":     Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl"),
    "train_csv_path":        Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\train.csv"),

    # Reproducibility
    'base_seed': 42,

    # Model parameters
    'backbone':       'resnet50',
    'image_size':     (224, 224),
    'pooling_type':   'max_mean',   # updated from Block 2: max_mean beat max and mean
    'use_fvc_branch': True,

    # Training
    'batch_size':               8,
    'learning_rate':            5e-4,
    'weight_decay':             0.05,
    'epochs':                   100,
    'early_stopping_patience':  20,
    'use_scheduler':            True,

    # Architecture
    'hidden_dims':    [256, 128],
    'dropout':        0.7,
    'use_batch_norm': True,

    'resume_from_checkpoint': True,
    'normalization_type':     'standard',
}

HAND_FEATURE_COLS = [
    'ApproxVol_30_60',
    'Avg_NumTissuePixel_30_60',
    'Avg_Tissue_30_60',
    'Avg_Tissue_thickness_30_60',
    'Avg_TissueByTotal_30_60',
    'Avg_TissueByLung_30_60',
    'Mean_30_60',
    'Skew_30_60',
    'Kurtosis_30_60',
]
DEMO_FEATURE_COLS = ['Age', 'Sex', 'SmokingStatus']


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"\n{'='*70}\n  RANDOM SEED SET TO: {seed}\n{'='*70}")


# =============================================================================
# DEMOGRAPHICS
# =============================================================================

def load_and_merge_demographics(train_csv_path: Path, patient_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load demographics from train.csv and merge with patient_features."""
    train_df = pd.read_csv(train_csv_path)

    print("\n" + "="*70)
    print("LOADING DEMOGRAPHICS FROM TRAIN.CSV")
    print("="*70)

    demo_cols = ['Patient']
    for col in DEMO_FEATURE_COLS:
        if col in train_df.columns:
            demo_cols.append(col)
            print(f"  Found: {col}")
        else:
            print(f"  ⚠️  Missing: {col}")

    demographics_df = train_df[demo_cols].copy().drop_duplicates('Patient')
    enhanced_df     = patient_features_df.merge(demographics_df, on='Patient', how='left')

    if 'Sex' in enhanced_df.columns:
        enhanced_df['Sex'] = enhanced_df['Sex'].map({'Male': 1, 'Female': 0})
        print(f"\nEncoding Sex: Male=1, Female=0")

    if 'SmokingStatus' in enhanced_df.columns:
        enhanced_df['SmokingStatus'] = enhanced_df['SmokingStatus'].map(
            {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}
        )
        print(f"Encoding SmokingStatus: Never=0, Ex=1, Current=2")

    missing = enhanced_df[demo_cols[1:]].isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing demographic values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} missing")

    return enhanced_df


def preprocess_demographics_improved(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    normalization_type: str = 'standard',
) -> tuple[pd.DataFrame, dict]:
    """Preprocess demographics with proper centering and encoding."""
    result_df     = features_df.copy()
    encoding_info = {}

    train_patient_df = (
        result_df[result_df['patient_id'].isin(train_patient_ids)]
        .groupby('patient_id').first().reset_index()
    )

    # Age
    if 'Age' in result_df.columns:
        print("\n=== PREPROCESSING AGE ===")
        if normalization_type == 'standard':
            age_scaler = StandardScaler()
        elif normalization_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            age_scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            age_scaler = MinMaxScaler(feature_range=(-1, 1))

        age_scaler.fit(train_patient_df[['Age']].values)
        result_df['Age_normalized'] = age_scaler.transform(result_df[['Age']].values)
        encoding_info['age_scaler'] = age_scaler
        print(f"  Normalized (train mean≈0, std≈1)")

    # Sex
    if 'Sex' in result_df.columns:
        result_df['Sex_encoded'] = result_df['Sex'].map({0: -1, 1: 1})
        encoding_info['sex_encoding'] = {0: -1, 1: 1}
        print("\n=== PREPROCESSING SEX: Female=-1, Male=1 ===")

    # SmokingStatus
    if 'SmokingStatus' in result_df.columns:
        print("\n=== PREPROCESSING SMOKING STATUS (one-hot, centered) ===")
        smoking_dummies = pd.get_dummies(result_df['SmokingStatus'], prefix='Smoking', dtype=float)
        smoking_dummies = smoking_dummies - 0.5
        result_df       = pd.concat([result_df, smoking_dummies], axis=1)
        smoking_cols    = sorted(smoking_dummies.columns.tolist())
        encoding_info['smoking_columns'] = smoking_cols
        print(f"  Encoded into: {smoking_cols}")

    # Drop original columns
    cols_to_drop = []
    if 'Age_normalized'  in result_df.columns and 'Age'           in result_df.columns: cols_to_drop.append('Age')
    if 'Sex_encoded'     in result_df.columns and 'Sex'           in result_df.columns: cols_to_drop.append('Sex')
    if encoding_info.get('smoking_columns')    and 'SmokingStatus' in result_df.columns: cols_to_drop.append('SmokingStatus')
    if cols_to_drop:
        result_df.drop(cols_to_drop, axis=1, inplace=True)

    return result_df, encoding_info


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_features_per_fold(
    features_df: pd.DataFrame,
    train_patient_ids: list,
    hand_feature_cols: list,
    demo_feature_cols: list,
    normalization_type: str = 'standard',
) -> tuple[pd.DataFrame, dict]:
    """Normalize all features using ONLY training-set statistics."""
    result_df = features_df.copy()
    scalers   = {}

    train_patient_df = (
        result_df[result_df['patient_id'].isin(train_patient_ids)]
        .groupby('patient_id').first().reset_index()
    )
    print(f"\nNormalizing features ({normalization_type}) – {len(train_patient_df)} training patients")

    def _make_scaler(norm_type):
        if norm_type == 'standard':
            return StandardScaler()
        elif norm_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()

    # FVC
    fvc_cols = [c for c in ['baselinefvc', 'gt_fvc52'] if c in result_df.columns]
    if fvc_cols:
        print(f"\n  FVC columns: {fvc_cols}")
        fvc_scaler = _make_scaler(normalization_type)
        fvc_scaler.fit(train_patient_df[fvc_cols].values)
        result_df[fvc_cols] = fvc_scaler.transform(result_df[fvc_cols].values)
        scalers['fvc'] = fvc_scaler
        print(f"  ✓ FVC normalised")

    # Hand-crafted
    available_hand = [c for c in hand_feature_cols if c in result_df.columns]
    if available_hand:
        print(f"\n  Hand-crafted: {len(available_hand)} features")
        hand_scaler = _make_scaler(normalization_type)
        hand_scaler.fit(train_patient_df[available_hand].values)
        result_df[available_hand] = hand_scaler.transform(result_df[available_hand].values)
        scalers['hand'] = hand_scaler
        print(f"  ✓ Hand-crafted normalised")

    # Demographics
    available_demo = [c for c in demo_feature_cols if c in result_df.columns]
    if available_demo:
        result_df, encoding_info = preprocess_demographics_improved(
            result_df, train_patient_ids, normalization_type
        )
        scalers['demo_encoding'] = encoding_info
        print(f"  ✓ Demographics preprocessed")

    return result_df, scalers


# =============================================================================
# FEATURE SET CONSTRUCTION
# =============================================================================

def create_feature_set_for_fold(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,
    ablation_config: dict,
    normalization_type: str = 'standard',
) -> tuple[pd.DataFrame, dict]:
    """Create normalised feature set for a specific fold and ablation config."""
    print("\n" + "="*70)
    print(f"FEATURE SET: {ablation_config['description']}")
    print("="*70)

    result_df = slice_features_df.copy()

    if 'Patient' in result_df.columns:
        result_df.drop('Patient', axis=1, inplace=True)

    # Drop CNN features if not needed
    if not ablation_config['use_cnn_features']:
        cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
        if cnn_cols:
            result_df.drop(cnn_cols, axis=1, inplace=True)
            print(f"  Dropped {len(cnn_cols)} CNN feature columns (use_cnn_features=False)")

    patient_level_cols = ['Patient']
    hand_to_add = demo_to_add = []

    if ablation_config['use_hand_features']:
        hand_to_add        = [f for f in HAND_FEATURE_COLS if f in patient_features_df.columns]
        patient_level_cols.extend(hand_to_add)
        print(f"  Adding {len(hand_to_add)} hand-crafted features")

    if ablation_config['use_demographics']:
        demo_to_add        = [f for f in DEMO_FEATURE_COLS if f in patient_features_df.columns]
        patient_level_cols.extend(demo_to_add)
        print(f"  Adding {len(demo_to_add)} demographic features")

    if len(patient_level_cols) > 1:
        patient_subset = patient_features_df[patient_level_cols].drop_duplicates(
            subset='Patient', keep='first'
        )
        result_df = result_df.merge(
            patient_subset, left_on='patient_id', right_on='Patient', how='left'
        )
        result_df.drop('Patient', axis=1, inplace=True)

        all_cols_to_check = hand_to_add + demo_to_add
        missing = result_df[all_cols_to_check].isnull().sum()
        if missing.any():
            train_stats = result_df[
                result_df['patient_id'].isin(fold_data['train'])
            ].groupby('patient_id').first()
            for col in all_cols_to_check:
                if result_df[col].isnull().any():
                    fill_val = train_stats[col].median() if col in hand_to_add else 0
                    result_df[col].fillna(fill_val, inplace=True)
                    print(f"    {col}: filled {missing[col]} NaN → {fill_val:.2f}")

    result_df, scalers = normalize_features_per_fold(
        features_df=result_df,
        train_patient_ids=fold_data['train'],
        hand_feature_cols=hand_to_add,
        demo_feature_cols=demo_to_add,
        normalization_type=normalization_type,
    )

    cnn_cols = [c for c in result_df.columns if c.startswith('cnn_feature_')]
    print(f"\nFinal feature composition:")
    print(f"  CNN features:  {len(cnn_cols)}")
    print(f"  FVC(0):        {'YES' if ablation_config.get('use_fvc_baseline', True) else 'NO'}")
    print(f"  Hand-crafted:  {len(hand_to_add)}")
    print(f"  Demographic:   {len(demo_to_add)}")

    return result_df, scalers


# =============================================================================
# SINGLE FOLD TRAINING
# =============================================================================

def train_single_fold(
    features_df: pd.DataFrame,
    fold_data: dict,
    fold_idx: int,
    config: dict,
    results_dir: Path,
    scalers: dict,
    resume_from_checkpoint: bool = True,
    hand_feature_cols: list = None,
    demo_feature_cols: list = None,
):
    """Train FVC prediction model on a single fold."""
    if hand_feature_cols is None: hand_feature_cols = HAND_FEATURE_COLS
    if demo_feature_cols is None: demo_feature_cols = DEMO_FEATURE_COLS

    fold_dir        = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best_model.pth"

    # ── Resume check ──────────────────────────────────────────────────────────
    if resume_from_checkpoint and checkpoint_path.exists():
        print(f"\n{'='*70}\nCHECKPOINT FOUND FOR FOLD {fold_idx}")
        checkpoint  = torch.load(checkpoint_path, weights_only=False)
        is_complete = 'test_metrics' in checkpoint and 'val_mae' in checkpoint

        if is_complete:
            print("✓ Fold already completed – loading saved results.")
            print(f"  Val loss (normalised): {checkpoint['val_mae']:.4f}")
            print(f"  Test MAE:  {checkpoint['test_metrics']['mae']:.2f} mL")
            print(f"  Test RMSE: {checkpoint['test_metrics']['rmse']:.2f} mL")
            print(f"  Test R²:   {checkpoint['test_metrics']['r2']:.4f}")
            return {
                'fold_idx':               fold_idx,
                'val_mae':                checkpoint['val_mae'],
                'test_metrics':           checkpoint['test_metrics'],
                'loaded_from_checkpoint': True,
            }
        else:
            print("⚠ Incomplete checkpoint – retraining.")

    print(f"\n{'='*70}\nTRAINING FOLD {fold_idx}\n{'='*70}")

    train_ids, val_ids, test_ids = fold_data['train'], fold_data['val'], fold_data['test']
    print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

    use_fvc_baseline  = config.get('use_fvc_baseline', True)
    encoding_info     = scalers.get('demo_encoding', {})
    available_hand    = [c for c in hand_feature_cols if c in features_df.columns]

    # Build preprocessed demo column names — originals were dropped after preprocessing
    preprocessed_demo = []
    if demo_feature_cols:
        if 'Age_normalized' in features_df.columns:
            preprocessed_demo.append('Age_normalized')
        if 'Sex_encoded' in features_df.columns:
            preprocessed_demo.append('Sex_encoded')
        for col in encoding_info.get('smoking_columns', []):
            if col in features_df.columns:
                preprocessed_demo.append(col)

    print(f"  FVC(0) as input: {'YES' if use_fvc_baseline else 'NO – imaging/clinical only'}")
    if preprocessed_demo:
        print(f"  Demo cols: {preprocessed_demo}")
    else:
        print("  No demographic features for this experiment")

    train_loader, val_loader, test_loader = create_fvc_dataloaders(
        features_df,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=config['batch_size'],
        num_workers=4,
        hand_feature_cols=available_hand,
        demo_feature_cols=preprocessed_demo,
        encoding_info=encoding_info,
        include_fvc_baseline=use_fvc_baseline,   # Block 4 flag
    )

    sample_batch    = next(iter(train_loader))
    actual_cnn_dim  = sample_batch['cnn_features'].shape[2]
    hand_t          = sample_batch.get('hand_features')
    demo_t          = sample_batch.get('demo_features')
    actual_hand_dim = hand_t.shape[1] if (hand_t is not None and hasattr(hand_t, 'shape')) else 0
    actual_demo_dim = demo_t.shape[1] if (demo_t is not None and hasattr(demo_t, 'shape')) else 0

    print(f"\n  CNN dim: {actual_cnn_dim}  Hand dim: {actual_hand_dim}  Demo dim: {actual_demo_dim}")
    if actual_demo_dim == 0 and preprocessed_demo:
        print("  ⚠ WARNING: demo features expected but batch dim=0 – check dataloader")

    # Disable FVC branch when FVC(0) is zeroed (Block 4) — branch would just learn to ignore zeros
    use_fvc_branch = use_fvc_baseline and config.get('use_fvc_branch', True)

    model = FVCPredictionModel(
        cnn_feature_dim=actual_cnn_dim,
        hand_feature_dim=actual_hand_dim,
        demo_feature_dim=actual_demo_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        pooling_type=config['pooling_type'],
        use_fvc_branch=use_fvc_branch,
    )

    trainer = FVCModelTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        use_scheduler=config['use_scheduler'],
    )

    best_val_loss = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True,
    )

    trainer.plot_training_history(save_path=str(fold_dir / "training_history.png"))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*70}\nTEST SET EVALUATION\n{'='*70}")
    test_results = trainer.evaluate(test_loader)

    fvc_scaler = scalers['fvc']
    n     = len(test_results['targets'])
    dummy = np.zeros((n, 1))

    targets_denorm = fvc_scaler.inverse_transform(
        np.column_stack([dummy, test_results['targets']])
    )[:, 1]
    preds_denorm = fvc_scaler.inverse_transform(
        np.column_stack([dummy, test_results['predictions']])
    )[:, 1]

    mae_denorm  = mean_absolute_error(targets_denorm, preds_denorm)
    rmse_denorm = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
    r2_denorm   = r2_score(targets_denorm, preds_denorm)

    print(f"\nDenormalised Test Metrics (mL):")
    print(f"  MAE:  {mae_denorm:.2f} mL")
    print(f"  RMSE: {rmse_denorm:.2f} mL")
    print(f"  R²:   {r2_denorm:.4f}")

    plot_evaluation_metrics(
        y_true=targets_denorm, y_pred=preds_denorm,
        save_path=str(fold_dir / "test_evaluation.png"),
    )

    test_metrics = {
        'mae':      mae_denorm,
        'rmse':     rmse_denorm,
        'r2':       r2_denorm,
        'mae_norm': test_results['mae'],
        'r2_norm':  test_results['r2'],
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'config':           config,
        'fold_idx':         fold_idx,
        'val_mae':          best_val_loss,   # normalised MSE (early stopping criterion)
        'test_metrics':     test_metrics,    # denormalised mL
    }, checkpoint_path)

    pd.DataFrame({
        'patient_id':       test_ids,
        'true_fvc_ml':      targets_denorm,
        'predicted_fvc_ml': preds_denorm,
    }).to_csv(fold_dir / "test_predictions.csv", index=False)

    print(f"\n✓ Fold {fold_idx} complete → {fold_dir}")

    return {
        'fold_idx':               fold_idx,
        'val_mae':                best_val_loss,
        'test_metrics':           test_metrics,
        'loaded_from_checkpoint': False,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_fold_results(fold_results: list, save_path: Path):
    """Aggregate and print results across all folds."""
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("="*70)

    n_resumed = sum(1 for r in fold_results if r.get('loaded_from_checkpoint', False))
    n_trained = len(fold_results) - n_resumed
    print(f"  Folds trained from scratch: {n_trained}  |  Loaded from checkpoint: {n_resumed}")

    val_losses = [r['val_mae']              for r in fold_results]
    test_maes  = [r['test_metrics']['mae']  for r in fold_results]
    test_rmses = [r['test_metrics']['rmse'] for r in fold_results]
    test_r2s   = [r['test_metrics']['r2']   for r in fold_results]

    summary_df = pd.DataFrame({
        'Metric': ['Val Loss (normalised MSE)', 'Test MAE (mL)', 'Test RMSE (mL)', 'Test R²'],
        'Mean':   [np.mean(val_losses), np.mean(test_maes), np.mean(test_rmses), np.mean(test_r2s)],
        'Std':    [np.std(val_losses),  np.std(test_maes),  np.std(test_rmses),  np.std(test_r2s)],
    })
    print(summary_df.to_string(index=False))
    print("\nNote: Val Loss is normalised MSE. Test MAE/RMSE/R² are in original mL units.")

    summary_df.to_csv(save_path / "aggregate_metrics_summary.csv", index=False)

    detailed_df = pd.DataFrame([{
        'fold':      r['fold_idx'],
        'val_loss':  r['val_mae'],
        'test_mae':  r['test_metrics']['mae'],
        'test_rmse': r['test_metrics']['rmse'],
        'test_r2':   r['test_metrics']['r2'],
    } for r in fold_results])
    detailed_df.to_csv(save_path / "detailed_fold_results.csv", index=False)

    return summary_df, detailed_df


# =============================================================================
# ABLATION RUNNER
# =============================================================================

def run_ablation_study(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path,
):
    """Run the full FVC prediction ablation study (4 blocks)."""
    print("\n" + "="*70)
    print("FVC PREDICTION ABLATION STUDY")
    print("="*70)

    all_results = {}

    # Block membership: explicit flags — robust against any config name
    block_1 = {k: v for k, v in ABLATION_CONFIGS.items()
               if not v['use_cnn_features'] and v.get('use_fvc_baseline', True)}
    block_2 = {k: v for k, v in ABLATION_CONFIGS.items()
               if v['use_cnn_features'] and not v['use_hand_features']
               and v.get('use_fvc_baseline', True)}
    block_3 = {k: v for k, v in ABLATION_CONFIGS.items()
               if v['use_cnn_features'] and v['use_hand_features']
               and v.get('use_fvc_baseline', True)}
    block_4 = {k: v for k, v in ABLATION_CONFIGS.items()
               if not v.get('use_fvc_baseline', True)}

    print(f"  BLOCK 1 (Clinical baseline, with FVC(0)):    {list(block_1.keys())}")
    print(f"  BLOCK 2 (Imaging only, with FVC(0)):         {list(block_2.keys())}")
    print(f"  BLOCK 3 (Full multimodal, with FVC(0)):      {list(block_3.keys())}")
    print(f"  BLOCK 4 (FVC(0) ablation):                   {list(block_4.keys())}")
    print(f"\n  Block 3 pooling: '{base_config['pooling_type']}'")
    print(f"  (Update BASE_CONFIG['pooling_type'] after Block 2 if needed)")

    def _run_block(configs_subset: dict):
        for config_name, ablation_config in configs_subset.items():
            print("\n" + "="*80)
            print(f"EXPERIMENT: {config_name.upper()}")
            print("="*80)
            print(f"  {ablation_config['description']}")

            ablation_dir = results_base_dir / f"ablation_{config_name}"
            ablation_dir.mkdir(parents=True, exist_ok=True)
            fold_results = []

            for fold_idx in sorted(kfold_splits.keys()):
                fold_seed = base_config['base_seed'] + fold_idx * 100
                set_seed(fold_seed)

                fold_data = kfold_splits[fold_idx]
                features_df, scalers = create_feature_set_for_fold(
                    slice_features_df=slice_features_df,
                    patient_features_df=patient_features_df,
                    fold_data=fold_data,
                    ablation_config=ablation_config,
                    normalization_type=base_config['normalization_type'],
                )

                config = base_config.copy()
                config['results_save_dir'] = ablation_dir
                config['use_fvc_baseline'] = ablation_config.get('use_fvc_baseline', True)

                # Per-experiment pooling override (Block 2 + Block 4)
                if 'pooling_type' in ablation_config:
                    config['pooling_type'] = ablation_config['pooling_type']
                    print(f"  Pooling override: {ablation_config['pooling_type']}")

                result = train_single_fold(
                    features_df=features_df,
                    fold_data=fold_data,
                    fold_idx=fold_idx,
                    config=config,
                    results_dir=ablation_dir,
                    scalers=scalers,
                    resume_from_checkpoint=config['resume_from_checkpoint'],
                )
                fold_results.append(result)

            summary_df, detailed_df = aggregate_fold_results(fold_results, ablation_dir)
            all_results[config_name] = {
                'config':       ablation_config,
                'summary':      summary_df,
                'detailed':     detailed_df,
                'fold_results': fold_results,
            }
            print(f"\n✓ '{config_name}' complete!")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*80 + "\nBLOCK 1 – CLINICAL BASELINE\n" + "="*80)
    _run_block(block_1)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    print("\n" + "="*80 + "\nBLOCK 2 – IMAGING ONLY (POOLING COMPARISON)\n" + "="*80)
    _run_block(block_2)

    print("\n" + "="*80 + "\nBLOCK 2 POOLING RESULTS\n" + "="*80)
    b2_maes = {}
    for k in block_2:
        if k not in all_results:
            continue
        maes    = [f['test_metrics']['mae'] for f in all_results[k]['fold_results']]
        pooling = ABLATION_CONFIGS[k].get('pooling_type', base_config['pooling_type'])
        b2_maes[k] = np.mean(maes)
        print(f"  {pooling:10s}  MAE = {np.mean(maes):.2f} ± {np.std(maes):.2f} mL")
    if b2_maes:
        best_k = min(b2_maes, key=b2_maes.get)
        best_p = ABLATION_CONFIGS[best_k].get('pooling_type', base_config['pooling_type'])
        print(f"\n→ Best pooling: {best_p}  (update BASE_CONFIG['pooling_type'] before re-running if needed)")

    # ── Block 3 ───────────────────────────────────────────────────────────────
    print("\n" + "="*80 + f"\nBLOCK 3 – FULL MULTIMODAL (pooling='{base_config['pooling_type']}')\n" + "="*80)
    _run_block(block_3)

    # ── Block 4 ───────────────────────────────────────────────────────────────
    print("\n" + "="*80 + "\nBLOCK 4 – FVC(0) ABLATION\n" + "="*80)
    print("  Quantifies how much FVC(0) contributes vs imaging/handcrafted alone.")
    _run_block(block_4)

    # Block 4 contribution summary
    print("\n" + "="*80 + "\nBLOCK 4 – FVC(0) CONTRIBUTION SUMMARY\n" + "="*80)
    pairs = [
        ('hand_only',    'hand_only_nofvc',    'Hand-crafted only'),
        ('cnn_max_mean', 'cnn_only_nofvc',     'CNN only (max_mean)'),
        ('cnn_hand',     'cnn_hand_nofvc',     'CNN + Hand'),
        ('cnn_hand_demo','cnn_hand_demo_nofvc','Full model'),
    ]
    for with_k, no_k, label in pairs:
        if with_k not in all_results or no_k not in all_results:
            continue
        mae_with    = np.mean([f['test_metrics']['mae'] for f in all_results[with_k]['fold_results']])
        mae_without = np.mean([f['test_metrics']['mae'] for f in all_results[no_k]['fold_results']])
        r2_with     = np.mean([f['test_metrics']['r2']  for f in all_results[with_k]['fold_results']])
        r2_without  = np.mean([f['test_metrics']['r2']  for f in all_results[no_k]['fold_results']])
        print(f"\n  {label}:")
        print(f"    With FVC(0):    MAE={mae_with:.1f} mL   R²={r2_with:.3f}")
        print(f"    Without FVC(0): MAE={mae_without:.1f} mL   R²={r2_without:.3f}")
        print(f"    Contribution:   ΔMAE={mae_without - mae_with:+.1f} mL   ΔR²={r2_without - r2_with:+.3f}")

    create_ablation_comparison(all_results, results_base_dir)
    return all_results


# =============================================================================
# COMPARISON PLOT
# =============================================================================

def create_ablation_comparison(all_results: dict, results_dir: Path):
    """Bar chart comparison across all ablation experiments."""
    import matplotlib.pyplot as plt

    print("\n" + "="*70 + "\nABLATION COMPARISON\n" + "="*70)

    rows = []
    for config_name, results in all_results.items():
        s = results['summary']
        rows.append({
            'Configuration': config_name,
            'Description':   results['config']['description'],
            'Test_MAE':  s[s['Metric'] == 'Test MAE (mL)']['Mean'].values[0],
            'Test_RMSE': s[s['Metric'] == 'Test RMSE (mL)']['Mean'].values[0],
            'Test_R2':   s[s['Metric'] == 'Test R²']['Mean'].values[0],
        })

    df = pd.DataFrame(rows).sort_values('Test_MAE')
    df.to_csv(results_dir / "ablation_comparison.csv", index=False)
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\n🏆 BEST: {best['Configuration']}  MAE={best['Test_MAE']:.2f} mL  R²={best['Test_R2']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(len(df))
    for ax, col, ylabel, color in [
        (axes[0], 'Test_MAE',  'Test MAE (mL)',  'steelblue'),
        (axes[1], 'Test_RMSE', 'Test RMSE (mL)', 'seagreen'),
        (axes[2], 'Test_R2',   'Test R²',        'coral'),
    ]:
        ax.bar(x, df[col], alpha=0.8, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'].values, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {results_dir / 'ablation_comparison.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(BASE_CONFIG['base_seed'])

    results_base_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\2_fvc_prediction\ablation_no_fvc_results")
    results_base_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70 + "\nLOADING DATA\n" + "="*70)
    data_loader = IPFDataLoader(
        csv_path=BASE_CONFIG['gt_path'],
        features_path=BASE_CONFIG['patient_features_path'],
        npy_dir=BASE_CONFIG['ct_scan_path'],
    )
    patient_data, _ = data_loader.get_patient_data()

    print("\n" + "="*70 + "\nEXTRACTING CNN FEATURES\n" + "="*70)
    feature_extractor = CNNFeatureExtractor(
        model_name='resnet50',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    slice_features_df = feature_extractor.extract_features_patient_grouping(
        patient_data=patient_data,
        patients_per_batch=4,
        save_path=None,
    )

    print("\n" + "="*70 + "\nLOADING PATIENT FEATURES\n" + "="*70)
    patient_features_df = pd.read_csv(BASE_CONFIG['patient_features_path'])
    patient_features_df = load_and_merge_demographics(
        train_csv_path=BASE_CONFIG['train_csv_path'],
        patient_features_df=patient_features_df,
    )

    with open(BASE_CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    print(f"\nLoaded {len(kfold_splits)} folds")

    run_ablation_study(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        kfold_splits=kfold_splits,
        base_config=BASE_CONFIG,
        results_base_dir=results_base_dir,
    )

    print("\n" + "="*70 + "\n✓ ABLATION STUDY COMPLETE!\n" + "="*70)
    print(f"Results saved to: {results_base_dir}")


if __name__ == "__main__":
    main()