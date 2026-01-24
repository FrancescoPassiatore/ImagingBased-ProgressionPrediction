"""
Predict FVC@52 using MLP Corrector Slopes
=========================================

For each fold and each MLP corrector approach (demographics, handcrafted, full):
- Loads predicted slopes from MLP corrector
- Calculates FVC@52 for each patient
- Compares with ground truth FVC@52
- Saves predictions and metrics
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import (
    IPFDataLoader,
    predict_fvc_at_week,
    compute_fvc52_metrics
)

CONFIG = {
    'npy_dir': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Dataset/extracted_npy/extracted_npy',
    'train_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'features_csv': 'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training/CNN_Slope_Prediction/patient_features.csv',
    'mlp_predictions_dir': Path('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\MLP_Corrector\\Cyclic_kfold\\predictions\\mse'),
    'results_dir': Path('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\MLP_Corrector\\Cyclic_kfold\\fvc52_results\\mse'),
    'n_folds': 5,
    'feature_types': ['demographics', 'handcrafted', 'full'],
    'target_week': 52,
    'cnn_predictions_dir': Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/CNN_Training/Cyclic_kfold/predictions_trainings/predictions_mse'),
}

CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

# Load patient data
print("\n📁 Loading patient data...")
data_loader = IPFDataLoader(
    csv_path=CONFIG['train_csv'],
    features_path=CONFIG['features_csv'],
    npy_dir=CONFIG['npy_dir']
)
patient_data, features_data = data_loader.get_patient_data()
print(f"✓ Loaded {len(patient_data)} patients")

# Load K-Fold splits
splits_path = Path('D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/Kfold_cyclic/kfold_cyclic_splits.pkl')
with open(splits_path, 'rb') as f:
    splits = pickle.load(f)
print(f"✓ Loaded {len(splits)} folds")

summary_data = []

print("\n" + "="*80)
print("CNN-ONLY SLOPE → FVC@52")
print("="*80)

cnn_summary = []

for fold_idx in range(CONFIG['n_folds']):
    print(f"\nCNN-ONLY | FOLD {fold_idx+1}/{CONFIG['n_folds']}")

    cnn_pred_path = CONFIG['cnn_predictions_dir'] / f'cnn_predictions_fold{fold_idx}.pkl'
    if not cnn_pred_path.exists():
        print(f"  ⚠️ CNN predictions not found: {cnn_pred_path}")
        continue

    with open(cnn_pred_path, 'rb') as f:
        cnn_preds_all = pickle.load(f)

    test_ids = splits[fold_idx]['test'] if isinstance(splits[fold_idx], dict) else splits[fold_idx][2]

    fold_rows = []

    for pid in test_ids:
        if pid not in cnn_preds_all['test']:
            continue
        if pid not in patient_data:
            continue

        slope_pred = cnn_preds_all['test'][pid]
        baseline_fvc = patient_data[pid]['intercept']

        fvc52_pred = predict_fvc_at_week(
            baseline_fvc,
            slope_pred,
            CONFIG['target_week'],
            baseline_week=0.0
        )

        weeks = np.array(patient_data[pid]['weeks'])
        fvc_values = np.array(patient_data[pid]['fvc_values'])

        idx = np.argmin(np.abs(weeks - CONFIG['target_week']))
        if np.abs(weeks[idx] - CONFIG['target_week']) <= 8:
            fvc52_true = fvc_values[idx]
            has_true = True
        else:
            fvc52_true = None
            has_true = False

        fold_rows.append({
            'patient_id': pid,
            'baseline_fvc': baseline_fvc,
            'slope_pred': slope_pred,
            'fvc52_predicted': fvc52_pred,
            'fvc52_true': fvc52_true,
            'has_true_fvc52': has_true
        })

    df = pd.DataFrame(fold_rows)
    eval_df = df[df['has_true_fvc52']]

    if len(eval_df) > 0:
        metrics = compute_fvc52_metrics(
            eval_df['fvc52_true'].values,
            eval_df['fvc52_predicted'].values
        )
        print(f"  Patients evaluated: {len(eval_df)}")
        print(f"  MAE: {metrics['mae']:.2f} ml")
        print(f"  RMSE: {metrics['rmse']:.2f} ml")
        print(f"  R²: {metrics['r2']:.4f}")
        cnn_summary.append(metrics)

    df['fold'] = fold_idx
    df.to_csv(
        CONFIG['results_dir'] / f'cnn_only_fold{fold_idx}_fvc52_predictions.csv',
        index=False
    )

if cnn_summary:
    mae = [m['mae'] for m in cnn_summary]
    rmse = [m['rmse'] for m in cnn_summary]
    r2 = [m['r2'] for m in cnn_summary]

    print("\nCNN-ONLY AGGREGATE:")
    print(f"  MAE: {np.mean(mae):.2f} ± {np.std(mae):.2f} ml")
    print(f"  RMSE: {np.mean(rmse):.2f} ± {np.std(rmse):.2f} ml")
    print(f"  R²: {np.mean(r2):.4f} ± {np.std(r2):.4f}")



for feature_type in CONFIG['feature_types']:
    print(f"\n{'='*80}")
    print(f"MLP CORRECTOR: {feature_type.upper()}")
    print(f"{'='*80}")
    all_fold_metrics = []
    all_fold_predictions = []
    for fold_idx in range(CONFIG['n_folds']):
        print(f"\nFOLD {fold_idx+1}/{CONFIG['n_folds']}")
        # Load predictions
        pred_path = CONFIG['mlp_predictions_dir'] / f'{feature_type}_predictions_fold{fold_idx}.pkl'
        if not pred_path.exists():
            print(f"  ⚠️ Predictions not found: {pred_path}")
            continue
        with open(pred_path, 'rb') as f:
            predictions_all = pickle.load(f)
        # Use test set for evaluation
        test_ids = splits[fold_idx]['test'] if isinstance(splits[fold_idx], dict) else splits[fold_idx][2]
        fold_preds = []
        for pid in test_ids:
            if pid not in predictions_all['test']:
                continue
            slope_pred = predictions_all['test'][pid]
            if pid not in patient_data:
                continue
            baseline_fvc = patient_data[pid]['intercept']
            baseline_week = 0.0
            fvc52_pred = predict_fvc_at_week(baseline_fvc, slope_pred, CONFIG['target_week'], baseline_week)
            # Get true FVC@52
            weeks = np.array(patient_data[pid]['weeks'])
            fvc_values = np.array(patient_data[pid]['fvc_values'])
            week_52_idx = np.argmin(np.abs(weeks - CONFIG['target_week']))
            if np.abs(weeks[week_52_idx] - CONFIG['target_week']) <= 8:
                fvc52_true = fvc_values[week_52_idx]
                has_true = True
            else:
                fvc52_true = None
                has_true = False
            fold_preds.append({
                'patient_id': pid,
                'baseline_fvc': baseline_fvc,
                'slope_pred': slope_pred,
                'fvc52_predicted': fvc52_pred,
                'fvc52_true': fvc52_true,
                'has_true_fvc52': has_true
            })
        preds_df = pd.DataFrame(fold_preds)
        # Filter to patients with true FVC@52
        eval_df = preds_df[preds_df['has_true_fvc52']].copy()
        if len(eval_df) > 0:
            metrics = compute_fvc52_metrics(
                eval_df['fvc52_true'].values,
                eval_df['fvc52_predicted'].values
            )
            print(f"  Patients evaluated: {len(eval_df)}/{len(preds_df)}")
            print(f"  MAE: {metrics['mae']:.2f} ml")
            print(f"  RMSE: {metrics['rmse']:.2f} ml")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Mean % Error: {metrics['mean_pct_error']:.2f}%")
            all_fold_metrics.append(metrics)
        else:
            print(f"  No patients with FVC@52 measurements")
        # Save predictions
        preds_df['fold'] = fold_idx
        preds_df.to_csv(CONFIG['results_dir'] / f'{feature_type}_fold{fold_idx}_fvc52_predictions.csv', index=False)
        all_fold_predictions.append(preds_df)
    # Aggregate results
    if all_fold_metrics:
        mae_values = [m['mae'] for m in all_fold_metrics]
        rmse_values = [m['rmse'] for m in all_fold_metrics]
        r2_values = [m['r2'] for m in all_fold_metrics]
        pct_error_values = [m['mean_pct_error'] for m in all_fold_metrics]
        print(f"\n{feature_type.upper()} AGGREGATE:")
        print(f"  MAE: {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} ml")
        print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} ml")
        print(f"  R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
        print(f"  Mean % Error: {np.mean(pct_error_values):.2f} ± {np.std(pct_error_values):.2f}%")
        summary_data.append({
            'FeatureType': feature_type,
            'MAE_Mean': np.mean(mae_values),
            'MAE_Std': np.std(mae_values),
            'RMSE_Mean': np.mean(rmse_values),
            'RMSE_Std': np.std(rmse_values),
            'R2_Mean': np.mean(r2_values),
            'R2_Std': np.std(r2_values),
            'PctError_Mean': np.mean(pct_error_values),
            'PctError_Std': np.std(pct_error_values),
            'N_Folds': len(all_fold_metrics)
        })
    # Save combined predictions
    all_preds_df = pd.concat(all_fold_predictions, ignore_index=True)
    combined_path = CONFIG['results_dir'] / f'{feature_type}_all_folds_fvc52_predictions.csv'
    all_preds_df.to_csv(combined_path, index=False)
    print(f"  ✓ Saved: {combined_path}")
# Save summary
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_path = CONFIG['results_dir'] / 'fvc52_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("✅ FVC@52 MLP CORRECTOR PREDICTION COMPLETE")
print("="*80)
print(f"\nResults saved in: {CONFIG['results_dir']}")
