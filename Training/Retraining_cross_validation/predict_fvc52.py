"""
Predict FVC@52 and Evaluate All Approaches
===========================================

This script:
1. Loads trained models for all 4 approaches
2. Predicts FVC at week 52 using each approach
3. Evaluates performance against true FVC@52 values
4. Saves predictions and metrics for each fold
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')

from utilities import (
    IPFDataLoader,
    IPFSliceDataset,
    PatientBatchSampler,
    patient_group_collate,
    ImprovedSliceLevelCNN,
    SlopeCorrectorCNNOnly,
    SlopeCorrectorCNNHandcrafted,
    SlopeCorrectorCNNDemographics,
    SlopeCorrectorFull,
    extract_patient_features,
    HAND_FEATURE_ORDER,
    DEMOGRAPHIC_FEATURES,
    predict_fvc_at_week,
    compute_fvc52_metrics
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    
    # Data
    'image_size': (224, 224),
    'target_week': 52,  # Week to predict FVC at
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Input/Output
    'checkpoint_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Retraining_cross_validation\Training\Retraining_cross_validation\checkpoints'),
    'results_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Retraining_cross_validation\Training\Retraining_cross_validation\results'),
}

# Define approaches
APPROACHES = {
    'cnn_only': {
        'name': 'CNN Only',
        'feature_type': 'none',
        'model_class': SlopeCorrectorCNNOnly,
        'model_kwargs': {}
    },
    'cnn_handcrafted': {
        'name': 'CNN + Handcrafted',
        'feature_type': 'handcrafted',
        'model_class': SlopeCorrectorCNNHandcrafted,
        'model_kwargs': {'n_handcrafted': len(HAND_FEATURE_ORDER)}
    },
    'cnn_demographics': {
        'name': 'CNN + Demographics',
        'feature_type': 'demographics',
        'model_class': SlopeCorrectorCNNDemographics,
        'model_kwargs': {'n_demographics': len(DEMOGRAPHIC_FEATURES)}
    },
    'cnn_full': {
        'name': 'CNN + Handcrafted + Demographics',
        'feature_type': 'full',
        'model_class': SlopeCorrectorFull,
        'model_kwargs': {
            'n_handcrafted': len(HAND_FEATURE_ORDER),
            'n_demographics': len(DEMOGRAPHIC_FEATURES)
        }
    }
}

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def extract_cnn_slopes_for_patients(model, patient_ids, patient_data, features_data, 
                                    slope_scaler, device, image_size=(224, 224)):
    """Extract mean CNN slopes for each patient"""
    
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = IPFSliceDataset(
        patient_ids,
        patient_data,
        features_data,
        image_size=image_size,
        normalize_slope=True,
        slope_scaler=slope_scaler
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_sampler=PatientBatchSampler(
            dataset,
            patients_per_batch=4,
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract slopes
    model.eval()
    patient_slopes = defaultdict(list)
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            patient_ids_batch = batch['patient_ids']
            
            predictions = model(images).cpu().numpy()
            
            for pred, pid in zip(predictions, patient_ids_batch):
                patient_slopes[pid].append(pred)
    
    # Average slopes per patient
    mean_slopes = {pid: np.mean(slopes) for pid, slopes in patient_slopes.items()}
    
    return mean_slopes


def predict_fvc52_for_approach(fold_idx, approach_key, approach_config, patient_ids,
                               patient_data, features_data, config):
    """Predict FVC@52 for all patients using a specific approach"""
    
    # Load CNN model
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_checkpoint = config['checkpoint_dir'] / f'cnn_fold{fold_idx}.pth'
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, weights_only=True))
    cnn_model = cnn_model.to(config['device'])
    
    # Load slope scaler
    with open(config['checkpoint_dir'] / f'slope_scaler_fold{fold_idx}.pkl', 'rb') as f:
        slope_scaler = pickle.load(f)
    
    # Extract CNN slopes
    cnn_slopes_norm = extract_cnn_slopes_for_patients(
        cnn_model, patient_ids, patient_data, features_data,
        slope_scaler, config['device'], config['image_size']
    )
    
    # Load corrector model and scaler
    corrector_model = approach_config['model_class'](**approach_config['model_kwargs'])
    corrector_checkpoint = config['checkpoint_dir'] / f'{approach_key}_fold{fold_idx}.pth'
    
    if corrector_checkpoint.exists():
        corrector_model.load_state_dict(torch.load(corrector_checkpoint, weights_only=True))
    
    corrector_model = corrector_model.to(config['device'])
    corrector_model.eval()
    
    # Load corrector scaler
    scaler_path = config['checkpoint_dir'] / f'{approach_key}_scaler_fold{fold_idx}.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            corrector_scaler = pickle.load(f)
    else:
        corrector_scaler = None
    
    # Predict for each patient
    predictions = []
    
    for patient_id in tqdm(patient_ids, desc=f"Predicting {approach_config['name']}"):
        if patient_id not in cnn_slopes_norm or patient_id not in patient_data:
            continue
        
        # Get normalized CNN slope
        slope_cnn_norm = cnn_slopes_norm[patient_id]
        
        # Get additional features
        features = extract_patient_features(patient_id, features_data, approach_config['feature_type'])
        
        # Build input vector
        if approach_config['feature_type'] == 'none':
            input_vector = np.array([slope_cnn_norm])
        else:
            input_vector = np.concatenate([[slope_cnn_norm], features])
        
        # Normalize features
        if corrector_scaler is not None:
            input_vector = corrector_scaler.transform(input_vector.reshape(1, -1))[0]
        
        # Predict corrected slope (already denormalized from corrector)
        with torch.no_grad():
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(config['device'])
            slope_corrected = corrector_model(input_tensor).cpu().item()
        
        # Note: Corrector outputs raw (denormalized) slope, no need to inverse_transform
        
        # Get baseline FVC (intercept at week 0) and baseline week
        baseline_fvc = patient_data[patient_id]['intercept']
        baseline_week = 0.0  # Intercept is defined at week 0
        
        # Predict FVC@52
        fvc52_pred = predict_fvc_at_week(baseline_fvc, slope_corrected, config['target_week'], baseline_week)
        
        # Get true FVC@52 (if available)
        weeks = np.array(patient_data[patient_id]['weeks'])
        fvc_values = np.array(patient_data[patient_id]['fvc_values'])
        
        # Find closest measurement to week 52
        week_52_idx = np.argmin(np.abs(weeks - config['target_week']))
        if np.abs(weeks[week_52_idx] - config['target_week']) <= 8:  # Within 8 weeks
            fvc52_true = fvc_values[week_52_idx]
        else:
            fvc52_true = None
        
        predictions.append({
            'patient_id': patient_id,
            'baseline_fvc': baseline_fvc,
            'baseline_week': baseline_week,
            'slope_cnn_norm': slope_cnn_norm,
            'slope_corrected': slope_corrected,
            'fvc52_predicted': fvc52_pred,
            'fvc52_true': fvc52_true,
            'has_true_fvc52': fvc52_true is not None
        })
    
    return pd.DataFrame(predictions)


def get_true_fvc52(patient_id, patient_data, target_week=52, tolerance=8):
    """Get true FVC@52 for a patient (if available)"""
    if patient_id not in patient_data:
        return None
    
    weeks = np.array(patient_data[patient_id]['weeks'])
    fvc_values = np.array(patient_data[patient_id]['fvc_values'])
    
    # Find closest measurement to target week
    week_idx = np.argmin(np.abs(weeks - target_week))
    
    if np.abs(weeks[week_idx] - target_week) <= tolerance:
        return fvc_values[week_idx]
    else:
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("FVC@52 PREDICTION AND EVALUATION")
    print("="*80)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_loader = IPFDataLoader(
        CONFIG['csv_path'],
        CONFIG['features_path'],
        CONFIG['npy_dir']
    )
    
    patient_data, features_data = data_loader.get_patient_data()
    print(f"✓ Loaded {len(patient_data)} patients")
    
    # Load K-Fold splits
    splits_path = CONFIG['results_dir'] / 'kfold_splits.pkl'
    if not splits_path.exists():
        # Try alternate path
        splits_path = Path('Training/Retraining_cross_validation/Training/Retraining_cross_validation/results/kfold_splits.pkl')
    
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    print(f"✓ Loaded {len(splits)} folds")
    
    # Predict for each fold and approach
    all_fold_predictions = {key: [] for key in APPROACHES.keys()}
    all_fold_metrics = {key: [] for key in APPROACHES.keys()}
    
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{len(splits)}")
        print(f"Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        print(f"{'='*80}")
        
        # Predict for each approach
        for approach_key, approach_config in APPROACHES.items():
            print(f"\n{approach_config['name']}:")
            
            # Predict on TEST set (held-out data for final evaluation)
            predictions_df = predict_fvc52_for_approach(
                fold_idx,
                approach_key,
                approach_config,
                test_ids,  # Changed from val_ids to test_ids
                patient_data,
                features_data,
                CONFIG
            )
            
            # Filter to patients with true FVC@52
            eval_df = predictions_df[predictions_df['has_true_fvc52']].copy()
            
            if len(eval_df) > 0:
                # Compute metrics
                metrics = compute_fvc52_metrics(
                    eval_df['fvc52_true'].values,
                    eval_df['fvc52_predicted'].values
                )
                
                print(f"  Patients evaluated: {len(eval_df)}/{len(predictions_df)}")
                print(f"  MAE: {metrics['mae']:.2f} ml")
                print(f"  RMSE: {metrics['rmse']:.2f} ml")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  Mean % Error: {metrics['mean_pct_error']:.2f}%")
                
                all_fold_metrics[approach_key].append(metrics)
            else:
                print(f"  No patients with FVC@52 measurements")
                metrics = None
            
            # Filter out patients without valid FVC@52 before saving
            predictions_to_save = predictions_df[predictions_df['has_true_fvc52'] == True].copy()
            
            # Save predictions (only those with valid FVC@52)
            predictions_to_save['fold'] = fold_idx
            all_fold_predictions[approach_key].append(predictions_to_save)
            
            # Save individual fold predictions
            save_path = CONFIG['results_dir'] / f'{approach_key}_fold{fold_idx}_fvc52_predictions.csv'
            predictions_to_save.to_csv(save_path, index=False)
            print(f"  Saved {len(predictions_to_save)} patients (filtered from {len(predictions_df)} total)")
    
    # Aggregate results across folds
    print("\n" + "="*80)
    print("AGGREGATE RESULTS ACROSS FOLDS")
    print("="*80)
    
    summary_data = []
    
    for approach_key, approach_config in APPROACHES.items():
        print(f"\n{approach_config['name']}:")
        
        # Combine all fold predictions
        all_preds_df = pd.concat(all_fold_predictions[approach_key], ignore_index=True)
        
        # Save combined predictions
        combined_path = CONFIG['results_dir'] / f'{approach_key}_all_folds_fvc52_predictions.csv'
        all_preds_df.to_csv(combined_path, index=False)
        print(f"  ✓ Saved: {combined_path}")
        
        # Compute aggregate metrics
        if len(all_fold_metrics[approach_key]) > 0:
            mae_values = [m['mae'] for m in all_fold_metrics[approach_key]]
            rmse_values = [m['rmse'] for m in all_fold_metrics[approach_key]]
            r2_values = [m['r2'] for m in all_fold_metrics[approach_key]]
            pct_error_values = [m['mean_pct_error'] for m in all_fold_metrics[approach_key]]
            
            print(f"  MAE: {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} ml")
            print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} ml")
            print(f"  R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            print(f"  Mean % Error: {np.mean(pct_error_values):.2f} ± {np.std(pct_error_values):.2f}%")
            
            summary_data.append({
                'Approach': approach_config['name'],
                'MAE_Mean': np.mean(mae_values),
                'MAE_Std': np.std(mae_values),
                'RMSE_Mean': np.mean(rmse_values),
                'RMSE_Std': np.std(rmse_values),
                'R2_Mean': np.mean(r2_values),
                'R2_Std': np.std(r2_values),
                'PctError_Mean': np.mean(pct_error_values),
                'PctError_Std': np.std(pct_error_values),
                'N_Folds': len(all_fold_metrics[approach_key])
            })
    
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
    print("✅ FVC@52 PREDICTION COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {CONFIG['results_dir']}")


if __name__ == "__main__":
    main()
