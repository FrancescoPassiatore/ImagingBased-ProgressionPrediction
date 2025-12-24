"""
IPF PROGRESSION PREDICTION PIPELINE

Pipeline Flow:
1. CNN → Slope (per-slice predictions)
2. Aggregate → Patient-level slope
3. Corrector → Final corrected slope (using handcrafted + demographics)
4. FVC Prediction → Predict FVC at future timepoints
5. Progression Classification → ≥10% decline = Progression

Progression Definition (Standard IPF Clinical Criteria):
- ≥10% relative decline in FVC from baseline within follow-up period
"""
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
#Get utilites from utilities.py
from utilities import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

def get_patient_progression_labels(patient_data):
    labels = {}
    for pid, pdata in patient_data.items():
        fvc = np.asarray(pdata["fvc_values"])
        if len(fvc) < 2:
            continue
        baseline = fvc[0]
        future = fvc[1:]
        _, decline_pct = classify_progression(baseline, future)
        labels[pid] = int(np.any(decline_pct >= PROGRESSION_THRESHOLD_PERCENT))
    return labels

#Progression Thresholds
PROGRESSION_THRESHOLD_PERCENT = 10.0  # 10% relative decline in FVC

def predict_slope_from_cnn(cnn_model,images,device='cuda'):

    cnn_model.eval()
    images = images.to(device)

    with torch.no_grad():
        slopes = cnn_model(images).cpu().numpy()

    return slopes  

def aggregate_patient_slope(slice_slopes: np.ndarray) :

    return np.mean(slice_slopes)
   
def correct_slope_with_features(slope_cnn, features_dict, corrector_model, scaler,feature_cols, device='cuda'):
    """
    Correct slope using the full feature set in EXACT training order
    """
    
    feature_vector=[]

    for col in feature_cols:
        if col == 'slope_cnn_mean':
            feature_vector.append(slope_cnn)
        else:
            feature_vector.append(features_dict.get(col, 0.0))  # Default to 0.0 if missing

    feature_vector = np.array(feature_vector, dtype=float)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    
    # CRITICAL: reshape to (1, n_features) for single sample
    feature_scaled = scaler.transform(feature_vector.reshape(1, -1))
    
    print(f"Scaled features (first 5): {feature_scaled[0, :5]}")
    print(f"  scaled_slope_cnn={feature_scaled[0, 0]:.4f}")
    
    # Convert to tensor
    feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32).to(device)
    
    # Predict corrected slope
    corrector_model.eval()
    with torch.no_grad():
        slope_corrected = corrector_model(feature_tensor).cpu().item()
    
    return slope_corrected
    
def predict_fvc_trajectory(baseline_fvc, slope, weeks):
    """
    Step 4: Predict FVC at future timepoints using linear model
    
    Args:
        baseline_fvc: float, FVC at baseline (ml)
        slope: float, rate of decline (ml/week)
        weeks: array-like, timepoints to predict
    
    Returns:
        fvc_predicted: array, predicted FVC at each timepoint
    """
    weeks = np.array(weeks)
    fvc_predicted = baseline_fvc + slope * weeks
    
    return fvc_predicted

def classify_progression(baseline_fvc, future_fvc, threshold_percent=10.0):
    """
    Step 5: Classify progression based on FVC decline
    
    Args:
        baseline_fvc: float, FVC at baseline (ml)
        future_fvc: float or array, FVC at future timepoint(s)
        threshold_percent: float, percentage decline threshold (default 10%)
    
    Returns:
        is_progression: bool or array, True if progression detected
        decline_percent: float or array, percentage decline
    """
    decline_absolute = baseline_fvc - future_fvc
    decline_percent = 100 * decline_absolute / baseline_fvc
    
    is_progression = decline_percent >= threshold_percent
    
    return is_progression, decline_percent

class ProgressionPredictor:
    """
    Complete pipeline for IPF progression prediction
    """
    
    
    def __init__(self, cnn_model, corrector_model, scaler, 
                 patient_data, features_data, slope_scaler=None, feature_cols=None,
                 device='cuda'):
        """
        Args:
            cnn_model: Trained CNN
            corrector_model: Trained slope corrector
            scaler: StandardScaler for corrector features
            patient_data: Dict with 'intercept', 'weeks', 'fvc_values'
            features_data: Dict with handcrafted + demographics
            slope_scaler: StandardScaler for slope denormalization (optional)
            device: cuda/cpu
        """
        self.cnn = cnn_model.to(device).eval()
        self.corrector = corrector_model.to(device).eval()
        self.scaler = scaler
        self.patient_data = patient_data
        self.features_data = features_data
        self.slope_scaler = slope_scaler
        self.feature_cols = feature_cols
        self.device = device
    
    def predict_patient(self, patient_id, dataset, use_corrector=True):
        """
        Predict slope and progression for a single patient
        
        Args:
            patient_id: str, patient ID
            dataset: Dataset with patient slices
            use_corrector: bool, whether to use corrector (vs CNN-only)
        
        Returns:
            results: Dict with predictions and progression status
        """
        # Get patient slices
        if patient_id not in dataset.patient_to_indices:
            return None
        
        patient_indices = dataset.patient_to_indices[patient_id]
        
        # Step 1: CNN predictions for all slices
        slice_images = []
        for idx in patient_indices:
            sample = dataset[idx]
            if sample is not None:
                slice_images.append(sample['image'])
        
        if len(slice_images) == 0:
            return None
        
        images = torch.stack(slice_images)
        slope_predictions = predict_slope_from_cnn(self.cnn, images, self.device)
        
        # Step 2: Aggregate to patient-level
        slope_cnn = aggregate_patient_slope(slope_predictions)
        
        # Step 3: Apply corrector (optional)
        if use_corrector and self.corrector is not None:
            features_dict = self.features_data.get(patient_id, {})
    
            print(f"\n{'='*60}")
            print(f"Patient {patient_id} - Feature Correction")
            print(f"{'='*60}")
            print(f"Input slope_cnn (normalized): {slope_cnn:.4f}")
            
            slope_final_norm = correct_slope_with_features(
                slope_cnn, 
                features_dict, 
                self.corrector, 
                self.scaler,
                self.feature_cols,
                self.device
            )
            
            print(f"Output slope_final_norm: {slope_final_norm:.4f}")
            print(f"{'='*60}\n")
        else:
            slope_final_norm = slope_cnn

        print(f"Post-feature_correction: Patient {patient_id}: Slope CNN={slope_cnn:.3f}, Slope Final Norm={slope_final_norm:.3f}")
        
        # Denormalize slope
        if self.slope_scaler is not None:
            slope_final = self.slope_scaler.inverse_transform([[slope_final_norm]])[0][0]
        else:
            slope_final = slope_final_norm
        
        # Step 4: Get baseline FVC and predict trajectory
        pdata = self.patient_data[patient_id]
        baseline_fvc = pdata['fvc_values'][0]  # First measurement
        baseline_week = pdata['weeks'][0]
        PROGRESSION_TIMEPOINTS = pdata['weeks']  # weeks
        # Predict at standard timepoints
        future_weeks = np.array(PROGRESSION_TIMEPOINTS)
        weeks_from_baseline = future_weeks - baseline_week
        
        fvc_predicted = predict_fvc_trajectory(baseline_fvc, slope_final, weeks_from_baseline)
        
        # Step 5: Classify progression
        is_progression, decline_percent = classify_progression(
            baseline_fvc, fvc_predicted, threshold_percent=PROGRESSION_THRESHOLD_PERCENT
        )
        
        # Package results
        results = {
            'patient_id': patient_id,
            'slope_cnn': slope_cnn,
            'slope_final': slope_final,
            'baseline_fvc': baseline_fvc,
            'predicted_fvc': fvc_predicted,
            'timepoints_weeks': future_weeks,
            'is_progression': is_progression,
            'decline_percent': decline_percent,
            'progression_detected': np.any(is_progression),
        }
        
        return results
    
    def predict_test_set(self, test_dataset, test_patients, use_corrector=True):
        """
        Predict progression for entire test set
        
        Args:
            test_dataset: Test dataset
            test_patients: List of patient IDs
            use_corrector: bool, use corrector vs CNN-only
        
        Returns:
            results_df: DataFrame with all predictions
        """
        all_results = []
        
        for patient_id in tqdm(test_patients, desc="Predicting progression"):
            result = self.predict_patient(patient_id, test_dataset, use_corrector)
            
            if result is None:
                continue
            
            # Flatten results for DataFrame
            for i, week in enumerate(result['timepoints_weeks']):
                all_results.append({
                    'patient_id': result['patient_id'],
                    'timepoint_weeks': week,
                    'slope_cnn': result['slope_cnn'],
                    'slope_final': result['slope_final'],
                    'baseline_fvc': result['baseline_fvc'],
                    'predicted_fvc': result['predicted_fvc'][i],
                    'decline_percent': result['decline_percent'][i],
                    'is_progression': result['is_progression'][i],
                })
        
        results_df = pd.DataFrame(all_results)
        return results_df
    
    def evaluate_progression_accuracy(self, test_dataset, test_patients, 
                                     true_progression_labels=None):
        """
        Evaluate progression prediction accuracy
        
        Args:
            test_dataset: Test dataset
            test_patients: List of patient IDs
            true_progression_labels: Dict {patient_id: bool} with true labels
        
        Returns:
            metrics: Dict with accuracy, sensitivity, specificity
        """
        results_df = self.predict_test_set(test_dataset, test_patients)
        
        if true_progression_labels is None:
            # Try to infer from actual FVC measurements
            true_progression_labels = self._infer_true_progression(test_patients)
        
        # Get predictions per patient (any timepoint showing progression)
        patient_predictions = results_df.groupby('patient_id')['is_progression'].any()
        
        # Compare to true labels
        y_true = []
        y_pred = []
        
        for patient_id in patient_predictions.index:
            if patient_id in true_progression_labels:
                y_true.append(true_progression_labels[patient_id])
                y_pred.append(patient_predictions[patient_id])
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'n_patients': len(y_true),
            'n_progression': y_true.sum(),
            'n_stable': (~y_true).sum(),
        }
        
        return metrics, results_df
    
    def _infer_true_progression(self, test_patients):
        """
        Infer true progression from actual FVC measurements
        """
        true_labels = {}
        
        for patient_id in test_patients:
            if patient_id not in self.patient_data:
                continue
            
            pdata = self.patient_data[patient_id]
            fvc_values = np.array(pdata['fvc_values'])
            
            if len(fvc_values) < 2:
                continue
            
            baseline_fvc = fvc_values[0]
            future_fvc = fvc_values[1:]
            
            # Check if any measurement shows ≥10% decline
            _, decline_percent = classify_progression(baseline_fvc, future_fvc)
            true_labels[patient_id] = np.any(decline_percent >= PROGRESSION_THRESHOLD_PERCENT)
        
        return true_labels

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_progression_predictions(predictor, test_dataset, patient_ids, 
                                 save_path=None):
    """
    Visualize progression predictions for sample patients
    """
    import matplotlib.pyplot as plt
    
    n_patients = len(patient_ids)
    fig, axes = plt.subplots(n_patients, 1, figsize=(10, 4*n_patients))
    
    if n_patients == 1:
        axes = [axes]
    
    for i, patient_id in enumerate(patient_ids):
        ax = axes[i]
        
        # Get prediction
        result = predictor.predict_patient(patient_id, test_dataset)
        
        if result is None:
            continue
        
        # Get true measurements
        pdata = predictor.patient_data[patient_id]
        true_weeks = np.array(pdata['weeks'])
        true_fvc = np.array(pdata['fvc_values'])
        
        # Plot true FVC
        ax.scatter(true_weeks, true_fvc, s=100, color='black', 
                  marker='o', label='True FVC', zorder=5)
        ax.plot(true_weeks, true_fvc, 'k--', alpha=0.3, linewidth=2)
        
        # Plot predicted trajectory
        baseline_week = true_weeks[0]
        pred_weeks = result['timepoints_weeks']
        pred_fvc = result['predicted_fvc']
        
        # Extend line to predictions
        all_weeks = np.concatenate([[baseline_week], pred_weeks])
        all_fvc = np.concatenate([[result['baseline_fvc']], pred_fvc])
        
        ax.plot(all_weeks, all_fvc, '-', linewidth=2.5, 
               color='#e74c3c', label='Predicted Trajectory', alpha=0.8)
        
        # Mark progression threshold
        threshold_fvc = result['baseline_fvc'] * (1 - PROGRESSION_THRESHOLD_PERCENT/100)
        ax.axhline(threshold_fvc, color='red', linestyle=':', 
                  label=f'{PROGRESSION_THRESHOLD_PERCENT}% decline threshold', 
                  linewidth=2)
        
        # Highlight progression timepoints
        for j, (week, fvc, prog) in enumerate(zip(pred_weeks, pred_fvc, 
                                                   result['is_progression'])):
            color = 'red' if prog else 'green'
            marker = 'X' if prog else 'o'
            ax.scatter([week], [fvc], s=150, color=color, marker=marker, 
                      edgecolors='black', linewidths=2, zorder=10)
        
        # Formatting
        ax.set_xlabel('Weeks from Baseline', fontsize=12)
        ax.set_ylabel('FVC (ml)', fontsize=12)
        
        prog_status = "PROGRESSION" if result['progression_detected'] else "STABLE"
        ax.set_title(f'Patient {patient_id}\n'
                    f'Slope: {result["slope_final"]:.2f} ml/week | '
                    f'Status: {prog_status}', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import warnings

    # Disabilita tutti i warning
    warnings.filterwarnings('ignore')

    # Oppure disabilita solo i warning specifici
    warnings.filterwarnings('ignore', category=UserWarning)

    # Oppure ancora più specifico
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    # 1. Load models
    cnn_model = ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', map_location=torch.device('cpu')))
    
    corrector_model = SlopeCorrector(input_dim=13)
    corrector_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope\files\corrector_full.pth', map_location=torch.device('cpu')))
    
    # Load scaler
    with open(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope\files\scaler_full.pkl', 'rb') as f:
        scaler, feature_cols = pickle.load(f)

    print(f"✓ Loaded corrector with {len(feature_cols)} features:")
    print(f"  Features: {feature_cols}")

    # Verify scaler looks reasonable
    print(f"\n{'='*70}")
    print("LOADED SCALER PARAMETERS:")
    print(f"{'='*70}")
    for i, col in enumerate(feature_cols):
        print(f"{col:25s}: mean={scaler.mean_[i]:12.2f}, std={scaler.scale_[i]:12.2f}")
    print(f"{'='*70}\n")

    # Paths
    CSV_PATH = 'Training/CNN_Slope_Prediction/train_with_coefs.csv'
    CSV_FEATURES_PATH = 'Training/CNN_Slope_Prediction/patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'

    # Hyperparameters
    IMAGE_SIZE = (224, 224)
    PATIENTS_PER_BATCH = 4

    # Device
    device = "cuda" 

    print(torch.__version__)
    print(torch.cuda.is_available())  # Should return True if CUDA is available

    # STEP 2: LOAD DATA

    print("\n" + "="*80)
    print("[1/10] LOADING DATA")
    print("="*80)

    dl = IPFDataLoader(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()

    print(f"\n✓ Loaded patient_data for {len(patient_data)} patients")
    print(f"✓ Loaded features_data for {len(features_data)} patients")

    # Verify data structure
    sample_patient = list(patient_data.keys())[0]
    print(f"\n📋 Sample patient data structure (ID: {sample_patient}):")
    for key, value in patient_data[sample_patient].items():
        if isinstance(value, list):
            print(f"   {key}: list with {len(value)} items")
        else:
            print(f"   {key}: {type(value).__name__} = {value}")

    print(f"\n📋 Sample feature data structure:")
    for key, value in features_data[sample_patient].items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    # =============================================================================
    # STEP 3: TRAIN/TEST SPLIT
    # =============================================================================

    """print("\n" + "="*80)
    print("[2/10] CREATING TRAIN/TEST SPLIT")
    print("="*80)

    all_patients = list(patient_data.keys())
    print(f"\n✓ Total patients available: {len(all_patients)}")

    # Recreate exact same split as training
    np.random.seed(42)
    np.random.shuffle(all_patients)

    test_size = int(len(all_patients) * 0.2)
    test_patients = all_patients[:test_size]
    train_patients = all_patients[test_size:]

    print(f"✓ Train patients: {len(train_patients)}")
    print(f"✓ Test patients:  {len(test_patients)}")
    print(f"\n📝 First 5 test patients: {test_patients[:5]}")
    print(f"📝 First 5 train patients: {train_patients[:5]}")

    # Verify no overlap
    overlap = set(train_patients) & set(test_patients)
    if overlap:
        print(f"❌ ERROR: {len(overlap)} patients in both train and test!")
    else:
        print(f"✓ No patient overlap between train and test")"""
        
    patient_labels = get_patient_progression_labels(patient_data)

    patient_ids = np.array(list(patient_labels.keys()))
    y_patients = np.array([patient_labels[p] for p in patient_ids])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_metrics = []

    print("\n" + "="*80)
    print("[CV] PATIENT-LEVEL STRATIFIED CROSS-VALIDATION")
    print("="*80)

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, y_patients), 1):
        print(f"\n--- Fold {fold} ---")

        train_patients = patient_ids[train_idx].tolist()
        val_patients   = patient_ids[val_idx].tolist()

        print(f"Train patients: {len(train_patients)}")
        print(f"Val patients:   {len(val_patients)}")
        print(f"Val positives: {sum(patient_labels[p] for p in val_patients)}")

        # --------------------------------------------------
        # Build datasets for this fold
        # --------------------------------------------------
        train_ds = IPFSliceDataset(
            train_patients,
            patient_data,
            features_data,
            normalize_slope=True,
            image_size=IMAGE_SIZE
        )

        val_ds = IPFSliceDataset(
            val_patients,
            patient_data,
            features_data,
            normalize_slope=True,
            image_size=IMAGE_SIZE
        )

        # IMPORTANT: share slope scaler
        val_ds.slope_scaler = train_ds.slope_scaler

        # --------------------------------------------------
        # Initialize predictor (models are frozen)
        # --------------------------------------------------
        predictor = ProgressionPredictor(
            cnn_model=cnn_model,
            corrector_model=corrector_model,
            scaler=scaler,
            patient_data=patient_data,
            features_data=features_data,
            feature_cols=feature_cols,
            slope_scaler=train_ds.slope_scaler,
            device=device
        )

        # --------------------------------------------------
        # Evaluate fold
        # --------------------------------------------------
        metrics, _ = predictor.evaluate_progression_accuracy(
            val_ds,
            val_patients
        )

        print(
            f"Fold {fold} | "
            f"Acc={metrics['accuracy']:.3f} | "
            f"Prec={metrics['precision']:.3f} | "
            f"Rec={metrics['recall']:.3f} | "
            f"F1={metrics['f1_score']:.3f}"
        )

        all_fold_metrics.append(metrics)
        
    print("\n" + "="*80)
    print("[CV] SUMMARY")
    print("="*80)

    def mean_std(key):
        vals = [m[key] for m in all_fold_metrics]
        return np.mean(vals), np.std(vals)

    for k in ["accuracy", "precision", "recall", "f1_score"]:
        m, s = mean_std(k)
        print(f"{k:10s}: {m:.3f} ± {s:.3f}")

    print(f"\nTotal patients evaluated: {sum(m['n_patients'] for m in all_fold_metrics)}")
    print(f"Total progressions: {sum(m['n_progression'] for m in all_fold_metrics)}")



"""
    # =============================================================================
    # STEP 4: CREATE DATASETS
    # =============================================================================

    print("\n" + "="*80)
    print("[3/10] CREATING DATASETS")
    print("="*80)

    train_ds = IPFSliceDataset(
        train_patients,
        patient_data,
        features_data,
        normalize_slope=True,
        image_size=IMAGE_SIZE
    )

    print(f"\n✓ Train dataset created:")
    print(f"   Total slices: {len(train_ds)}")
    print(f"   Patients: {len(train_ds.patients)}")
    print(f"   Slope scaler fitted: {train_ds.slope_scaler is not None}")

    test_ds = IPFSliceDataset(
        test_patients,
        patient_data,
        features_data,
        normalize_slope=True,
        image_size=IMAGE_SIZE
    )
    test_ds.slope_scaler = train_ds.slope_scaler

    print(f"\n✓ Test dataset created:")
    print(f"   Total slices: {len(test_ds)}")
    print(f"   Patients: {len(test_ds.patients)}")

    # Verify slope scaler
    if train_ds.slope_scaler:
        print(f"\n📊 Slope normalization:")
        print(f"   Mean: {train_ds.slope_scaler.mean_[0]:.2f} ml/week")
        print(f"   Std:  {train_ds.slope_scaler.scale_[0]:.2f} ml/week")

        # Test denormalization
        test_norm = np.array([[0.0]])
        test_denorm = train_ds.slope_scaler.inverse_transform(test_norm)[0][0]
        print(f"   Test: normalized=0.0 → denormalized={test_denorm:.2f}")

    # Sample a few items to verify loading
    print(f"\n🔍 Testing dataset loading (3 random samples)...")
    for i in np.random.choice(len(test_ds), 3, replace=False):
        sample = test_ds[i]
        if sample is not None:
            print(f"   ✓ Sample {i}: image shape={sample['image'].shape}, "
                f"slope={sample['slope'].item():.3f}, patient={sample['patient_id'][:10]}...")
        else:
            print(f"   ❌ Sample {i}: Failed to load")

    # =============================================================================
    # STEP 5: CREATE DATALOADERS
    # =============================================================================

    print("\n" + "="*80)
    print("[4/10] CREATING DATALOADERS")
    print("="*80)

    train_loader = DataLoader(
            train_ds,
            batch_sampler=PatientBatchSampler(
                train_ds,
                patients_per_batch=PATIENTS_PER_BATCH,
                shuffle=True  # ✓ Shuffle for training
            ),
            collate_fn=patient_group_collate,
            num_workers=4,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_ds,
        batch_sampler=PatientBatchSampler(
            test_ds,
            patients_per_batch=PATIENTS_PER_BATCH,
            shuffle=False
        ),
        collate_fn=patient_group_collate,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n✓ Test loader created:")
    print(f"   Total batches: {len(test_loader)}")
    print(f"   Patients per batch: {PATIENTS_PER_BATCH}")

    # Test loading one batch
    print(f"\n🔍 Testing batch loading...")
    try:
        test_batch = next(iter(test_loader))
        print(f"   ✓ Batch loaded successfully")
        print(f"   Images shape: {test_batch['images'].shape}")
        print(f"   Slopes shape: {test_batch['slopes'].shape}")
        print(f"   Patients in batch: {len(test_batch['patient_ids'])}")
        print(f"   Patient IDs: {test_batch['patient_ids']}")
        print(f"   Lengths: {test_batch['lengths'].tolist()}")
    except Exception as e:
        print(f"   ❌ Error loading batch: {e}")

    
    # 2. Initialize predictor
    predictor = ProgressionPredictor(
        cnn_model=cnn_model,
        corrector_model=corrector_model,
        scaler=scaler,
        patient_data=patient_data,
        features_data=features_data,
        feature_cols=feature_cols,
        slope_scaler=train_ds.slope_scaler,
        device='cuda'
    )
    
    # 3. Predict for test set
    results_df = predictor.predict_test_set(test_ds, test_patients)
    
    print(results_df.head())

    # 4. Evaluate accuracy
    metrics, _ = predictor.evaluate_progression_accuracy(
        test_ds, test_patients
    )
    
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    # 5. Visualize sample patients
    sample_patient_ids = test_patients[:5]  # First 3 patients
    plot_progression_predictions(predictor, test_ds, sample_patient_ids,save_path=r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope\output_progression')
    
    print("IPF Progression Prediction Pipeline")
    print("="*80)
    print("\nPipeline Steps:")
    print("  1. CNN → Per-slice slope predictions")
    print("  2. Aggregation → Patient-level slope")
    print("  3. Corrector → Refined slope (with features)")
    print("  4. FVC Prediction → Future FVC trajectory")
    print("  5. Classification → Progression vs Stable")
    print("\nProgression Criteria:")
    print(f"  - ≥{PROGRESSION_THRESHOLD_PERCENT}% relative FVC decline")
    print(f"  - Evaluated at: personalized weeks per patient")"""