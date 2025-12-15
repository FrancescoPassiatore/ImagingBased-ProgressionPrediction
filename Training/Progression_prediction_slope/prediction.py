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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
#Get utilites from utilities.py
from .utilities import *


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
   
def correct_slope_with_features(slope_cnn,features_dict,corrector_model,scaler,device='cuda'):

    #Bild features vector
    feature_vector = np.array([
        slope_cnn,
        features_dict.get('age', 0.0),
        features_dict.get('sex', 0.0),
        features_dict.get('smoking_status', 0.0),
        features_dict.get('approx_vol', 0.0),
        features_dict.get('avg_num_tissue_pixel', 0.0),
        features_dict.get('avg_tissue', 0.0),
        features_dict.get('avg_tissue_thickness', 0.0),
        features_dict.get('avg_tissue_by_total', 0.0),
        features_dict.get('avg_tissue_by_lung', 0.0),
        features_dict.get('mean', 0.0),
        features_dict.get('skew', 0.0),
        features_dict.get('kurtosis', 0.0),
    ], dtype=float)

    
    # Handle NaN -> replace with mean value
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    # Scale features
    feature_scaled = scaler.transform([feature_vector])
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
    
    return is_progression, decline_percen

class ProgressionPredictor:
    """
    Complete pipeline for IPF progression prediction
    """
    
    def __init__(self, cnn_model, corrector_model, scaler, 
                 patient_data, features_data, slope_scaler=None,
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
        slope_cnn = aggregate_patient_slope(slope_predictions, aggregation='mean')
        
        # Step 3: Apply corrector (optional)
        if use_corrector and self.corrector is not None:
            features_dict = self.features_data.get(patient_id, {})
            slope_final_norm = correct_slope_with_features(
                slope_cnn, features_dict, self.corrector, self.scaler, self.device
            )
        else:
            slope_final_norm = slope_cnn
        
        # Denormalize slope
        if self.slope_scaler is not None:
            slope_final = self.slope_scaler.inverse_transform([[slope_final_norm]])[0][0]
        else:
            slope_final = slope_final_norm
        
        # Step 4: Get baseline FVC and predict trajectory
        pdata = self.patient_data[patient_id]
        baseline_fvc = pdata['fvc_values'][0]  # First measurement
        baseline_week = pdata['weeks'][0]
        
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

    # 1. Load models
    cnn_model = utilites.ImprovedSliceLevelCNN(backbone_name='efficientnet_b0', pretrained=False)
    cnn_model.load_state_dict(torch.load('D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth'))
    
    corrector_model = FlexibleSlopeCorrector(input_dim=13)
    corrector_model.load_state_dict(torch.load('D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\corrector_final.pth'))
    
    # Load scaler
    with open('D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\scaler.pkl', 'rb') as f:
        scaler, _ = pickle.load(f)
    
    # 2. Initialize predictor
    predictor = ProgressionPredictor(
        cnn_model=cnn_model,
        corrector_model=corrector_model,
        scaler=scaler,
        patient_data=patient_data,
        features_data=features_data,
        slope_scaler=slope_scaler,
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
    sample_patient_ids = test_patients[:3]  # First 3 patients
    plot_progression_predictions(predictor, test_ds, sample_patient_ids)
    
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
    print(f"  - Evaluated at: {PROGRESSION_TIMEPOINTS} weeks")