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
        baseline_fvc = pdata['intercept']  # FVC intercept at baseline
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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score
        
        # Get probabilities for AUC calculation
        y_probs = results_df.groupby('patient_id')['decline_percent'].max().values
        y_probs_normalized = np.clip(y_probs / 10.0, 0, 1)  # Normalize to [0,1]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1] + 1e-7),
            'auc_roc': roc_auc_score(y_true, y_probs_normalized),
            'ap': average_precision_score(y_true, y_probs_normalized),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_curve': roc_curve(y_true, y_probs_normalized),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs_normalized,
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
    cnn_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\checkpoints_kfold_added_hf_tabular_attention_adj_lr\checkpoints_kfold_added_hf_tabular_attention_adj_lr\cnn_final.pth', map_location=torch.device('cpu'), weights_only=False))
    
    corrector_model = SlopeCorrector(input_dim=13)
    corrector_model.load_state_dict(torch.load(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope_1\files\corrector_full.pth', map_location=torch.device('cpu'), weights_only=False))
    
    # Load scaler
    with open(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope_1\files\scaler_full.pkl', 'rb') as f:
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
    # EVALUATE ALL PATIENTS
    # =============================================================================
    
    print("\n" + "="*80)
    print("[EVALUATION] PROGRESSION PREDICTION ON ALL PATIENTS")
    print("="*80)
    
    # Get progression labels for all patients
    patient_labels = get_patient_progression_labels(patient_data)
    all_patients = list(patient_labels.keys())
    
    print(f"\n✓ Total patients: {len(all_patients)}")
    print(f"✓ Progression cases: {sum(patient_labels.values())}")
    print(f"✓ Stable cases: {len(all_patients) - sum(patient_labels.values())}")
    
    # Create dataset with all patients
    dataset = IPFSliceDataset(
        all_patients,
        patient_data,
        features_data,
        normalize_slope=True,
        image_size=IMAGE_SIZE
    )
    
    print(f"\n✓ Dataset created:")
    print(f"   Total slices: {len(dataset)}")
    print(f"   Patients: {len(dataset.patients)}")
    
    # Initialize predictor (frozen models)
    predictor = ProgressionPredictor(
        cnn_model=cnn_model,
        corrector_model=corrector_model,
        scaler=scaler,
        patient_data=patient_data,
        features_data=features_data,
        feature_cols=feature_cols,
        slope_scaler=dataset.slope_scaler,
        device=device
    )
    
    print("\n✓ Predictor initialized with frozen models")
    
    # Evaluate all patients
    print("\nRunning predictions...")
    metrics, results_df = predictor.evaluate_progression_accuracy(
        dataset,
        all_patients
    )
    
    # Display results
    print("\n" + "="*80)
    print("PROGRESSION PREDICTION RESULTS")
    print("="*80)
    print(f"\nAccuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1_score']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"AP:        {metrics['ap']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    
    print(f"\nPatient Distribution:")
    print(f"  Total evaluated: {metrics['n_patients']}")
    print(f"  True progressions: {metrics['n_progression']}")
    print(f"  True stable: {metrics['n_stable']}")
    
    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    print("\n" + "="*80)
    print("[VISUALIZE] CREATING EVALUATION PLOTS")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = metrics['roc_curve']
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Stable', 'Progression'],
                yticklabels=['Stable', 'Progression'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 3. Metrics Bar Chart
    ax = axes[1, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1_score']
    ]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Classification Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_probs'])
    ax = axes[1, 1]
    ax.plot(recall, precision, color='#e74c3c', lw=2.5, label=f'PR Curve (AP = {metrics["ap"]:.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('progression_prediction_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✅ PLOTS SAVED: progression_prediction_evaluation.png")
    plt.show()
    
    # =============================================================================
    # SAVE PREDICTIONS
    # =============================================================================
    print("\n" + "="*80)
    print("[SAVE] SAVING PREDICTIONS")
    print("="*80)
    
    # Aggregate by patient for final predictions
    patient_predictions = []
    
    for patient_id in all_patients:
        patient_results = results_df[results_df['patient_id'] == patient_id]
        
        if len(patient_results) == 0:
            continue
        
        # Get maximum decline as confidence
        max_decline = patient_results['decline_percent'].max()
        is_progression = int(np.any(patient_results['is_progression']))
        
        # Probability: normalized decline (0-1 scale, 1.0 = 10% threshold)
        probability = float(max(0.0, min(max_decline / 10.0, 1.0)))
        
        # True label
        true_label = patient_labels.get(patient_id, None)
        
        patient_predictions.append({
            'patient_id': patient_id,
            'predicted_progression': is_progression,
            'true_progression': true_label,
            'probability': probability,
            'max_decline_percent': float(max_decline),
            'correct': (is_progression == true_label) if true_label is not None else None
        })
    
    # Save to CSV
    final_df = pd.DataFrame(patient_predictions)
    final_df.to_csv('prediction_fold_final.csv', index=False)
    
    print(f"\n✅ PREDICTIONS SAVED")
    print(f"   Location: prediction_fold_final.csv")
    print(f"   Total patients: {len(final_df)}")
    print(f"   Predicted progressions: {final_df['predicted_progression'].sum()}")
    print(f"   Predicted stable: {len(final_df) - final_df['predicted_progression'].sum()}")
    print(f"   Correct predictions: {final_df['correct'].sum()}")
    print(f"\n   Mean probability: {final_df['probability'].mean():.3f}")
    print(f"   Mean decline %: {final_df['max_decline_percent'].mean():.2f}%")
    
    print("\n📋 PREVIEW (first 10 rows):")
    print(final_df.head(10).to_string())
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)



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