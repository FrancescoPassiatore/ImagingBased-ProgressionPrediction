"""
Progression Assessment Utilities for FVC Prediction Training

Integrates progression risk assessment into the training pipeline:
- Computes progression on validation set for threshold optimization
- Applies optimized threshold to test set
- Avoids data leakage by using validation for threshold selection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


def compute_progression_from_fvc_predictions(
    baseline_fvc: np.ndarray,
    predicted_fvc_52: np.ndarray,
    true_fvc_52: np.ndarray,
    threshold: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Compute progression assessment from FVC predictions
    
    Args:
        baseline_fvc: Baseline FVC values (denormalized)
        predicted_fvc_52: Predicted week 52 FVC (denormalized)
        true_fvc_52: True week 52 FVC (denormalized)
        threshold: Percent decline threshold for progression
    
    Returns:
        Dictionary with progression predictions and metrics
    """
    # Compute percent decline
    percent_decline_pred = ((baseline_fvc - predicted_fvc_52) / baseline_fvc) * 100
    percent_decline_true = ((baseline_fvc - true_fvc_52) / baseline_fvc) * 100
    
    # Classify progression
    predicted_progression = (percent_decline_pred > threshold).astype(int)
    true_progression = (percent_decline_true > threshold).astype(int)
    
    return {
        'percent_decline_pred': percent_decline_pred,
        'percent_decline_true': percent_decline_true,
        'predicted_progression': predicted_progression,
        'true_progression': true_progression
    }


def find_optimal_threshold_on_validation(
    percent_decline_pred: np.ndarray,
    true_progression: np.ndarray,
    criterion: str = 'f1',
    threshold_range: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Find optimal progression threshold using validation set
    
    Args:
        percent_decline_pred: Predicted percent decline on validation set
        true_progression: True progression labels on validation set
        criterion: Optimization criterion ('f1', 'youden', 'accuracy', 'balanced', 'top_left')
        threshold_range: Array of thresholds to test
    
    Returns:
        Tuple of (optimal_threshold, metrics_dict, threshold_df)
    """
    if threshold_range is None:
        threshold_range = np.arange(0, 31, 1)  # 0% to 30% in 1% steps
    
    best_score = -np.inf if criterion != 'top_left' else np.inf
    best_threshold = 10.0
    best_metrics = {}
    
    results = []
    
    for threshold in threshold_range:
        predicted_prog = (percent_decline_pred > threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(true_progression, predicted_prog)
        prec = precision_score(true_progression, predicted_prog, zero_division=0)
        rec = recall_score(true_progression, predicted_prog, zero_division=0)
        f1 = f1_score(true_progression, predicted_prog, zero_division=0)
        
        cm = confusion_matrix(true_progression, predicted_prog)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sens + spec - 1
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = sens  # TPR = Sensitivity
        else:
            sens = 0
            spec = 0
            youden = 0
            fpr = 0
            tpr = 0
        
        # Calculate distance to top-left corner (0, 1) in ROC space
        # Top-left is at (FPR=0, TPR=1), so distance = sqrt((FPR-0)^2 + (TPR-1)^2)
        distance_to_top_left = np.sqrt(fpr**2 + (1 - tpr)**2)
        
        # Determine score based on criterion
        if criterion == 'f1':
            score = f1
        elif criterion == 'youden':
            score = youden
        elif criterion == 'accuracy':
            score = acc
        elif criterion == 'balanced':
            score = (f1 + acc + youden) / 3
        elif criterion == 'top_left':
            score = distance_to_top_left  # Lower is better
        else:
            score = f1
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'sensitivity': sens,
            'specificity': spec,
            'f1_score': f1,
            'youden_j': youden,
            'fpr': fpr,
            'tpr': tpr,
            'distance_to_top_left': distance_to_top_left,
            'score': score
        })
        
        # Update best based on criterion
        if criterion == 'top_left':
            if score < best_score:  # Lower distance is better
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'sensitivity': sens,
                    'specificity': spec,
                    'f1_score': f1,
                    'youden_j': youden,
                    'fpr': fpr,
                    'tpr': tpr,
                    'distance_to_top_left': distance_to_top_left,
                    'criterion': criterion,
                    'score': score
                }
        else:
            if score > best_score:  # Higher is better for other metrics
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'sensitivity': sens,
                    'specificity': spec,
                    'f1_score': f1,
                    'youden_j': youden,
                    'fpr': fpr,
                    'tpr': tpr,
                    'distance_to_top_left': distance_to_top_left,
                    'criterion': criterion,
                    'score': score
                }
    
    return best_threshold, best_metrics, pd.DataFrame(results)


def evaluate_progression_on_set(
    patient_ids: List[str],
    baseline_fvc: np.ndarray,
    predicted_fvc_52_norm: np.ndarray,
    true_fvc_52_norm: np.ndarray,
    fvc_scaler: StandardScaler,
    gt_df: pd.DataFrame,
    threshold: float,
    set_name: str = "test"
) -> Dict:
    """
    Evaluate progression assessment on a dataset (validation or test)
    
    Args:
        patient_ids: List of patient IDs
        baseline_fvc: Baseline FVC (denormalized)
        predicted_fvc_52_norm: Predicted week 52 FVC (normalized)
        true_fvc_52_norm: True week 52 FVC (normalized)
        fvc_scaler: StandardScaler used for FVC normalization
        gt_df: Ground truth dataframe with progression labels
        threshold: Progression threshold (%)
        set_name: Name of the set ("validation" or "test")
    
    Returns:
        Dictionary with progression metrics and predictions
    """
    # Denormalize predictions and targets
    dummy_baseline = np.zeros((len(true_fvc_52_norm), 1))
    
    targets_with_dummy = np.column_stack([dummy_baseline, true_fvc_52_norm])
    true_fvc_52_denorm = fvc_scaler.inverse_transform(targets_with_dummy)[:, 1]
    
    preds_with_dummy = np.column_stack([dummy_baseline, predicted_fvc_52_norm])
    predicted_fvc_52_denorm = fvc_scaler.inverse_transform(preds_with_dummy)[:, 1]
    
    # Compute progression
    progression_results = compute_progression_from_fvc_predictions(
        baseline_fvc=baseline_fvc,
        predicted_fvc_52=predicted_fvc_52_denorm,
        true_fvc_52=true_fvc_52_denorm,
        threshold=threshold
    )
    
    # Get ground truth progression from dataframe
    gt_progression = []
    for pid in patient_ids:
        prog_val = gt_df[gt_df['PatientID'] == pid]['has_progressed'].values[0]
        gt_progression.append(prog_val)
    gt_progression = np.array(gt_progression)
    
    # Calculate metrics using ground truth progression labels
    predicted_prog = progression_results['predicted_progression']
    
    acc = accuracy_score(gt_progression, predicted_prog)
    prec = precision_score(gt_progression, predicted_prog, zero_division=0)
    rec = recall_score(gt_progression, predicted_prog, zero_division=0)
    f1 = f1_score(gt_progression, predicted_prog, zero_division=0)
    
    cm = confusion_matrix(gt_progression, predicted_prog)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens = 0
        spec = 0
    
    # Try to compute AUC if we have both classes
    try:
        if len(np.unique(gt_progression)) > 1:
            auc = roc_auc_score(gt_progression, progression_results['percent_decline_pred'])
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    results = {
        f'{set_name}_threshold': threshold,
        f'{set_name}_progression_accuracy': acc,
        f'{set_name}_progression_precision': prec,
        f'{set_name}_progression_recall': rec,
        f'{set_name}_progression_sensitivity': sens,
        f'{set_name}_progression_specificity': spec,
        f'{set_name}_progression_f1': f1,
        f'{set_name}_progression_auc': auc,
        f'{set_name}_confusion_matrix': cm,
        f'{set_name}_predicted_progression': predicted_prog,
        f'{set_name}_gt_progression': gt_progression,
        f'{set_name}_percent_decline_pred': progression_results['percent_decline_pred'],
        f'{set_name}_percent_decline_true': progression_results['percent_decline_true'],
        f'{set_name}_n_predicted_prog': predicted_prog.sum(),
        f'{set_name}_n_true_prog': gt_progression.sum()
    }
    
    return results


def plot_progression_comparison(
    val_results: Dict,
    test_results: Dict,
    save_path: Path
):
    """
    Plot progression assessment comparison between validation and test sets
    
    Args:
        val_results: Validation progression results
        test_results: Test progression results  
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Extract metrics
    val_threshold = val_results['validation_threshold']
    test_threshold = test_results['test_threshold']
    
    # 1. Validation Confusion Matrix
    ax = axes[0, 0]
    cm_val = val_results['validation_confusion_matrix']
    im = ax.imshow(cm_val, cmap='Blues', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Prog', 'Prog'])
    ax.set_yticklabels(['No Prog', 'Prog'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Validation Set (Threshold: {val_threshold}%)\n' + 
                 f'F1={val_results["validation_progression_f1"]:.3f}', fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{int(cm_val[i, j])}',
                          ha='center', va='center', fontsize=12,
                          color='white' if cm_val[i, j] > cm_val.max()/2 else 'black')
    
    # 2. Test Confusion Matrix
    ax = axes[0, 1]
    cm_test = test_results['test_confusion_matrix']
    im = ax.imshow(cm_test, cmap='Greens', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Prog', 'Prog'])
    ax.set_yticklabels(['No Prog', 'Prog'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Test Set (Threshold: {test_threshold}%)\n' + 
                 f'F1={test_results["test_progression_f1"]:.3f}', fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{int(cm_test[i, j])}',
                          ha='center', va='center', fontsize=12,
                          color='white' if cm_test[i, j] > cm_test.max()/2 else 'black')
    
    # 3. Metrics Comparison
    ax = axes[0, 2]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    val_vals = [val_results['validation_progression_accuracy'], 
                val_results['validation_progression_precision'],
                val_results['validation_progression_recall'],
                val_results['validation_progression_f1']]
    test_vals = [test_results['test_progression_accuracy'],
                 test_results['test_progression_precision'],
                 test_results['test_progression_recall'],
                 test_results['test_progression_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, val_vals, width, label='Validation', alpha=0.7, color='#3498db')
    ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.7, color='#2ecc71')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Validation Decline Distribution
    ax = axes[1, 0]
    val_decline_pred = val_results['validation_percent_decline_pred']
    val_gt_prog = val_results['validation_gt_progression']
    ax.hist(val_decline_pred[val_gt_prog == 0], bins=20, alpha=0.5, label='No Progression', color='blue')
    ax.hist(val_decline_pred[val_gt_prog == 1], bins=20, alpha=0.5, label='Progression', color='red')
    ax.axvline(x=val_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {val_threshold}%')
    ax.set_xlabel('Predicted FVC Decline (%)')
    ax.set_ylabel('Count')
    ax.set_title('Validation: Predicted Decline Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Test Decline Distribution
    ax = axes[1, 1]
    test_decline_pred = test_results['test_percent_decline_pred']
    test_gt_prog = test_results['test_gt_progression']
    ax.hist(test_decline_pred[test_gt_prog == 0], bins=20, alpha=0.5, label='No Progression', color='blue')
    ax.hist(test_decline_pred[test_gt_prog == 1], bins=20, alpha=0.5, label='Progression', color='red')
    ax.axvline(x=test_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {test_threshold}%')
    ax.set_xlabel('Predicted FVC Decline (%)')
    ax.set_ylabel('Count')
    ax.set_title('Test: Predicted Decline Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Validation', 'Test'],
        ['Threshold (%)', f'{val_threshold:.1f}', f'{test_threshold:.1f}'],
        ['Accuracy', f'{val_vals[0]:.3f}', f'{test_vals[0]:.3f}'],
        ['Precision', f'{val_vals[1]:.3f}', f'{test_vals[1]:.3f}'],
        ['Recall', f'{val_vals[2]:.3f}', f'{test_vals[2]:.3f}'],
        ['F1-Score', f'{val_vals[3]:.3f}', f'{test_vals[3]:.3f}'],
        ['Sensitivity', f'{val_results["validation_progression_sensitivity"]:.3f}', 
         f'{test_results["test_progression_sensitivity"]:.3f}'],
        ['Specificity', f'{val_results["validation_progression_specificity"]:.3f}',
         f'{test_results["test_progression_specificity"]:.3f}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved progression comparison plot: {save_path}")
