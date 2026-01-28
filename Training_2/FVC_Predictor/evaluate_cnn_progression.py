
# This script evaluates progression prediction from CNN FVC@52 predictions only.
def save_combined_fvc52_and_progression_summary(fvc52_summary_path, progression_summary_path, output_path):
    """
    Merge FVC@52 and progression prediction summaries and save as a combined CSV.
    Matches on feature type/approach, robust to naming differences.
    """
    fvc52_df = pd.read_csv(fvc52_summary_path)
    prog_df = pd.read_csv(progression_summary_path)

    # Normalize keys for matching
    def norm_key(x):
        x = str(x).lower()
        x = x.replace('features', '').replace('full', 'full').replace('+', '').replace('handcrafted', 'handcrafted').replace('demographics', 'demographics')
        x = x.replace('_', '').replace(' ', '')
        return x.strip()

    fvc52_df['key'] = fvc52_df['FeatureType'].apply(norm_key)
    prog_df['key'] = prog_df['Approach'].apply(norm_key)

    merged = pd.merge(fvc52_df, prog_df, on='key', suffixes=('_fvc52', '_progression'))
    merged = merged.drop(columns=['key'])
    merged.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined FVC@52 and progression summary: {output_path}")
import os
import random

def plot_patient_trajectory(patient_id, patient_data, pred_row, output_dir, approach_name, progression_threshold=10.0):
    """
    Plot FVC trajectory for a single patient: true FVC, predicted trajectory, 10% decline threshold, and status.
    """
    baseline_fvc = pred_row['baseline_fvc']
    fvc52_pred = pred_row['fvc52_predicted']
    slope = (fvc52_pred - baseline_fvc) / 52.0
    status = 'PROGRESSION' if pred_row['predicted_progression'] else 'STABLE'
    
    # True FVC
    weeks = np.array(patient_data[patient_id]['weeks'])
    fvc_values = np.array(patient_data[patient_id]['fvc_values'])
    
    # Predicted line
    pred_weeks = np.linspace(0, max(weeks.max(), 52), 100)
    pred_fvc = baseline_fvc + slope * pred_weeks
    
    # 10% decline threshold
    decline_threshold = baseline_fvc * (1 - progression_threshold / 100)
    
    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(weeks, fvc_values, 'o-', color='black', label='True FVC', alpha=0.7)
    plt.plot(pred_weeks, pred_fvc, '-', color='red', label='Predicted Trajectory', alpha=0.8)
    plt.axhline(decline_threshold, color='red', linestyle=':', label=f'{progression_threshold:.1f}% decline threshold')
    
    # Highlight baseline and week 52
    plt.scatter([0], [baseline_fvc], color='green', s=80, zorder=5, label='Baseline')
    plt.scatter([52], [fvc52_pred], color='red', s=80, marker='x', zorder=5, label='Predicted FVC@52')
    
    # If we have true FVC@52
    if 52 in weeks:
        true_fvc52_idx = np.where(weeks == 52)[0][0]
        plt.scatter([52], [fvc_values[true_fvc52_idx]], color='blue', s=80, zorder=5, label='True FVC@52')
    
    # Title
    plt.title(f"Patient {patient_id}\nSlope: {slope:.2f} ml/week | Status: {status}", 
              fontsize=12, fontweight='bold')
    plt.xlabel('Weeks from Baseline')
    plt.ylabel('FVC (ml)')
    plt.legend()
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{approach_name}_patient_{patient_id}.png', dpi=200)
    plt.close()

"""
Evaluate Progression Prediction from CNN FVC@52 Predictions
==========================================================

Uses the FVC@52 predictions from the CNN-only approach to classify progression
(≥10% FVC decline from baseline) and evaluates against ground truth.
Computes metrics per fold and averages across folds.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'predictions_dir': Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training_2\FVC_Predictor\Cyclic_kfold\predictions_cyclic_kfold\direct'),
    'results_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/FVC_Predictor/Cyclic_kfold/progression_results/direct_intercept'),
    'plots_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/FVC_Predictor/Cyclic_kfold/progression_plots/direct_intercept'),
    'csv_path': 'Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'csv_features_path': 'Training/CNN_Slope_Prediction/patient_features.csv',
    'npy_dir': 'Dataset/extracted_npy/extracted_npy',
    'n_folds': 5,
    'progression_threshold': 10,  # 10% decline
    'label_csv': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv'
}


# ...existing code from evaluate_progression.py for classify_progression, get_ground_truth_progression, classify_progression_from_fvc52, compute_progression_metrics, plot_progression_comparison...


def get_ground_truth_progression(patient_data, threshold_percent=10.0):
    """
    Get ground truth progression labels from actual FVC measurements
    Uses same logic as prediction_fold.py
    
    Args:
        patient_data: Dict with patient FVC measurements
        threshold_percent: Decline threshold for progression
    
    Returns:
        Dict {patient_id: bool} indicating progression
    """

    #get ground truth progression labels from D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data\patient_progression_52w.csv
    labels = {}
    gt_df = pd.read_csv(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_risk_2\data_intercept\patient_progression_52w_intercept.csv')
    for _, row in gt_df.iterrows():
        patient_id = row['Patient']
        gt_progression = row['event_52']  # 1 if progression, 0 if stable
        labels[patient_id] = bool(gt_progression) # True = progression, False = stable

    return labels

def classify_progression_from_fvc52(predictions_df, threshold_percent=10.0):
    """
    Classify progression based on predicted FVC@52
    
    Args:
        predictions_df: DataFrame with columns ['patient_id', 'baseline_fvc', 'fvc52_predicted']
        threshold_percent: Decline threshold for progression
    
    Returns:
        DataFrame with added columns:
            'predicted_progression': bool
            'decline_percent': float
            'probability': float (simple probability estimate)
    """
    results = []

    train_df = pd.read_csv(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv')
    baseline_fvc_dict = {}
    #Get intercept
    for patient_id in train_df['Patient'].unique():
        #Extract just one value of intercept0 since they are all the same for each patient
        baseline_fvc = train_df[train_df['Patient'] == patient_id]['fvc_intercept0'].values[0]
        if baseline_fvc is not None :
            baseline_fvc_dict[patient_id] = baseline_fvc

    for _, row in predictions_df.iterrows():
        patient_id = row['patient_id']

        if patient_id not in baseline_fvc_dict:
            continue

        baseline_fvc = baseline_fvc_dict[patient_id]
        fvc52_pred = row['predicted_fvc52']

        decline_pred_abs = fvc52_pred - baseline_fvc
        decline_pred_pct = 100 * (decline_pred_abs / baseline_fvc)

        predicted_progression = decline_pred_pct >= threshold_percent


        k = 0.3  # controls softness of transition
        probability = 1 / (1 + np.exp(-(decline_pred_pct - threshold_percent) / k))


        results.append({
            'patient_id': patient_id,
            'fold': row.get('fold', None),
            'baseline_fvc': baseline_fvc,
            'fvc52_predicted': fvc52_pred,
            'true_fvc52': row.get('true_fvc52', None),
            'decline_pred_pct': decline_pred_pct,
            'predicted_progression': int(predicted_progression),
            'probability': probability
        })
    
    return pd.DataFrame(results)

# =============================================================================
# FIX 1: Compute confusion matrix properly in evaluate_approach_progression
# =============================================================================

def evaluate_approach_progression(config, ground_truth_labels):
    """Evaluate progression prediction for one approach across all folds"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating progression prediction")
    print(f"{'='*80}")
    
    fold_metrics_list = []
    all_predictions_for_save = []
    
    
    
    pkl_path = os.path.join(CONFIG['predictions_dir'], 'all_predictions.pkl')

    with open(pkl_path, 'rb') as f:
        all_folds_predictions = pickle.load(f)

    
    print(f"✓ Loaded predictions from {pkl_path}")

    print(all_folds_predictions.keys())
    """

    # Load train.csv to get baseline FVC
    train_csv_path = 'Training/CNN_Slope_Prediction/train.csv'
    train_df = pd.read_csv(train_csv_path)
    
    # Get baseline FVC (earliest measurement) for each patient
    baseline_fvc = {}
    for patient_id in patient_data.keys():
        patient_weeks = train_df[train_df['Patient'] == patient_id]['Weeks'].values
        if len(patient_weeks) > 0:
            earliest_week_idx = np.argmin(patient_weeks)
            baseline_fvc_val = train_df[train_df['Patient'] == patient_id].iloc[earliest_week_idx]['FVC']
            baseline_fvc[patient_id] = baseline_fvc_val
        else:
            baseline_fvc[patient_id] = 0.0  # Fallback
    
    print(f"✓ Loaded baseline FVC for {len(baseline_fvc)} patients")
    print(f"  Baseline FVC range: {np.min(list(baseline_fvc.values())):.0f} - {np.max(list(baseline_fvc.values())):.0f} mL")"""
    
    # Evaluate each fold separately
    for fold_idx in range(config['n_folds']):
        fold_key = f'fold_{fold_idx}'
        
        if fold_key not in all_folds_predictions:
            print(f"⚠️  Fold {fold_idx}: No predictions found")
            continue
        
        fold_preds = all_folds_predictions[fold_key]
        
        # Get test predictions for this fold
        test_preds_dict = fold_preds['test'] # {patient_id: predicted_fvc52}
        
        if len(test_preds_dict) == 0:
            print(f"⚠️  Fold {fold_idx}: No test predictions")
            continue
        
        # Convert to DataFrame
        fold_df = pd.DataFrame([
            {'patient_id': pid, 'predicted_fvc52': pred}
            for pid, pred in test_preds_dict.items()
        ])
        
        # Add ground truth FVC@52 if available
        label_df = pd.read_csv(config['label_csv'])
        fold_df = fold_df.merge(
            label_df[['Patient', 'fvc_52']], 
            left_on='patient_id', 
            right_on='Patient', 
            how='left'
        )

        fold_df = fold_df.rename(columns={'fvc_52': 'true_fvc52'})
        fold_df = fold_df.drop(columns=['Patient'])
        # Remove patients without valid FVC@52
        fold_df = fold_df[fold_df['true_fvc52'].notna()].copy()
        
        if len(fold_df) == 0:
            print(f"⚠️  Fold {fold_idx}: No patients with ground truth FVC@52")
            continue
        
        fold_df['fold'] = fold_idx
        fold_df['has_true_fvc52'] = True
        
        print(f"\n  Fold {fold_idx}: {len(fold_df)} patients with ground truth")
        
        # Classify progression for this fold
        progression_df = classify_progression_from_fvc52(
            fold_df, 
            threshold_percent=config['progression_threshold']
        )
        
        # Add ground truth labels
        progression_df['true_progression_gt'] = progression_df['patient_id'].map(ground_truth_labels)
        
        # Filter to patients with ground truth
        progression_df = progression_df[progression_df['true_progression_gt'].notna()].copy()
        
        if len(progression_df) == 0:
            print(f"    ⚠️  No patients with ground truth progression labels")
            continue
        
        all_predictions_for_save.append(progression_df)
        
        # Compute metrics for this fold
        fold_metrics = compute_progression_metrics(
            progression_df['true_progression_gt'].values,
            progression_df['predicted_progression'].values,
            progression_df['probability'].values
        )
        
        fold_metrics_list.append(fold_metrics)
        
        print(f"    Patients: {fold_metrics['n_samples']}")
        print(f"    True prog: {fold_metrics['n_positive']}, Pred prog: {int(progression_df['predicted_progression'].sum())}")
        print(f"    Acc={fold_metrics['accuracy']:.3f}, AUC={fold_metrics['auc_roc']:.3f}")
    
    if len(fold_metrics_list) == 0:
        print(f"❌ No valid folds found")
        return None
    
    # Display per-fold metrics table
    print(f"\n{'='*80}")
    print(f"PER-FOLD METRICS (Test Set Evaluation)")
    print(f"{'='*80}")
    
    fold_table_data = []
    for fold_idx, fold_metrics in enumerate(fold_metrics_list):
        fold_table_data.append({
            'Fold': fold_idx,
            'N': fold_metrics['n_samples'],
            'Prog': fold_metrics['n_positive'],
            'Acc': f"{fold_metrics['accuracy']:.3f}",
            'Prec': f"{fold_metrics['precision']:.3f}",
            'Rec': f"{fold_metrics['recall']:.3f}",
            'F1': f"{fold_metrics['f1_score']:.3f}",
            'Spec': f"{fold_metrics['specificity']:.3f}",
            'AUC': f"{fold_metrics['auc_roc']:.3f}",
            'AP': f"{fold_metrics['average_precision']:.3f}"
        })
    
    fold_df_table = pd.DataFrame(fold_table_data)
    print("\n" + fold_df_table.to_string(index=False))
    
    # Compute mean and std across folds
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc', 'average_precision']
    
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in metrics_names:
        values = [m[metric_name] for m in fold_metrics_list]
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)

    # Combine all predictions for aggregate metrics
    combined_df = pd.concat(all_predictions_for_save, ignore_index=True)
    
    # Compute AGGREGATE confusion matrix from combined predictions
    y_true_combined = combined_df['true_progression_gt'].values.astype(int)
    y_pred_combined = combined_df['predicted_progression'].values.astype(int)
    y_prob_combined = combined_df['probability'].values
    
    cm_aggregate = confusion_matrix(y_true_combined, y_pred_combined)
    
    # Ensure it's 2x2 (handle edge case where one class might be missing)
    if cm_aggregate.shape != (2, 2):
        cm_full = np.zeros((2, 2), dtype=int)
        for i in range(min(cm_aggregate.shape[0], 2)):
            for j in range(min(cm_aggregate.shape[1], 2)):
                cm_full[i, j] = cm_aggregate[i, j]
        cm_aggregate = cm_full
    
    mean_metrics['confusion_matrix'] = cm_aggregate
    
    # Compute ROC and PR curves from combined data
    if len(np.unique(y_true_combined)) > 1 and len(np.unique(y_prob_combined)) > 1:
        fpr, tpr, _ = roc_curve(y_true_combined, y_prob_combined)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_combined, y_prob_combined)
        mean_metrics['roc_curve'] = (fpr, tpr)
        mean_metrics['pr_curve'] = (precision_curve, recall_curve)
    else:
        mean_metrics['roc_curve'] = (np.array([0, 1]), np.array([0, 1]))
        mean_metrics['pr_curve'] = (np.array([0, 1]), np.array([0, 1]))
    
    # Total counts
    mean_metrics['n_samples'] = int(combined_df['true_progression_gt'].notna().sum())
    mean_metrics['n_positive'] = int(combined_df['true_progression_gt'].sum())
    mean_metrics['n_negative'] = mean_metrics['n_samples'] - mean_metrics['n_positive']
    
    # Display aggregated metrics
    print(f"\n{'='*60}")
    print(f"AGGREGATE METRICS (Mean ± Std across {len(fold_metrics_list)} folds)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {mean_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}")
    print(f"  Precision:   {mean_metrics['precision']:.3f} ± {std_metrics['precision']:.3f}")
    print(f"  Recall:      {mean_metrics['recall']:.3f} ± {std_metrics['recall']:.3f}")
    print(f"  F1-Score:    {mean_metrics['f1_score']:.3f} ± {std_metrics['f1_score']:.3f}")
    print(f"  Specificity: {mean_metrics['specificity']:.3f} ± {std_metrics['specificity']:.3f}")
    print(f"  AUC-ROC:     {mean_metrics['auc_roc']:.3f} ± {std_metrics['auc_roc']:.3f}")
    print(f"  AP:          {mean_metrics['average_precision']:.3f} ± {std_metrics['average_precision']:.3f}")
    print(f"\n  Total patients: {mean_metrics['n_samples']} (Prog: {mean_metrics['n_positive']}, Stable: {mean_metrics['n_negative']})")
    
    # Display aggregate confusion matrix
    print(f"\n{'='*40}")
    print("AGGREGATE CONFUSION MATRIX (All Folds)")
    print(f"{'='*40}")
    print(f"                  Predicted")
    print(f"                Stable  Prog")
    print(f"True  Stable  {cm_aggregate[0,0]:>6d}  {cm_aggregate[0,1]:>5d}")
    print(f"      Prog    {cm_aggregate[1,0]:>6d}  {cm_aggregate[1,1]:>5d}")
    print(f"{'='*40}")
    
    return {
        'approach': 'FVC@52 CNN',
        'metrics': mean_metrics,
        'std_metrics': std_metrics,
        'predictions': combined_df,
        'fold_metrics': fold_metrics_list
    }



# =============================================================================
# FIX 2: Correct confusion matrix plotting
# =============================================================================

def plot_progression_comparison(results, config):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    metrics = results['metrics']
    
    # 1. Metrics Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['specificity'],
        metrics['auc_roc']
    ]
    
    bars = ax1.bar(metrics_names, values, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Progression Prediction Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[1, 0])
    fpr, tpr = metrics['roc_curve']
    auc = metrics['auc_roc']
    ax2.plot(fpr, tpr, color='#3498db', lw=2.5, alpha=0.8, label=f'AUC={auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[1, 1])
    precision, recall = metrics['pr_curve']
    ap = metrics['average_precision']
    ax3.plot(recall, precision, color='#e74c3c', lw=2.5, alpha=0.8, label=f'AP={ap:.3f}')
    ax3.set_xlabel('Recall', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    plt.suptitle(f'IPF Progression Prediction Evaluation (≥{config["progression_threshold"]}% FVC Decline)',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    save_path = config['plots_dir'] / 'progression_prediction_evaluation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
    plt.close()
    
    # Separate confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               cbar=False, ax=ax,
               xticklabels=['Stable', 'Prog'],
               yticklabels=['Stable', 'Prog'],
               annot_kws={'fontsize': 16, 'fontweight': 'bold'},
               linewidths=2, linecolor='black')
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=11, color='gray')
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.set_title(f'Confusion Matrix\nAccuracy={acc:.3f}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    cm_path = config['plots_dir'] / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {cm_path}")
    plt.close()

def compute_progression_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive progression prediction metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases where one class might be missing
    if cm.shape == (1, 1):
        # Only one class present
        if y_true[0] == 0:  # Only negative class
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # Only positive class
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    elif cm.shape == (2, 1):
        tn, fp = cm[0, 0], 0
        fn, tp = cm[1, 0], 0
    elif cm.shape == (1, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    if len(np.unique(y_true)) > 1 and len(np.unique(y_probs)) > 1:
        auc_roc = roc_auc_score(y_true, y_probs)
    else:
        auc_roc = 0.5  # Random classifier
    
    # Average Precision
    if len(np.unique(y_true)) > 1:
        ap = average_precision_score(y_true, y_probs)
    else:
        ap = 0.0
    
    # ROC and PR curves
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        precision_curve, recall_curve = np.array([1, 0]), np.array([0, 1])
    
    # Rebuild 2x2 confusion matrix
    cm_full = np.array([[tn, fp], [fn, tp]])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'average_precision': ap,
        'confusion_matrix': cm_full,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision_curve, recall_curve),
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((~y_true.astype(bool)).sum())
    }



"""
def classify_progression_from_fvc52(predictions_df, threshold_percent=10.0):
    
    results = []
    print(predictions_df.head())
    for _, row in predictions_df.iterrows():
        baseline_fvc = row['baseline_fvc']
        fvc52_pred = row['fvc52_predicted']
        
        # Predicted decline from FVC@52 prediction
        decline_pred_abs = baseline_fvc - fvc52_pred
        decline_pred_pct = 100 * decline_pred_abs / baseline_fvc
        
        # Classification based on predicted FVC@52
        predicted_progression = decline_pred_pct >= threshold_percent
        
        # Probability (normalized decline percentage to [0, 1])
        probability = max(0.0, min(decline_pred_pct / threshold_percent, 1.0))
        
        results.append({
            'patient_id': row['patient_id'],
            'fold': row.get('fold', None),
            'baseline_fvc': baseline_fvc,
            'fvc52_predicted': fvc52_pred,
            'decline_pred_pct': decline_pred_pct,
            'predicted_progression': int(predicted_progression),
            'probability': probability
        })
    
    return pd.DataFrame(results)
"""

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("CNN PROGRESSION PREDICTION EVALUATION")
    print("="*80)
    print(f"\nProgression Threshold: ≥{CONFIG['progression_threshold']}% FVC decline from baseline")
    print(f"Evaluation: {CONFIG['n_folds']}-Fold Cross-Validation")

    # Load patient data to create ground truth labels
    print("\n" + "="*80)
    print("LOADING PATIENT DATA FOR GROUND TRUTH")
    print("="*80)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utilities import IPFDataLoader

    dl = IPFDataLoader(CONFIG['csv_path'], CONFIG['csv_features_path'], CONFIG['npy_dir'])
    patient_data, _ = dl.get_patient_data()

    print(f"✓ Loaded patient_data for {len(patient_data)} patients")

    # Create ground truth labels
    ground_truth_labels = get_ground_truth_progression(patient_data, CONFIG['progression_threshold'])

    print(f"✓ Created ground truth labels for {len(ground_truth_labels)} patients")
    print(f"  Progression cases: {sum(ground_truth_labels.values())}")
    print(f"  Stable cases: {len(ground_truth_labels) - sum(ground_truth_labels.values())}")

    # Evaluate
    results = evaluate_approach_progression(CONFIG, ground_truth_labels)
    
    if results is None:
        print("❌ No results to process.")
        return

    # Create patient trajectory plots
    print("\n" + "="*80)
    print("CREATING PATIENT TRAJECTORY PLOTS")
    print("="*80)
    
    plot_dir = CONFIG['plots_dir'] 
    plot_dir.mkdir(parents=True, exist_ok=True)

    prog_dir = CONFIG['results_dir']
    prog_dir.mkdir(parents=True, exist_ok=True)

    
    preds_df = results['predictions']
    # Only plot for patients with ground truth and valid FVC@52
    valid_patients = preds_df[
        (~preds_df['fvc52_predicted'].isna()) & 
        (preds_df['true_progression_gt'].notna())
    ]['patient_id'].unique()
    
    sample_patients = random.sample(list(valid_patients), min(5, len(valid_patients)))
    
    for pid in sample_patients:
        if pid not in patient_data:
            continue
        row = preds_df[preds_df['patient_id'] == pid].iloc[0]
        plot_patient_trajectory(
            pid, patient_data, row, plot_dir, 
            'FVC52_CNN', 
            progression_threshold=CONFIG['progression_threshold']
        )
    
    print(f"✓ Saved {len(sample_patients)} patient trajectory plots to {plot_dir}")

    # Create summary
    print("\n" + "="*80)
    print("SUMMARY: PROGRESSION PREDICTION PERFORMANCE")
    print("="*80)

    metrics = results['metrics']
    std_metrics = results['std_metrics']

    summary_data = [{
        'Approach': results['approach'],
        'Accuracy': f"{metrics['accuracy']:.3f}±{std_metrics['accuracy']:.3f}",
        'Precision': f"{metrics['precision']:.3f}±{std_metrics['precision']:.3f}",
        'Recall': f"{metrics['recall']:.3f}±{std_metrics['recall']:.3f}",
        'F1-Score': f"{metrics['f1_score']:.3f}±{std_metrics['f1_score']:.3f}",
        'Specificity': f"{metrics['specificity']:.3f}±{std_metrics['specificity']:.3f}",
        'AUC-ROC': f"{metrics['auc_roc']:.3f}±{std_metrics['auc_roc']:.3f}",
        'AP': f"{metrics['average_precision']:.3f}±{std_metrics['average_precision']:.3f}",
        'N': metrics['n_samples'],
        'Prog': metrics['n_positive'],
        'Stable': metrics['n_negative']
    }]

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_path = CONFIG['results_dir'] / 'progression_prediction_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")

    # Save detailed predictions
    pred_path = CONFIG['results_dir'] / 'progression_predictions.csv'
    results['predictions'].to_csv(pred_path, index=False)
    print(f"✓ Saved predictions: {pred_path}")

    # Visualize
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    plot_progression_comparison(results, CONFIG)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
