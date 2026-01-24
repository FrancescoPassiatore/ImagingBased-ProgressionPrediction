
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
import random

def plot_patient_trajectory(patient_id, patient_data, pred_row, output_dir, approach_name, progression_threshold=10.0):
    """
    Plot FVC trajectory for a single patient: true FVC, predicted trajectory, 10% decline threshold, and status.
    """
    import matplotlib.pyplot as plt
    baseline_fvc = pred_row['baseline_fvc']
    fvc52_pred = pred_row['fvc52_predicted']
    slope = (fvc52_pred - baseline_fvc) / 52.0
    status = 'PROGRESSION' if pred_row['predicted_progression'] else 'STABLE'
    # True FVC
    weeks = np.array(patient_data[patient_id]['weeks'])
    fvc_values = np.array(patient_data[patient_id]['fvc_values'])
    # Predicted line
    pred_weeks = np.linspace(0, weeks.max(), 100)
    pred_fvc = baseline_fvc + slope * pred_weeks
    # 10% decline threshold
    decline_threshold = baseline_fvc * (1 - progression_threshold / 100)
    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(weeks, fvc_values, 'o-', color='black', label='True FVC', alpha=0.7)
    plt.plot(pred_weeks, pred_fvc, '-', color='red', label='Predicted Trajectory', alpha=0.8)
    plt.axhline(decline_threshold, color='red', linestyle=':', label=f'{progression_threshold:.1f}% decline threshold')
    # Highlight baseline and week 52
    plt.scatter([0], [baseline_fvc], color='green', s=80, zorder=5)
    if 52 in weeks:
        plt.scatter([52], [fvc_values[np.where(weeks==52)[0][0]]], color='green', s=80, zorder=5)
    plt.scatter([52], [fvc52_pred], color='red', s=80, marker='x', zorder=5)
    # Title
    plt.title(f"Patient {patient_id}\nSlope: {slope:.2f} ml/week | Status: {status}", fontsize=12, fontweight='bold')
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
    'results_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/CNN_Training/Cyclic_kfold/fvc52_results/mse_norm_attention'),
    'plots_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_2/CNN_Training/Cyclic_kfold/progression_plots/mse_norm_attention'),
    'csv_path': 'Training/CNN_Slope_Prediction/train_with_coefs.csv',
    'csv_features_path': 'Training/CNN_Slope_Prediction/patient_features.csv',
    'npy_dir': 'Dataset/extracted_npy/extracted_npy',
    'n_folds': 5,
    'progression_threshold': 10  # 10% decline
}

APPROACHES = {
    'cnn_only': 'CNN-Only',
}

# ...existing code from evaluate_progression.py for classify_progression, get_ground_truth_progression, classify_progression_from_fvc52, compute_progression_metrics, plot_progression_comparison...
def classify_progression(baseline_fvc, future_fvc, threshold_percent=10.0):
    """
    Classify progression based on FVC decline
    
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
    labels = {}
    
    for pid, pdata in patient_data.items():
        fvc = np.asarray(pdata['fvc_values'])
        if len(fvc) < 2:
            continue
        baseline = fvc[0]
        future = fvc[1:]
        _, decline_pct = classify_progression(baseline, future, threshold_percent)
        labels[pid] = int(np.any(decline_pct >= threshold_percent))
    
    return labels

"""
Fixed confusion matrix computation and plotting
"""

# =============================================================================
# FIX 1: Compute confusion matrix properly in evaluate_approach_progression
# =============================================================================

def evaluate_approach_progression(approach_key, approach_name, config, ground_truth_labels):
    """Evaluate progression prediction for one approach across all folds"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {approach_name}")
    print(f"{'='*80}")
    
    fold_metrics_list = []
    all_predictions_for_save = []
    
    # Evaluate each fold separately
    for fold_idx in range(config['n_folds']):
        pred_file = config['results_dir'] / f'{approach_key}_fold{fold_idx}_fvc52_predictions.csv'
        
        if not pred_file.exists():
            print(f"⚠️  Fold {fold_idx}: Missing {pred_file}")
            continue
        
        fold_preds = pd.read_csv(pred_file)
        
        # Remove patients without valid FVC@52
        fold_preds = fold_preds[fold_preds['has_true_fvc52'] != False].copy()
        fold_preds = fold_preds[fold_preds['has_true_fvc52'] != 'False'].copy()
        fold_preds['fold'] = fold_idx
        
        print(f"\n  Fold {fold_idx}: {len(fold_preds)} patients")
        
        # Classify progression for this fold
        progression_df = classify_progression_from_fvc52(
            fold_preds, 
            threshold_percent=config['progression_threshold']
        )
        
        # Add ground truth labels
        progression_df['true_progression_gt'] = progression_df['patient_id'].map(ground_truth_labels)
        
        # Filter to patients with ground truth
        progression_df = progression_df[progression_df['true_progression_gt'].notna()].copy()
        
        if len(progression_df) == 0:
            print(f"    ⚠️  No patients with ground truth")
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
        print(f"❌ No valid folds found for {approach_name}")
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
    
    fold_df = pd.DataFrame(fold_table_data)
    print("\n" + fold_df.to_string(index=False))
    
    # Compute mean and std across folds
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc', 'average_precision']
    
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in metrics_names:
        values = [m[metric_name] for m in fold_metrics_list]
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)

    # ✅ FIX: Combine all predictions for aggregate metrics
    combined_df = pd.concat(all_predictions_for_save, ignore_index=True)
    
    # ✅ FIX: Compute AGGREGATE confusion matrix from combined predictions
    # This gives the overall confusion matrix across all test folds
    y_true_combined = combined_df['true_progression_gt'].values.astype(int)
    y_pred_combined = combined_df['predicted_progression'].values.astype(int)
    y_prob_combined = combined_df['probability'].values
    
    cm_aggregate = confusion_matrix(y_true_combined, y_pred_combined)
    
    # Ensure it's 2x2 (handle edge case where one class might be missing)
    if cm_aggregate.shape != (2, 2):
        cm_full = np.zeros((2, 2), dtype=int)
        for i in range(cm_aggregate.shape[0]):
            for j in range(cm_aggregate.shape[1]):
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
    
    # ✅ Display aggregate confusion matrix
    print(f"\n{'='*40}")
    print("AGGREGATE CONFUSION MATRIX (All Folds)")
    print(f"{'='*40}")
    print(f"                  Predicted")
    print(f"                Stable  Prog")
    print(f"True  Stable  {cm_aggregate[0,0]:>6d}  {cm_aggregate[0,1]:>5d}")
    print(f"      Prog    {cm_aggregate[1,0]:>6d}  {cm_aggregate[1,1]:>5d}")
    print(f"{'='*40}")
    
    return {
        'approach': approach_name,
        'metrics': mean_metrics,
        'std_metrics': std_metrics,
        'predictions': combined_df,
        'fold_metrics': fold_metrics_list
    }


# =============================================================================
# FIX 2: Correct confusion matrix plotting
# =============================================================================

def plot_progression_comparison(all_results, config):
    """Create comprehensive visualization comparing all approaches"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    approaches = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    # 1. Metrics Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC']
    x = np.arange(len(metrics_names))
    width = 0.2
    
    for i, (approach_key, results) in enumerate(all_results.items()):
        metrics = results['metrics']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['specificity'],
            metrics['auc_roc']
        ]
        offset = width * (i - len(approaches)/2 + 0.5)
        bars = ax1.bar(x + offset, values, width, label=results['approach'], 
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Progression Prediction Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. ROC Curves
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (approach_key, results) in enumerate(all_results.items()):
        fpr, tpr = results['metrics']['roc_curve']
        auc = results['metrics']['auc_roc']
        ax2.plot(fpr, tpr, color=colors[i], lw=2.5, alpha=0.8,
                label=f"{results['approach']} (AUC={auc:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (approach_key, results) in enumerate(all_results.items()):
        precision, recall = results['metrics']['pr_curve']
        ap = results['metrics']['average_precision']
        ax3.plot(recall, precision, color=colors[i], lw=2.5, alpha=0.8,
                label=f"{results['approach']} (AP={ap:.3f})")
    
    ax3.set_xlabel('Recall', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # ✅ FIX: Confusion Matrices with proper heatmap
    for i, (approach_key, results) in enumerate(all_results.items()):
        # Only plot first 3 approaches (we have 3 subplot positions)
        if i >= 3:
            continue
            
        ax = fig.add_subplot(gs[2, i])
        cm = results['metrics'].get('confusion_matrix', None)
        
        if cm is None or cm.shape != (2, 2):
            ax.text(0.5, 0.5, f"{results['approach']}\nNo valid data", 
                   ha='center', va='center', fontsize=11)
            ax.axis("off")
            continue
        
        # ✅ Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['Stable', 'Prog'],
                   yticklabels=['Stable', 'Prog'],
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                   linewidths=2, linecolor='black')
        
        # ✅ Add percentage annotations
        total = cm.sum()
        for j in range(2):
            for k in range(2):
                pct = 100 * cm[j, k] / total
                ax.text(k + 0.5, j + 0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
        
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
        
        # ✅ Compute accuracy from confusion matrix
        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.set_title(f'{results["approach"]}\nAcc={acc:.3f}',
                    fontsize=11, fontweight='bold')
    
    plt.suptitle(f'IPF Progression Prediction Evaluation (≥{config["progression_threshold"]}% FVC Decline)',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    config['plots_dir'].mkdir(parents=True, exist_ok=True)
    save_path = config['plots_dir'] / 'progression_prediction_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
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
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    if len(np.unique(y_true)) > 1 and len(np.unique(y_probs)) > 1:
        auc_roc = roc_auc_score(y_true, y_probs)
    else:
        auc_roc = np.nan

    ap = average_precision_score(y_true, y_probs)
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'average_precision': ap,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision_curve, recall_curve),
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((~y_true.astype(bool)).sum())
    }



def classify_progression_from_fvc52(predictions_df, threshold_percent=10.0):
    """
    Classify progression from FVC@52 predictions
    
    Args:
        predictions_df: DataFrame with columns ['patient_id', 'fvc52_predicted', 'baseline_fvc']
                       (already filtered to patients with valid FVC@52)
        threshold_percent: Decline threshold
    
    Returns:
        DataFrame with progression predictions and probabilities
    """
    results = []
    
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

    # For patient-level plots
    plot_dir = CONFIG['plots_dir'] / 'patient_trajectories'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each approach
    all_results = {}

    for approach_key, approach_name in APPROACHES.items():
        results = evaluate_approach_progression(approach_key, approach_name, CONFIG, ground_truth_labels)
        if results is not None:
            all_results[approach_key] = results

        # --- Patient-level plots: random sample of 5 patients per approach ---
        preds_df = results['predictions']
        # Only plot for patients with ground truth and valid FVC@52
        valid_patients = preds_df[~preds_df['fvc52_predicted'].isna() & preds_df['true_progression_gt'].notna()]['patient_id'].unique()
        sample_patients = random.sample(list(valid_patients), min(5, len(valid_patients)))
        for pid in sample_patients:
            row = preds_df[preds_df['patient_id'] == pid].iloc[0]
            plot_patient_trajectory(pid, patient_data, row, plot_dir, approach_key, progression_threshold=CONFIG['progression_threshold'])
        print(f"✓ Saved patient trajectory plots for {approach_name} (sample of {len(sample_patients)}) to {plot_dir}")

    if len(all_results) == 0:
        print("\n❌ No results to evaluate!")
        return

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY: PROGRESSION PREDICTION PERFORMANCE")
    print("="*80)

    summary_data = []
    for approach_key, results in all_results.items():
        metrics = results['metrics']
        std_metrics = results['std_metrics']
        summary_data.append({
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
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_path = CONFIG['results_dir'] / 'cnn_progression_prediction_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")

    # Save detailed predictions for CNN approach
    for approach_key, results in all_results.items():
        pred_path = CONFIG['results_dir'] / f'{approach_key}_progression_predictions.csv'
        results['predictions'].to_csv(pred_path, index=False)
        print(f"✓ Saved {results['approach']} predictions: {pred_path}")

    # Visualize comparison
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    plot_progression_comparison(all_results, CONFIG)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
