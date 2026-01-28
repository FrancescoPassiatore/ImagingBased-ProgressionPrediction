"""
Evaluate Progression Prediction from Dual-Head CNN
==================================================

Evaluates progression prediction (≥10% FVC decline) from the dual-head model
which predicts baseline FVC(0), slope, and FVC(52) from CT scans only.
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
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'predictions_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_3/CNN_Training/Cyclic_kfold/predictions_dual_head'),
    'plots_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_3/CNN_Training/Cyclic_kfold/progression_evaluation'),
    'results_dir': Path(r'D:/FrancescoP/ImagingBased-ProgressionPrediction/Training_3/CNN_Training/Cyclic_kfold/final_results_dual_head'),
    'csv_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv',
    'csv_features_path': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv',
    'npy_dir': r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy',
    'n_folds': 5,
    'progression_threshold': 10.0  # 10% decline
}

# Create output directories
CONFIG['plots_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ground_truth_progression(patient_data, threshold_percent=10.0):
    """
    Get ground truth progression labels from actual FVC measurements
    
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
        
        # Check if any future measurement shows ≥threshold% decline
        decline_pct = 100 * (baseline - future) / baseline
        labels[pid] = int(np.any(decline_pct >= threshold_percent))
    
    return labels


def classify_progression_from_dual_head(predictions_dict, threshold_percent=10.0):
    """
    Classify progression from dual-head predictions
    
    Args:
        predictions_dict: Dict with keys 'baseline_fvc', 'slope', 'fvc_52'
                         Each contains {patient_id: predicted_value}
        threshold_percent: Decline threshold
    
    Returns:
        DataFrame with progression predictions and probabilities
    """
    results = []
    
    # Get all patient IDs
    patient_ids = set(predictions_dict['baseline_fvc'].keys())
    
    for pid in patient_ids:
        baseline_pred = predictions_dict['baseline_fvc'][pid]
        slope_pred = predictions_dict['slope'][pid]
        fvc52_pred = predictions_dict['fvc_52'][pid]
        
        # Predicted decline percentage
        decline_abs = baseline_pred - fvc52_pred
        decline_pct = 100 * decline_abs / baseline_pred if baseline_pred > 0 else 0.0
        
        # Binary classification
        predicted_progression = decline_pct >= threshold_percent
        
        # Soft probability (sigmoid around threshold)
        k = 0.3  # controls softness
        probability = 1 / (1 + np.exp(-(decline_pct - threshold_percent) / k))
        
        results.append({
            'patient_id': pid,
            'baseline_fvc_pred': baseline_pred,
            'slope_pred': slope_pred,
            'fvc52_pred': fvc52_pred,
            'decline_pct': decline_pct,
            'predicted_progression': int(predicted_progression),
            'probability': probability
        })
    
    return pd.DataFrame(results)


def compute_progression_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive progression prediction metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases
    if cm.shape == (1, 1):
        # Only one class present
        if y_true[0] == 0:
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, cm[0, 0]]])
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC metrics
    if len(np.unique(y_true)) > 1 and len(np.unique(y_probs)) > 1:
        auc_roc = roc_auc_score(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    else:
        auc_roc = np.nan
        ap = np.nan
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        precision_curve, recall_curve = np.array([1, 0]), np.array([0, 1])
    
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


def plot_patient_trajectory(patient_id, patient_data, pred_row, output_dir, progression_threshold=10.0):
    """
    Plot FVC trajectory for a single patient
    """
    baseline_pred = pred_row['baseline_fvc_pred']
    slope_pred = pred_row['slope_pred']
    fvc52_pred = pred_row['fvc52_pred']
    status = 'PROGRESSION' if pred_row['predicted_progression'] else 'STABLE'
    
    # True FVC measurements
    weeks = np.array(patient_data[patient_id]['weeks'])
    fvc_values = np.array(patient_data[patient_id]['fvc_values'])
    
    # Predicted trajectory
    pred_weeks = np.linspace(0, max(weeks.max(), 52), 100)
    pred_fvc = baseline_pred + slope_pred * pred_weeks
    
    # 10% decline threshold
    decline_threshold = baseline_pred * (1 - progression_threshold / 100)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(weeks, fvc_values, 'o-', color='black', label='True FVC', linewidth=2, markersize=8, alpha=0.7)
    plt.plot(pred_weeks, pred_fvc, '-', color='red', label='Predicted Trajectory', linewidth=2.5, alpha=0.8)
    plt.axhline(decline_threshold, color='red', linestyle=':', linewidth=2, label=f'{progression_threshold:.0f}% decline threshold')
    
    # Highlight baseline and week 52
    plt.scatter([0], [baseline_pred], color='green', s=120, zorder=5, label='Predicted Baseline', marker='s')
    plt.scatter([52], [fvc52_pred], color='red', s=120, marker='X', zorder=5, label='Predicted FVC@52')
    
    if 52 in weeks:
        true_fvc52 = fvc_values[np.where(weeks == 52)[0][0]]
        plt.scatter([52], [true_fvc52], color='blue', s=120, zorder=5, label='True FVC@52', marker='D')
    
    plt.title(f"Patient {patient_id}\nSlope: {slope_pred:.2f} ml/week | Decline: {pred_row['decline_pct']:.1f}% | Status: {status}", 
              fontsize=13, fontweight='bold')
    plt.xlabel('Weeks from Baseline', fontsize=12)
    plt.ylabel('FVC (ml)', fontsize=12)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'patient_{patient_id}.png', dpi=200)
    plt.close()


def plot_progression_results(fold_results, config):
    """Create comprehensive visualization of progression prediction results"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Aggregate all predictions across folds
    all_preds = pd.concat([r['predictions'] for r in fold_results], ignore_index=True)
    y_true = all_preds['true_progression'].values
    y_pred = all_preds['predicted_progression'].values
    y_prob = all_preds['probability'].values
    
    # Compute aggregate metrics
    metrics = compute_progression_metrics(y_true, y_pred, y_prob)
    cm = metrics['confusion_matrix']
    
    # 1. Per-fold metrics comparison
    ax1 = fig.add_subplot(gs[0, :])
    fold_data = []
    for r in fold_results:
        fold_data.append({
            'Fold': r['fold'],
            'Accuracy': r['metrics']['accuracy'],
            'Precision': r['metrics']['precision'],
            'Recall': r['metrics']['recall'],
            'F1': r['metrics']['f1_score'],
            'AUC': r['metrics']['auc_roc']
        })
    
    fold_df = pd.DataFrame(fold_data)
    x = np.arange(len(fold_results))
    width = 0.15
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    colors_bar = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - 2)
        ax1.bar(x + offset, fold_df[metric], width, label=metric, color=colors_bar[i], alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Fold Progression Prediction Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i}' for i in range(len(fold_results))], fontsize=11)
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc='lower right', fontsize=10, ncol=5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Aggregate Confusion Matrix
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Stable', 'Prog'], yticklabels=['Stable', 'Prog'],
                annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                linewidths=3, linecolor='black')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax2.text(j + 0.5, i + 0.75, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    acc = (cm[0, 0] + cm[1, 1]) / total
    ax2.set_title(f'Confusion Matrix\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True', fontsize=11, fontweight='bold')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[1, 1])
    fpr, tpr = metrics['roc_curve']
    ax3.plot(fpr, tpr, color='#3498db', lw=3, alpha=0.8, label=f'AUC = {metrics["auc_roc"]:.3f}')
    ax3.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    ax4 = fig.add_subplot(gs[1, 2])
    precision, recall = metrics['pr_curve']
    ax4.plot(recall, precision, color='#e74c3c', lw=3, alpha=0.8, label=f'AP = {metrics["average_precision"]:.3f}')
    ax4.set_xlabel('Recall', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    # 5. Decline distribution
    ax5 = fig.add_subplot(gs[2, 0])
    prog_mask = all_preds['true_progression'] == 1
    ax5.hist(all_preds.loc[~prog_mask, 'decline_pct'], bins=30, alpha=0.6, 
            label='Stable', color='green', edgecolor='black')
    ax5.hist(all_preds.loc[prog_mask, 'decline_pct'], bins=30, alpha=0.6, 
            label='Progression', color='red', edgecolor='black')
    ax5.axvline(config['progression_threshold'], color='black', linestyle='--', linewidth=2, 
               label=f'Threshold ({config["progression_threshold"]}%)')
    ax5.set_xlabel('Predicted Decline (%)', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Distribution of Predicted Decline', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Probability distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(all_preds.loc[~prog_mask, 'probability'], bins=30, alpha=0.6, 
            label='Stable (True)', color='green', edgecolor='black')
    ax6.hist(all_preds.loc[prog_mask, 'probability'], bins=30, alpha=0.6, 
            label='Progression (True)', color='red', edgecolor='black')
    ax6.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax6.set_xlabel('Progression Probability', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Metrics summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = f"""
    AGGREGATE METRICS
    {'='*30}
    
    Total Patients: {metrics['n_samples']}
    Progression: {metrics['n_positive']}
    Stable: {metrics['n_negative']}
    
    {'─'*30}
    
    Accuracy:    {metrics['accuracy']:.3f}
    Precision:   {metrics['precision']:.3f}
    Recall:      {metrics['recall']:.3f}
    F1-Score:    {metrics['f1_score']:.3f}
    Specificity: {metrics['specificity']:.3f}
    
    {'─'*30}
    
    AUC-ROC:     {metrics['auc_roc']:.3f}
    AP:          {metrics['average_precision']:.3f}
    """
    
    ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Dual-Head CNN: IPF Progression Prediction (≥{config["progression_threshold"]}% FVC Decline)',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = config['plots_dir'] / 'progression_evaluation_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved summary plot: {save_path}")
    plt.close()


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("="*80)
    print("DUAL-HEAD CNN PROGRESSION PREDICTION EVALUATION")
    print("="*80)
    print(f"\nProgression Threshold: ≥{CONFIG['progression_threshold']}% FVC decline")
    print(f"Evaluation: {CONFIG['n_folds']}-Fold Cross-Validation")
    
    # Load patient data for ground truth
    print("\n" + "="*80)
    print("LOADING PATIENT DATA")
    print("="*80)
    
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utilities import IPFDataLoader
    
    dl = IPFDataLoader(CONFIG['csv_path'], CONFIG['csv_features_path'], CONFIG['npy_dir'])
    patient_data, _ = dl.get_patient_data()
    
    print(f"✓ Loaded data for {len(patient_data)} patients")
    
    # Get ground truth labels
    ground_truth_labels = get_ground_truth_progression(patient_data, CONFIG['progression_threshold'])
    print(f"✓ Ground truth: {sum(ground_truth_labels.values())} progression, "
          f"{len(ground_truth_labels) - sum(ground_truth_labels.values())} stable")
    
    # Evaluate each fold
    fold_results = []
    all_predictions = []
    
    for fold_idx in range(CONFIG['n_folds']):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*80}")
        
        # Load predictions
        pred_file = CONFIG['predictions_dir'] / f'dual_head_predictions_fold{fold_idx}.pkl'
        
        if not pred_file.exists():
            print(f"⚠️  Missing predictions file: {pred_file}")
            continue
        
        with open(pred_file, 'rb') as f:
            fold_preds = pickle.load(f)
        
        # Use test predictions
        test_preds = fold_preds['test']
        
        print(f"  Loaded predictions for {len(test_preds['baseline_fvc'])} test patients")
        
        # Classify progression
        progression_df = classify_progression_from_dual_head(test_preds, CONFIG['progression_threshold'])
        
        # Add ground truth
        progression_df['true_progression'] = progression_df['patient_id'].map(ground_truth_labels)
        
        # Filter to patients with ground truth
        progression_df = progression_df[progression_df['true_progression'].notna()].copy()
        progression_df['fold'] = fold_idx
        
        if len(progression_df) == 0:
            print(f"  ⚠️  No patients with ground truth in fold {fold_idx}")
            continue
        
        print(f"  Evaluated {len(progression_df)} patients")
        print(f"    True progression: {int(progression_df['true_progression'].sum())}")
        print(f"    Predicted progression: {int(progression_df['predicted_progression'].sum())}")
        
        # Compute metrics
        fold_metrics = compute_progression_metrics(
            progression_df['true_progression'].values.astype(int),
            progression_df['predicted_progression'].values.astype(int),
            progression_df['probability'].values
        )
        
        print(f"    Accuracy: {fold_metrics['accuracy']:.3f}")
        print(f"    AUC-ROC:  {fold_metrics['auc_roc']:.3f}")
        print(f"    Recall:   {fold_metrics['recall']:.3f}")
        
        fold_results.append({
            'fold': fold_idx,
            'predictions': progression_df,
            'metrics': fold_metrics
        })
        
        all_predictions.append(progression_df)
        
        # Plot sample trajectories
        trajectory_dir = CONFIG['plots_dir'] / 'patient_trajectories' / f'fold{fold_idx}'
        valid_patients = progression_df['patient_id'].values
        sample_patients = random.sample(list(valid_patients), min(5, len(valid_patients)))
        
        for pid in sample_patients:
            row = progression_df[progression_df['patient_id'] == pid].iloc[0]
            plot_patient_trajectory(pid, patient_data, row, trajectory_dir, CONFIG['progression_threshold'])
        
        print(f"  ✓ Saved {len(sample_patients)} sample trajectory plots")
    
    if len(fold_results) == 0:
        print("\n❌ No valid folds to evaluate!")
        return
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    
    # Per-fold summary
    fold_summary_data = []
    for r in fold_results:
        m = r['metrics']
        fold_summary_data.append({
            'Fold': r['fold'],
            'N': m['n_samples'],
            'Prog': m['n_positive'],
            'Accuracy': f"{m['accuracy']:.3f}",
            'Precision': f"{m['precision']:.3f}",
            'Recall': f"{m['recall']:.3f}",
            'F1': f"{m['f1_score']:.3f}",
            'Specificity': f"{m['specificity']:.3f}",
            'AUC': f"{m['auc_roc']:.3f}",
            'AP': f"{m['average_precision']:.3f}"
        })
    
    fold_summary_df = pd.DataFrame(fold_summary_data)
    print("\n" + fold_summary_df.to_string(index=False))
    
    # Mean ± std
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc', 'average_precision']
    
    print(f"\n{'='*60}")
    print("MEAN ± STD ACROSS FOLDS")
    print(f"{'='*60}")
    
    for metric in metrics_names:
        values = [r['metrics'][metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.replace('_', ' ').title():15s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Save results
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    pred_path = CONFIG['results_dir'] / 'dual_head_progression_predictions.csv'
    combined_predictions.to_csv(pred_path, index=False)
    print(f"\n✓ Saved predictions: {pred_path}")
    
    summary_path = CONFIG['results_dir'] / 'dual_head_progression_summary.csv'
    fold_summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_progression_results(fold_results, CONFIG)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📁 Results: {CONFIG['results_dir']}")
    print(f"📁 Plots: {CONFIG['plots_dir']}")


if __name__ == "__main__":
    main()