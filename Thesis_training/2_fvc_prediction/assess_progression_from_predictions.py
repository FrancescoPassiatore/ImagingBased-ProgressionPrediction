"""
Post-hoc Progression Assessment from FVC Predictions

Computes progression classification from saved test predictions:
- Loads normalized predictions from test_predictions.csv
- Denormalizes FVC values using scalers
- Computes progression: (FVC_baseline - FVC_52weeks) / FVC_baseline * 100 > threshold%
- Tests multiple thresholds to find optimal
- Compares with ground truth progression labels
- Generates classification metrics and saves results
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_ground_truth_data(gt_path: Path) -> pd.DataFrame:
    """Load ground truth data with progression labels and baseline FVC"""
    df = pd.read_csv(gt_path)
    return df


def load_fvc_statistics(gt_path: Path, kfold_splits_path: Path):
    """
    Load FVC statistics (mean, std) from the ground truth data
    Returns dict with stats and scalers for each fold
    """
    # Load ground truth
    gt_df = pd.read_csv(gt_path)
    
    # Load K-fold splits
    with open(kfold_splits_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    print("\n" + "="*80)
    print("LOADING FVC STATISTICS FOR DENORMALIZATION")
    print("="*80)
    
    fold_stats = {}
    
    for fold_idx in kfold_splits:
        fold_data = kfold_splits[fold_idx]
        train_patients = fold_data['train']
        
        # Get training set FVC values
        train_df = gt_df[gt_df['PatientID'].isin(train_patients)].copy()
        
        # For StandardScaler with 2 columns, it computes separate mean/std for each
        # We need the combined statistics as they were used together
        fvc_values = train_df[['BaselineFVC', 'Week52FVC']].values
        scaler = StandardScaler()
        scaler.fit(fvc_values)
        
        # scaler.mean_[0] = baseline mean, scaler.mean_[1] = fvc52 mean
        # scaler.scale_[0] = baseline std, scaler.scale_[1] = fvc52 std
        
        fold_stats[fold_idx] = {
            'baseline_mean': scaler.mean_[0],
            'baseline_std': scaler.scale_[0],
            'fvc52_mean': scaler.mean_[1],
            'fvc52_std': scaler.scale_[1],
            'n_train_patients': len(train_patients),
            'scaler': scaler  # Store the fitted scaler
        }
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train patients: {len(train_patients)}")
        print(f"  Baseline FVC: mean={scaler.mean_[0]:.2f} mL, std={scaler.scale_[0]:.2f} mL")
        print(f"  Week52 FVC:   mean={scaler.mean_[1]:.2f} mL, std={scaler.scale_[1]:.2f} mL")
    
    return fold_stats


def assess_progression_from_predictions(
    predictions_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    baseline_fvc_normalized: np.ndarray,
    predicted_fvc52_normalized: np.ndarray,
    true_fvc52_normalized: np.ndarray,
    fvc_scaler: StandardScaler = None,
    threshold: float = 10.0
) -> dict:
    """
    Assess progression from FVC predictions
    
    Args:
        predictions_df: DataFrame with patient_id, true_fvc, predicted_fvc
        gt_df: Ground truth DataFrame with has_progressed labels
        baseline_fvc_normalized: Normalized baseline FVC values
        predicted_fvc52_normalized: Normalized predicted FVC at 52 weeks
        true_fvc52_normalized: Normalized true FVC at 52 weeks
        fvc_scaler: Scaler to denormalize FVC values
        threshold: Percentage decline threshold for progression (default 10%)
    
    Returns:
        Dict with classification metrics
    """
    # Denormalize FVC values if scaler is provided
    if fvc_scaler is not None:
        # Create dummy arrays for denormalization
        dummy_baseline = np.zeros((len(predicted_fvc52_normalized), 1))
        
        # Denormalize baseline
        baseline_with_dummy = np.column_stack([baseline_fvc_normalized, dummy_baseline])
        baseline_fvc = fvc_scaler.inverse_transform(baseline_with_dummy)[:, 0]
        
        # Denormalize predicted week 52
        pred_with_dummy = np.column_stack([dummy_baseline, predicted_fvc52_normalized])
        predicted_fvc52 = fvc_scaler.inverse_transform(pred_with_dummy)[:, 1]
        
        # Denormalize true week 52
        true_with_dummy = np.column_stack([dummy_baseline, true_fvc52_normalized])
        true_fvc52 = fvc_scaler.inverse_transform(true_with_dummy)[:, 1]
    else:
        # Assume values are already denormalized
        baseline_fvc = baseline_fvc_normalized
        predicted_fvc52 = predicted_fvc52_normalized
        true_fvc52 = true_fvc52_normalized
    
    # Get ground truth progression labels
    patient_ids = predictions_df['patient_id'].values
    gt_progression_labels = []
    for pid in patient_ids:
        label = gt_df[gt_df['PatientID'] == pid]['has_progressed'].values[0]
        gt_progression_labels.append(label)
    gt_progression_labels = np.array(gt_progression_labels)
    
    # Calculate percent decline for predictions
    percent_decline_pred = (baseline_fvc - predicted_fvc52) / baseline_fvc * 100
    predicted_progression = (percent_decline_pred > threshold).astype(int)
    
    # Calculate percent decline for ground truth (for reference)
    percent_decline_true = (baseline_fvc - true_fvc52) / baseline_fvc * 100
    
    # Classification metrics
    accuracy = accuracy_score(gt_progression_labels, predicted_progression)
    precision = precision_score(gt_progression_labels, predicted_progression, zero_division=0)
    recall = recall_score(gt_progression_labels, predicted_progression, zero_division=0)
    f1 = f1_score(gt_progression_labels, predicted_progression, zero_division=0)
    cm = confusion_matrix(gt_progression_labels, predicted_progression)
    
    # Calculate AUC using continuous percent decline
    try:
        auc = roc_auc_score(gt_progression_labels, percent_decline_pred)
    except ValueError:
        auc = 0.5  # If only one class present
    
    # Calculate specificity and sensitivity from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        specificity = 0
        sensitivity = 0
    
    results = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predicted_progression': predicted_progression,
        'gt_progression': gt_progression_labels,
        'percent_decline_pred': percent_decline_pred,
        'percent_decline_true': percent_decline_true,
        'baseline_fvc': baseline_fvc,
        'predicted_fvc52': predicted_fvc52,
        'true_fvc52': true_fvc52,
        'patient_ids': patient_ids,
        'n_predicted_progression': predicted_progression.sum(),
        'n_true_progression': gt_progression_labels.sum(),
        'classification_report': classification_report(
            gt_progression_labels, 
            predicted_progression,
            target_names=['No Progression', 'Progression'],
            zero_division=0
        )
    }
    
    return results


def plot_progression_assessment(results: dict, save_path: Path):
    """Plot progression assessment results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Prog', 'Prog'],
                yticklabels=['No Prog', 'Prog'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    axes[0, 0].set_title(f'Confusion Matrix\nAccuracy: {results["accuracy"]:.3f}, F1: {results["f1_score"]:.3f}')
    
    # 2. Percent Decline Distribution
    axes[0, 1].hist(results['percent_decline_pred'][results['gt_progression'] == 0], 
                    bins=20, alpha=0.5, label='No Progression (GT)', color='green')
    axes[0, 1].hist(results['percent_decline_pred'][results['gt_progression'] == 1], 
                    bins=20, alpha=0.5, label='Progression (GT)', color='red')
    axes[0, 1].axvline(x=results['threshold'], color='black', linestyle='--', 
                       label=f'Threshold ({results["threshold"]}%)')
    axes[0, 1].set_xlabel('Predicted FVC Decline (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Predicted FVC Decline Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter: Baseline vs Predicted Week 52
    colors = ['green' if p == 0 else 'red' for p in results['gt_progression']]
    axes[1, 0].scatter(results['baseline_fvc'], results['predicted_fvc52'], 
                       c=colors, alpha=0.6, s=50)
    axes[1, 0].plot([results['baseline_fvc'].min(), results['baseline_fvc'].max()],
                    [results['baseline_fvc'].min(), results['baseline_fvc'].max()],
                    'k--', alpha=0.5, label='No change')
    axes[1, 0].set_xlabel('Baseline FVC (mL)')
    axes[1, 0].set_ylabel('Predicted FVC at 52 weeks (mL)')
    axes[1, 0].set_title('FVC Prediction vs Baseline')
    axes[1, 0].legend(['No change line', 'No Prog (GT)', 'Prog (GT)'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Classification Metrics Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1_score'],
        results['auc']
    ]
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=['steelblue', 'coral', 'green', 'purple', 'orange'])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Classification Metrics')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {save_path}")


def threshold_analysis(
    percent_decline_pred: np.ndarray,
    gt_progression_labels: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Analyze classification performance across multiple thresholds
    
    Args:
        percent_decline_pred: Predicted FVC decline percentages
        gt_progression_labels: Ground truth progression labels (0/1)
        thresholds: Array of thresholds to test (default: 0% to 30% in 1% steps)
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0, 31, 1)  # 0% to 30% in 1% steps
    
    results = []
    
    for threshold in thresholds:
        predicted_progression = (percent_decline_pred > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(gt_progression_labels, predicted_progression)
        precision = precision_score(gt_progression_labels, predicted_progression, zero_division=0)
        recall = recall_score(gt_progression_labels, predicted_progression, zero_division=0)
        f1 = f1_score(gt_progression_labels, predicted_progression, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(gt_progression_labels, predicted_progression)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            specificity = 0
            sensitivity = 0
        
        # Youden's J statistic
        youdens_j = sensitivity + specificity - 1
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'youdens_j': youdens_j,
            'n_predicted_prog': predicted_progression.sum()
        })
    
    return pd.DataFrame(results)


def plot_threshold_analysis(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: Path
):
    """Plot threshold analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Metrics vs Threshold
    ax = axes[0, 0]
    ax.plot(threshold_df['threshold'], threshold_df['accuracy'], 'o-', label='Accuracy', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['precision'], 's-', label='Precision', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['recall'], '^-', label='Recall/Sensitivity', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['f1_score'], 'd-', label='F1-Score', markersize=3)
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold}%)')
    ax.set_xlabel('Threshold (%)')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics vs Threshold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 2. Sensitivity and Specificity
    ax = axes[0, 1]
    ax.plot(threshold_df['threshold'], threshold_df['sensitivity'], 'o-', label='Sensitivity', color='green', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['specificity'], 's-', label='Specificity', color='blue', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['youdens_j'], '^-', label="Youden's J", color='purple', markersize=3)
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold}%)')
    ax.set_xlabel('Threshold (%)')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity/Specificity vs Threshold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 3. Number of Predicted Progressions
    ax = axes[1, 0]
    ax.plot(threshold_df['threshold'], threshold_df['n_predicted_prog'], 'o-', color='coral', markersize=3)
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold}%)')
    ax.set_xlabel('Threshold (%)')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Predicted Progressions vs Threshold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. F1, Accuracy, Youden's J comparison
    ax = axes[1, 1]
    ax.plot(threshold_df['threshold'], threshold_df['f1_score'], 'o-', label='F1-Score', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['accuracy'], 's-', label='Accuracy', markersize=3)
    ax.plot(threshold_df['threshold'], threshold_df['youdens_j'], '^-', label="Youden's J", markersize=3)
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold}%)')
    ax.set_xlabel('Threshold (%)')
    ax.set_ylabel('Score')
    ax.set_title('Key Metrics for Threshold Selection')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved threshold analysis plot: {save_path}")


def plot_aggregate_progression_assessment(
    all_fold_results: list,
    save_path: Path
):
    """
    Create aggregate plot showing progression assessment across all folds
    
    Args:
        all_fold_results: List of dictionaries containing results from each fold
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data from all folds
    fold_indices = [r['fold_idx'] for r in all_fold_results]
    n_folds = len(fold_indices)
    
    # Metrics across folds
    metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    
    # 1. Bar plot with error bars for all metrics (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    metric_values = [[r[m] for r in all_fold_results] for m in metrics]
    means = [np.mean(vals) for vals in metric_values]
    stds = [np.std(vals) for vals in metric_values]
    
    x_pos = np.arange(len(metrics))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title(f'Mean Metrics Across {n_folds} Folds (±1 SD)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2., mean,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # 2. Metric variance across folds - box plots (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    f1_values = [r['f1_score'] for r in all_fold_results]
    auc_values = [r['auc'] for r in all_fold_results]
    accuracy_values = [r['accuracy'] for r in all_fold_results]
    
    box_data = [f1_values, auc_values, accuracy_values]
    bp = ax2.boxplot(box_data, labels=['F1', 'AUC', 'Acc'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1abc9c', '#34495e', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Key Metrics Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Per-fold metrics heatmap (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_matrix = np.array([[r[m] for m in metrics] for r in all_fold_results])
    im = ax3.imshow(metrics_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_xticks(np.arange(n_folds))
    ax3.set_yticks(np.arange(len(metrics)))
    ax3.set_xticklabels([f'F{i}' for i in fold_indices], fontsize=9)
    ax3.set_yticklabels(metric_labels, fontsize=9)
    ax3.set_xlabel('Fold', fontsize=11)
    ax3.set_title('Metrics Heatmap by Fold', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Score', fontsize=10)
    
    # Add values to heatmap
    for i in range(len(metrics)):
        for j in range(n_folds):
            ax3.text(j, i, f'{metrics_matrix[j, i]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if metrics_matrix[j, i] < 0.5 else 'black')
    
    # 4. Confusion matrices for all folds (middle center and right)
    ax4 = fig.add_subplot(gs[1, 1])
    # Aggregate confusion matrix
    total_cm = np.zeros((2, 2))
    for r in all_fold_results:
        total_cm += r['confusion_matrix']
    
    im = ax4.imshow(total_cm, cmap='Blues', alpha=0.7)
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['No Prog', 'Prog'])
    ax4.set_yticklabels(['No Prog', 'Prog'])
    ax4.set_xlabel('Predicted', fontsize=11)
    ax4.set_ylabel('True', fontsize=11)
    ax4.set_title(f'Aggregate Confusion Matrix\n(Total: {int(total_cm.sum())} samples)', 
                  fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, f'{int(total_cm[i, j])}\n({total_cm[i, j]/total_cm.sum()*100:.1f}%)',
                           ha='center', va='center', fontsize=11,
                           color='white' if total_cm[i, j] > total_cm.max()/2 else 'black')
    
    # 5. Progression counts per fold (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    n_predicted = [r['n_predicted_progression'] for r in all_fold_results]
    n_true = [r['n_true_progression'] for r in all_fold_results]
    
    x = np.arange(n_folds)
    width = 0.35
    bars1 = ax5.bar(x - width/2, n_true, width, label='True', alpha=0.7, color='#3498db')
    bars2 = ax5.bar(x + width/2, n_predicted, width, label='Predicted', alpha=0.7, color='#e74c3c')
    
    ax5.set_xlabel('Fold', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Progression Counts by Fold', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'F{i}' for i in fold_indices])
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 6. F1-Score trend across folds (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(fold_indices, f1_values, 'o-', linewidth=2, markersize=8, color='#1abc9c', label='F1-Score')
    ax6.axhline(y=np.mean(f1_values), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(f1_values):.3f}')
    ax6.fill_between(fold_indices, 
                     np.mean(f1_values) - np.std(f1_values),
                     np.mean(f1_values) + np.std(f1_values),
                     alpha=0.2, color='red')
    ax6.set_xlabel('Fold', fontsize=11)
    ax6.set_ylabel('F1-Score', fontsize=11)
    ax6.set_title('F1-Score Across Folds', fontsize=12, fontweight='bold')
    ax6.set_xticks(fold_indices)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    # 7. Sensitivity vs Specificity per fold (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    sensitivity_values = [r['sensitivity'] for r in all_fold_results]
    specificity_values = [r['specificity'] for r in all_fold_results]
    
    ax7.scatter(specificity_values, sensitivity_values, s=150, alpha=0.6, 
               c=fold_indices, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add fold labels
    for i, (spec, sens, fold) in enumerate(zip(specificity_values, sensitivity_values, fold_indices)):
        ax7.annotate(f'F{fold}', (spec, sens), fontsize=9, ha='center', va='center')
    
    # Add mean point
    mean_spec = np.mean(specificity_values)
    mean_sens = np.mean(sensitivity_values)
    ax7.scatter([mean_spec], [mean_sens], s=300, c='red', marker='*', 
               edgecolors='black', linewidth=2, label='Mean', zorder=10)
    
    ax7.set_xlabel('Specificity', fontsize=11)
    ax7.set_ylabel('Sensitivity', fontsize=11)
    ax7.set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
    ax7.set_xlim([0, 1.05])
    ax7.set_ylim([0, 1.05])
    ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics table (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create summary table
    summary_data = []
    for metric, label in zip(metrics, metric_labels):
        values = [r[metric] for r in all_fold_results]
        summary_data.append([
            label,
            f'{np.mean(values):.3f}',
            f'{np.std(values):.3f}',
            f'{np.min(values):.3f}',
            f'{np.max(values):.3f}'
        ])
    
    table = ax8.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved aggregate plot: {save_path}")


def plot_threshold_comparison(
    percent_decline_pred: np.ndarray,
    gt_progression_labels: np.ndarray,
    reference_threshold: float,
    optimal_threshold: float,
    optimal_metrics: dict,
    save_path: Path
):
    """
    Compare confusion matrices and metrics between reference (e.g., 10%) and optimal threshold
    
    Args:
        percent_decline_pred: Predicted FVC decline percentages
        gt_progression_labels: Ground truth progression labels (0/1)
        reference_threshold: Reference threshold (e.g., 10%)
        optimal_threshold: Optimal threshold found by analysis
        optimal_metrics: Dictionary with optimal threshold metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Calculate predictions for both thresholds
    pred_reference = (percent_decline_pred > reference_threshold).astype(int)
    pred_optimal = (percent_decline_pred > optimal_threshold).astype(int)
    
    # Calculate metrics for reference threshold
    cm_ref = confusion_matrix(gt_progression_labels, pred_reference)
    acc_ref = accuracy_score(gt_progression_labels, pred_reference)
    prec_ref = precision_score(gt_progression_labels, pred_reference, zero_division=0)
    rec_ref = recall_score(gt_progression_labels, pred_reference, zero_division=0)
    f1_ref = f1_score(gt_progression_labels, pred_reference, zero_division=0)
    
    if cm_ref.shape == (2, 2):
        tn, fp, fn, tp = cm_ref.ravel()
        sens_ref = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_ref = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens_ref = 0
        spec_ref = 0
    
    # Calculate metrics for optimal threshold
    cm_opt = confusion_matrix(gt_progression_labels, pred_optimal)
    acc_opt = optimal_metrics['accuracy']
    prec_opt = optimal_metrics['precision']
    rec_opt = optimal_metrics['recall']
    f1_opt = optimal_metrics['f1_score']
    sens_opt = optimal_metrics['sensitivity']
    spec_opt = optimal_metrics['specificity']
    
    # 1. Reference Threshold Confusion Matrix (top left)
    ax = axes[0, 0]
    im = ax.imshow(cm_ref, cmap='Blues', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Prog', 'Prog'])
    ax.set_yticklabels(['No Prog', 'Prog'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Reference Threshold: {reference_threshold}%\nF1={f1_ref:.3f}, Acc={acc_ref:.3f}', 
                 fontsize=11, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{int(cm_ref[i, j])}\n({cm_ref[i, j]/cm_ref.sum()*100:.1f}%)',
                          ha='center', va='center', fontsize=11,
                          color='white' if cm_ref[i, j] > cm_ref.max()/2 else 'black')
    
    # 2. Optimal Threshold Confusion Matrix (top center)
    ax = axes[0, 1]
    im = ax.imshow(cm_opt, cmap='Greens', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Prog', 'Prog'])
    ax.set_yticklabels(['No Prog', 'Prog'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Optimal Threshold: {optimal_threshold}%\nF1={f1_opt:.3f}, Acc={acc_opt:.3f}', 
                 fontsize=11, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{int(cm_opt[i, j])}\n({cm_opt[i, j]/cm_opt.sum()*100:.1f}%)',
                          ha='center', va='center', fontsize=11,
                          color='white' if cm_opt[i, j] > cm_opt.max()/2 else 'black')
    
    # 3. Difference Matrix (top right)
    ax = axes[0, 2]
    cm_diff = cm_opt.astype(float) - cm_ref.astype(float)
    max_abs_diff = np.abs(cm_diff).max()
    im = ax.imshow(cm_diff, cmap='RdBu_r', alpha=0.7, vmin=-max_abs_diff, vmax=max_abs_diff)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Prog', 'Prog'])
    ax.set_yticklabels(['No Prog', 'Prog'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Difference (Optimal - Reference)\nΔF1={f1_opt-f1_ref:+.3f}', 
                 fontsize=11, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if abs(cm_diff[i, j]) > max_abs_diff/2 else 'black'
            text = ax.text(j, i, f'{int(cm_diff[i, j]):+d}',
                          ha='center', va='center', fontsize=12, fontweight='bold',
                          color=color)
    
    # 4. Metrics Comparison Bar Chart (bottom left)
    ax = axes[1, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    ref_values = [acc_ref, prec_ref, rec_ref, f1_ref]
    opt_values = [acc_opt, prec_opt, rec_opt, f1_opt]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, ref_values, width, label=f'{reference_threshold}%', alpha=0.7, color='#3498db')
    bars2 = ax.bar(x + width/2, opt_values, width, label=f'{optimal_threshold}%', alpha=0.7, color='#2ecc71')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Sensitivity & Specificity Comparison (bottom center)
    ax = axes[1, 1]
    metrics_names = ['Sensitivity', 'Specificity']
    ref_values = [sens_ref, spec_ref]
    opt_values = [sens_opt, spec_opt]
    
    x = np.arange(len(metrics_names))
    bars1 = ax.bar(x - width/2, ref_values, width, label=f'{reference_threshold}%', alpha=0.7, color='#e74c3c')
    bars2 = ax.bar(x + width/2, opt_values, width, label=f'{optimal_threshold}%', alpha=0.7, color='#f39c12')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Sensitivity & Specificity', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Summary Comparison Table (bottom right)
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Create comparison table
    comparison_data = [
        ['Threshold', f'{reference_threshold}%', f'{optimal_threshold}%', 'Δ'],
        ['Accuracy', f'{acc_ref:.3f}', f'{acc_opt:.3f}', f'{acc_opt-acc_ref:+.3f}'],
        ['Precision', f'{prec_ref:.3f}', f'{prec_opt:.3f}', f'{prec_opt-prec_ref:+.3f}'],
        ['Recall', f'{rec_ref:.3f}', f'{rec_opt:.3f}', f'{rec_opt-rec_ref:+.3f}'],
        ['Sensitivity', f'{sens_ref:.3f}', f'{sens_opt:.3f}', f'{sens_opt-sens_ref:+.3f}'],
        ['Specificity', f'{spec_ref:.3f}', f'{spec_opt:.3f}', f'{spec_opt-spec_ref:+.3f}'],
        ['F1-Score', f'{f1_ref:.3f}', f'{f1_opt:.3f}', f'{f1_opt-f1_ref:+.3f}'],
        ['Pred. Prog.', f'{pred_reference.sum()}', f'{pred_optimal.sum()}', f'{int(pred_optimal.sum()-pred_reference.sum()):+d}']
    ]
    
    table = ax.table(cellText=comparison_data,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color metric names column
    for i in range(1, len(comparison_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')
    
    # Color delta column based on value
    for i in range(1, len(comparison_data)):
        cell = table[(i, 3)]
        value_str = comparison_data[i][3]
        if value_str.startswith('+'):
            cell.set_facecolor('#d5f4e6')  # Light green
        elif value_str.startswith('-'):
            cell.set_facecolor('#fadbd8')  # Light red
    
    ax.set_title('Threshold Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved threshold comparison plot: {save_path}")


def find_optimal_threshold(
    threshold_df: pd.DataFrame,
    criterion: str = 'f1'
) -> dict:
    """
    Find optimal threshold based on specified criterion
    
    Args:
        threshold_df: DataFrame with metrics for each threshold
        criterion: Metric to optimize ('f1', 'accuracy', 'youdens_j', 'balanced')
    
    Returns:
        Dict with optimal threshold and corresponding metrics
    """
    if criterion == 'f1':
        optimal_idx = threshold_df['f1_score'].idxmax()
    elif criterion == 'accuracy':
        optimal_idx = threshold_df['accuracy'].idxmax()
    elif criterion == 'youdens_j':
        optimal_idx = threshold_df['youdens_j'].idxmax()
    elif criterion == 'balanced':
        # Average of F1, Accuracy, and Youden's J
        combined_metric = (threshold_df['f1_score'] + threshold_df['accuracy'] + threshold_df['youdens_j']) / 3
        optimal_idx = combined_metric.idxmax()
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    optimal_row = threshold_df.loc[optimal_idx]
    
    return {
        'threshold': optimal_row['threshold'],
        'criterion': criterion,
        'f1_score': optimal_row['f1_score'],
        'accuracy': optimal_row['accuracy'],
        'youdens_j': optimal_row['youdens_j'],
        'sensitivity': optimal_row['sensitivity'],
        'specificity': optimal_row['specificity'],
        'precision': optimal_row['precision'],
        'recall': optimal_row['recall']
    }


def process_ablation_experiment(
    ablation_dir: Path,
    gt_path: Path,
    kfold_splits_path: Path,
    threshold: float = 10.0,
    run_threshold_analysis: bool = False,
    threshold_range: np.ndarray = None,
    threshold_criterion: str = 'f1'
):
    """
    Process all folds for an ablation experiment
    
    Args:
        ablation_dir: Directory containing fold results
        gt_path: Path to ground truth CSV
        kfold_splits_path: Path to K-fold splits pickle
        threshold: Fixed threshold for progression (used if run_threshold_analysis=False)
        run_threshold_analysis: If True, perform threshold optimization
        threshold_range: Array of thresholds to test (default: 0-30% in 1% steps)
        threshold_criterion: Criterion for optimal threshold ('f1', 'accuracy', 'youdens_j', 'balanced')
    """
    print(f"\n{'='*70}")
    print(f"Processing: {ablation_dir.name}")
    print(f"{'='*70}")
    
    # Load ground truth data and FVC statistics
    gt_df = load_ground_truth_data(gt_path)
    fold_stats = load_fvc_statistics(gt_path, kfold_splits_path)
    
    # Load K-fold splits
    with open(kfold_splits_path, 'rb') as f:
        kfold_splits = pickle.load(f)
    
    fold_results = []
    fold_results_full = []  # Store full results including confusion matrices for plotting
    
    # Process each fold
    for fold_idx in sorted(kfold_splits.keys()):
        fold_dir = ablation_dir / f"fold_{fold_idx}"
        predictions_path = fold_dir / "test_predictions.csv"
        
        if not predictions_path.exists():
            print(f"⚠️  Fold {fold_idx}: No predictions found, skipping")
            continue
        
        print(f"\n--- Fold {fold_idx} ---")
        
        # Load predictions
        predictions_df = pd.read_csv(predictions_path)
        print(f"  Loaded {len(predictions_df)} test predictions")
        
        # Get baseline FVC values for each patient (denormalized)
        baseline_fvc_values = []
        for pid in predictions_df['patient_id']:
            baseline = gt_df[gt_df['PatientID'] == pid]['BaselineFVC'].values[0]
            baseline_fvc_values.append(baseline)
        baseline_fvc_values = np.array(baseline_fvc_values)
        
        # Get the scaler for this fold
        fvc_scaler = fold_stats[fold_idx]['scaler']
        
        # Get normalized predictions and targets from CSV
        predicted_fvc52_norm = predictions_df['predicted_fvc'].values
        true_fvc52_norm = predictions_df['true_fvc'].values
        
        # Normalize baseline FVC using the same scaler
        baseline_with_dummy = np.column_stack([baseline_fvc_values, np.zeros(len(baseline_fvc_values))])
        baseline_fvc_norm = fvc_scaler.transform(baseline_with_dummy)[:, 0]
        
        # Assess progression
        results = assess_progression_from_predictions(
            predictions_df=predictions_df,
            gt_df=gt_df,
            baseline_fvc_normalized=baseline_fvc_norm,
            predicted_fvc52_normalized=predicted_fvc52_norm,
            true_fvc52_normalized=true_fvc52_norm,
            fvc_scaler=fvc_scaler,
            threshold=threshold
        )
        
        # Print results
        print(f"\n  Progression Assessment (Threshold: {threshold}%):")
        print(f"    Accuracy:    {results['accuracy']:.4f}")
        print(f"    Precision:   {results['precision']:.4f}")
        print(f"    Recall:      {results['recall']:.4f}")
        print(f"    F1-Score:    {results['f1_score']:.4f}")
        print(f"    AUC:         {results['auc']:.4f}")
        print(f"    Sensitivity: {results['sensitivity']:.4f}")
        print(f"    Specificity: {results['specificity']:.4f}")
        print(f"\n  Predicted: {results['n_predicted_progression']}/{len(predictions_df)} progressed")
        print(f"  True:      {results['n_true_progression']}/{len(predictions_df)} progressed")
        
        print(f"\n  Classification Report:")
        print(results['classification_report'])
        
        # Save detailed results
        results_detailed_df = pd.DataFrame({
            'patient_id': results['patient_ids'],
            'baseline_fvc': results['baseline_fvc'],
            'predicted_fvc52': results['predicted_fvc52'],
            'true_fvc52': results['true_fvc52'],
            'percent_decline_pred': results['percent_decline_pred'],
            'percent_decline_true': results['percent_decline_true'],
            'predicted_progression': results['predicted_progression'],
            'true_progression': results['gt_progression']
        })
        results_detailed_df.to_csv(fold_dir / "progression_assessment_detailed.csv", index=False)
        print(f"  Saved: progression_assessment_detailed.csv")
        
        # Save summary metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Threshold', 'Accuracy', 'Precision', 'Recall', 'Sensitivity', 
                       'Specificity', 'F1-Score', 'AUC', 'N_Predicted_Prog', 'N_True_Prog'],
            'Value': [
                results['threshold'],
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['sensitivity'],
                results['specificity'],
                results['f1_score'],
                results['auc'],
                results['n_predicted_progression'],
                results['n_true_progression']
            ]
        })
        metrics_df.to_csv(fold_dir / "progression_assessment_metrics.csv", index=False)
        print(f"  Saved: progression_assessment_metrics.csv")
        
        # Plot results
        plot_progression_assessment(results, fold_dir / "progression_assessment_plot.png")
        
        # Threshold analysis (if requested)
        if run_threshold_analysis:
            print(f"\n  Running threshold analysis...")
            threshold_df = threshold_analysis(
                percent_decline_pred=results['percent_decline_pred'],
                gt_progression_labels=results['gt_progression'],
                thresholds=threshold_range
            )
            
            # Find optimal threshold
            optimal_info = find_optimal_threshold(threshold_df, criterion=threshold_criterion)
            print(f"  Optimal threshold ({threshold_criterion}): {optimal_info['threshold']}%")
            print(f"    F1={optimal_info['f1_score']:.4f}, Acc={optimal_info['accuracy']:.4f}, J={optimal_info['youdens_j']:.4f}")
            
            # Save threshold analysis results
            threshold_df.to_csv(fold_dir / "threshold_analysis.csv", index=False)
            print(f"  Saved: threshold_analysis.csv")
            
            # Save optimal threshold info
            optimal_df = pd.DataFrame([optimal_info])
            optimal_df.to_csv(fold_dir / "optimal_threshold.csv", index=False)
            print(f"  Saved: optimal_threshold.csv")
            
            # Plot threshold analysis
            plot_threshold_analysis(
                threshold_df=threshold_df,
                optimal_threshold=optimal_info['threshold'],
                save_path=fold_dir / "threshold_analysis_plot.png"
            )
            
            # Plot threshold comparison (reference 10% vs optimal)
            plot_threshold_comparison(
                percent_decline_pred=results['percent_decline_pred'],
                gt_progression_labels=results['gt_progression'],
                reference_threshold=10.0,  # Standard reference threshold
                optimal_threshold=optimal_info['threshold'],
                optimal_metrics=optimal_info,
                save_path=fold_dir / "threshold_comparison_plot.png"
            )
            
            # Store optimal threshold info in fold results
            fold_results.append({
                'fold_idx': fold_idx,
                'optimal_threshold': optimal_info['threshold'],
                **{k: v for k, v in results.items() if k not in ['confusion_matrix', 'predicted_progression', 
                                                                   'gt_progression', 'percent_decline_pred', 
                                                                   'percent_decline_true', 'patient_ids',
                                                                   'baseline_fvc', 'predicted_fvc52', 'true_fvc52',
                                                                   'classification_report']}
            })
            # Store full results for plotting
            fold_results_full.append({
                'fold_idx': fold_idx,
                **results
            })
        else:
            fold_results.append({
                'fold_idx': fold_idx,
                **{k: v for k, v in results.items() if k not in ['confusion_matrix', 'predicted_progression', 
                                                                   'gt_progression', 'percent_decline_pred', 
                                                               'percent_decline_true', 'baseline_fvc',
                                                               'predicted_fvc52', 'true_fvc52', 'patient_ids',
                                                               'classification_report']}
            })
            # Store full results for plotting
            fold_results_full.append({
                'fold_idx': fold_idx,
                **results
            })
    
    # Aggregate results across folds
    if fold_results:
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS ACROSS FOLDS")
        print(f"{'='*70}")
        
        metrics_to_aggregate = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'auc']
        aggregate_summary = {
            'Metric': metrics_to_aggregate,
            'Mean': [np.mean([r[m] for r in fold_results]) for m in metrics_to_aggregate],
            'Std': [np.std([r[m] for r in fold_results]) for m in metrics_to_aggregate],
            'Min': [np.min([r[m] for r in fold_results]) for m in metrics_to_aggregate],
            'Max': [np.max([r[m] for r in fold_results]) for m in metrics_to_aggregate]
        }
        
        summary_df = pd.DataFrame(aggregate_summary)
        print(f"\n{summary_df.to_string(index=False)}")
        
        summary_df.to_csv(ablation_dir / "progression_assessment_aggregate.csv", index=False)
        print(f"\nSaved: {ablation_dir / 'progression_assessment_aggregate.csv'}")
        
        # Create aggregate plot across all folds
        if fold_results_full:
            plot_aggregate_progression_assessment(
                all_fold_results=fold_results_full,
                save_path=ablation_dir / "progression_assessment_aggregate_plot.png"
            )
        
        # If threshold analysis was run, print optimal thresholds summary
        if run_threshold_analysis and fold_results:
            print(f"\n{'='*70}")
            print("OPTIMAL THRESHOLDS SUMMARY")
            print(f"{'='*70}")
            
            optimal_thresholds = [r['optimal_threshold'] for r in fold_results]
            print(f"Mean optimal threshold: {np.mean(optimal_thresholds):.2f}% (±{np.std(optimal_thresholds):.2f}%)")
            print(f"Range: [{np.min(optimal_thresholds):.1f}% - {np.max(optimal_thresholds):.1f}%]")
            print(f"Median: {np.median(optimal_thresholds):.2f}%")
            
            # Save optimal thresholds summary
            optimal_summary = pd.DataFrame({
                'Fold': [r['fold_idx'] for r in fold_results],
                'Optimal_Threshold': optimal_thresholds,
                'F1_Score': [r['f1_score'] for r in fold_results],
                'Accuracy': [r['accuracy'] for r in fold_results]
            })
            optimal_summary.to_csv(ablation_dir / "optimal_thresholds_summary.csv", index=False)
            print(f"\nSaved: {ablation_dir / 'optimal_thresholds_summary.csv'}")


def main():
    """Main function with command-line argument support"""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Assess progression from FVC predictions')
    parser.add_argument('--threshold', type=float, default=10.0, 
                        help='Fixed threshold for progression (default: 10.0%%)')
    parser.add_argument('--analyze-thresholds', action='store_true',
                        help='Run threshold analysis to find optimal threshold')
    parser.add_argument('--threshold-min', type=float, default=0.0,
                        help='Minimum threshold for analysis (default: 0.0%%)')
    parser.add_argument('--threshold-max', type=float, default=30.0,
                        help='Maximum threshold for analysis (default: 30.0%%)')
    parser.add_argument('--threshold-step', type=float, default=1.0,
                        help='Step size for threshold sweep (default: 1.0%%)')
    parser.add_argument('--criterion', type=str, default='f1', 
                        choices=['f1', 'accuracy', 'youdens_j', 'balanced'],
                        help='Optimization criterion for threshold selection (default: f1)')
    
    args = parser.parse_args()
    
    # Configuration
    BASE_DIR = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\2_fvc_prediction")
    RESULTS_DIR = BASE_DIR / "ablation_study_results_stratified"
    GT_PATH = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv")
    KFOLD_SPLITS_PATH = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl")
    
    print("="*70)
    print("PROGRESSION ASSESSMENT FROM FVC PREDICTIONS")
    print("="*70)
    
    if args.analyze_thresholds:
        print(f"Mode: THRESHOLD ANALYSIS")
        print(f"Threshold range: {args.threshold_min}% to {args.threshold_max}% (step: {args.threshold_step}%)")
        print(f"Optimization criterion: {args.criterion}")
        threshold_range = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    else:
        print(f"Mode: FIXED THRESHOLD")
        print(f"Threshold: {args.threshold}% FVC decline")
        threshold_range = None
    
    print(f"Results directory: {RESULTS_DIR}")
    
    # Process each ablation experiment
    ablation_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('ablation_')]
    
    if not ablation_dirs:
        print(f"\n⚠️  No ablation directories found in {RESULTS_DIR}")
        return
    
    for ablation_dir in sorted(ablation_dirs):
        try:
            process_ablation_experiment(
                ablation_dir=ablation_dir,
                gt_path=GT_PATH,
                kfold_splits_path=KFOLD_SPLITS_PATH,
                threshold=args.threshold,
                run_threshold_analysis=args.analyze_thresholds,
                threshold_range=threshold_range,
                threshold_criterion=args.criterion
            )
        except Exception as e:
            print(f"\n❌ Error processing {ablation_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✓ PROGRESSION ASSESSMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
