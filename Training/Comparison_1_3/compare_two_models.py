"""
Comparison: prediction_fold vs FVC@52
====================================

Models:
1. prediction_fold: CNN slope + corrector → FVC trajectory → binary progression
2. FVC@52: CNN embedding → FVC@52w value → convert to binary (≥10% decline)

Target: Binary classification - ≥10% FVC decline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD TRUE LABELS
# ============================================================================

def load_true_labels_from_fvc_file(fvc_csv_path, baseline_fvc_dict):
    """
    Load true progression labels from FVC predictions CSV.
    Uses actual FVC values to determine true label.
    
    Label = 1 if (baseline - actual_fvc) / baseline ≥ 0.10
    """
    df = pd.read_csv(fvc_csv_path)
    
    labels = {}
    
    for _, row in df.iterrows():
        pid = row['patient_id']
        true_fvc = row['true_fvc']
        
        if pid not in baseline_fvc_dict:
            continue
        
        baseline = baseline_fvc_dict[pid]
        decline_pct = 100 * (baseline - true_fvc) / baseline
        
        labels[pid] = int(decline_pct >= 10.0)
    
    return labels


def compute_feature_stats(handcrafted_dict, patient_ids, feature_names):
    """Extract mean/std from training data for normalization"""
    stats = {}
    for feature in feature_names:
        values = []
        for pid in patient_ids:
            if pid in handcrafted_dict and feature in handcrafted_dict[pid]:
                v = handcrafted_dict[pid][feature]
                if isinstance(v, (int, float)) and not np.isnan(v):
                    values.append(v)
        
        if len(values) > 0:
            stats[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
    return stats


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob=None, label="Model"):
    """
    Compute comprehensive metrics for binary classification
    """
    metrics = {
        'Model': label,
        'N': len(y_true),
        'Positives': int(np.sum(y_true)),
        'Negatives': int(len(y_true) - np.sum(y_true)),
    }
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Specificity'] = 0
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Probability-based metrics
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['AUC-ROC'] = np.nan
        
        try:
            metrics['AP'] = average_precision_score(y_true, y_prob)
        except:
            metrics['AP'] = np.nan
    
    metrics['CM'] = confusion_matrix(y_true, y_pred)
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(model1_name, model1_data, model2_name, model2_data, 
                   save_path='two_model_comparison.png'):
    """
    Create comparison dashboard for 2 models
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = ['#3498db', '#e74c3c']
    
    # ========================================================================
    # 1. ROC CURVES
    # ========================================================================
    ax_roc = fig.add_subplot(gs[0, 0])
    
    for (name, data), color in zip([(model1_name, model1_data), (model2_name, model2_data)], colors):
        y_true = data['y_true']
        y_prob = data['y_prob']
        metrics = data['metrics']
        
        if 'AUC-ROC' in metrics and not np.isnan(metrics['AUC-ROC']):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = metrics['AUC-ROC']
            ax_roc.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={auc_score:.3f})', color=color)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate', fontsize=10)
    ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. METRICS COMPARISON
    # ========================================================================
    ax_metrics = fig.add_subplot(gs[0, 1])
    
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metric_keys))
    width = 0.35
    
    for i, (name, data) in enumerate([(model1_name, model1_data), (model2_name, model2_data)]):
        metrics = data['metrics']
        values = [metrics.get(k, 0) for k in metric_keys]
        ax_metrics.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.8)
    
    ax_metrics.set_ylabel('Score', fontsize=10)
    ax_metrics.set_title('Key Metrics', fontsize=11, fontweight='bold')
    ax_metrics.set_xticks(x + width/2)
    ax_metrics.set_xticklabels(metric_keys, fontsize=9, rotation=45, ha='right')
    ax_metrics.legend(fontsize=9)
    ax_metrics.set_ylim([0, 1])
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 3. SENSITIVITY vs SPECIFICITY
    # ========================================================================
    ax_sens_spec = fig.add_subplot(gs[0, 2])
    
    names = [model1_name, model2_name]
    sensitivities = [model1_data['metrics']['Recall'], model2_data['metrics']['Recall']]
    specificities = [model1_data['metrics']['Specificity'], model2_data['metrics']['Specificity']]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax_sens_spec.bar(x - width/2, sensitivities, width, label='Sensitivity', color='#3498db', alpha=0.8)
    ax_sens_spec.bar(x + width/2, specificities, width, label='Specificity', color='#e74c3c', alpha=0.8)
    
    ax_sens_spec.set_ylabel('Score', fontsize=10)
    ax_sens_spec.set_title('Sensitivity vs Specificity', fontsize=11, fontweight='bold')
    ax_sens_spec.set_xticks(x)
    ax_sens_spec.set_xticklabels(names, fontsize=10)
    ax_sens_spec.set_ylim([0, 1])
    ax_sens_spec.legend(fontsize=9)
    ax_sens_spec.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 4. CONFUSION MATRICES
    # ========================================================================
    for idx, (name, data) in enumerate([(model1_name, model1_data), (model2_name, model2_data)]):
        ax = fig.add_subplot(gs[1, idx])
        cm = data['metrics']['CM']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                   xticklabels=['Stable', 'Progression'],
                   yticklabels=['Stable', 'Progression'])
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)
    
    # ========================================================================
    # 5. DETAILED METRICS TABLE
    # ========================================================================
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis('off')
    
    table_data = []
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']
    
    for name, data in [(model1_name, model1_data), (model2_name, model2_data)]:
        metrics = data['metrics']
        row = [name] + [f"{metrics.get(k, 0):.3f}" for k in metric_keys]
        table_data.append(row)
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Model'] + metric_keys,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(metric_keys) + 1):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, 3):
        for j in range(len(metric_keys) + 1):
            table[(i, j)].set_facecolor(['#ecf0f1', '#d5dbdb'][i % 2])
    
    plt.suptitle(f'Progression Prediction - {model1_name} vs {model2_name}', 
                fontsize=13, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to: {save_path}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("PROGRESSION PREDICTION - TWO MODEL COMPARISON")
    print("="*80)
    print("\nTarget: ≥10% FVC decline")
    print("Model 1: prediction_fold (Slope-based)")
    print("Model 2: FVC@52 (CNN embedding)")
    
    # ========================================================================
    # STEP 1: LOAD PREDICTIONS
    # ========================================================================
    print("\n[1/4] Loading predictions...")
    
    # Model 1: prediction_fold
    try:
        fold_df = pd.read_csv('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\prediction_fold_final.csv')
        print(f"✓ Model 1 (prediction_fold): {len(fold_df)} predictions")
    except FileNotFoundError:
        print("❌ Error: 'prediction_fold_final.csv' not found")
        print("   Run: python Training/Progression_prediction_slope_1/prediction_fold.py")
        exit()
    
    # Model 2: FVC@52
    try:
        fvc_df = pd.read_csv('D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training\\Progression_FVC_features_embedding_3\\results\\test_predictions_final.csv')
        print(f"✓ Model 2 (FVC@52): {len(fvc_df)} predictions")
    except FileNotFoundError:
        print("❌ Error: 'test_predictions_final.csv' not found")
        print("   Run: python Training/Progression_FVC_features_embedding_3/progression_fvc@52_kfold.py")
        exit()
    
    # ========================================================================
    # STEP 2: GET COMMON PATIENTS
    # ========================================================================
    print("\n[2/4] Aligning predictions...")
    
    fold_patients = set(fold_df['patient_id'].values)
    fvc_patients = set(fvc_df['patient_id'].values)
    
    common_patients = sorted(list(fold_patients & fvc_patients))
    
    print(f"✓ prediction_fold patients: {len(fold_patients)}")
    print(f"✓ FVC@52 patients: {len(fvc_patients)}")
    print(f"✓ Common patients: {len(common_patients)}")
    
    if len(common_patients) == 0:
        print("\n❌ No common patients found!")
        exit()
    
    # ========================================================================
    # STEP 3: COMPUTE TRUE LABELS
    # ========================================================================
    print("\n[3/4] Computing true labels...")
    
    # Load patient data to get baseline FVC
    from Training.Comparison_1_3.utilities import IPFDataLoader
    
    CSV_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train_with_coefs.csv'
    CSV_FEATURES_PATH = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\patient_features.csv'
    NPY_DIR = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy'
    
    dl = IPFDataLoader(CSV_PATH, CSV_FEATURES_PATH, NPY_DIR)
    patient_data, features_data = dl.get_patient_data()
    
    # Get baseline FVC
    baseline_fvc_dict = {}
    for pid in common_patients:
        if pid in patient_data and len(patient_data[pid]['fvc_values']) > 0:
            baseline_fvc_dict[pid] = patient_data[pid]['fvc_values'][0]
    
    # Compute true labels from FVC test set
    y_true = {}
    for _, row in fvc_df.iterrows():
        pid = row['patient_id']
        if pid not in common_patients or pid not in baseline_fvc_dict:
            continue
        
        baseline = baseline_fvc_dict[pid]
        true_fvc = row['true_fvc']
        decline_pct = 100 * (baseline - true_fvc) / baseline
        
        y_true[pid] = int(decline_pct >= 10.0)
    
    print(f"✓ True labels computed for {len(y_true)} patients")
    n_prog = sum(y_true.values())
    print(f"   Progressions: {n_prog}")
    print(f"   Stable: {len(y_true) - n_prog}")
    
    # ========================================================================
    # STEP 4: EXTRACT PREDICTIONS FOR COMMON PATIENTS
    # ========================================================================
    print("\n[4/4] Computing metrics...")
    
    # Model 1: prediction_fold
    y_true_list = []
    y_pred_fold = []
    y_prob_fold = []
    
    for pid in common_patients:
        if pid not in y_true:
            continue
        
        fold_row = fold_df[fold_df['patient_id'] == pid]
        if len(fold_row) == 0:
            continue
        
        y_true_list.append(y_true[pid])
        y_pred_fold.append(fold_row.iloc[0]['is_progression'])
        y_prob_fold.append(fold_row.iloc[0]['probability'])
    
    # Model 2: FVC@52
    y_pred_fvc = []
    y_prob_fvc = []
    
    for pid in common_patients:
        if pid not in y_true:
            continue
        
        fvc_row = fvc_df[fvc_df['patient_id'] == pid]
        if len(fvc_row) == 0:
            continue
        
        baseline = baseline_fvc_dict[pid]
        pred_fvc = fvc_row.iloc[0]['pred_fvc']
        decline_pct = 100 * (baseline - pred_fvc) / baseline
        
        y_pred_fvc.append(int(decline_pct >= 10.0))
        # Probability: normalized decline
        prob = min(decline_pct / 10.0, 1.0)
        prob = max(prob, 0.0)
        y_prob_fvc.append(prob)
    
    # Convert to numpy
    y_true_array = np.array(y_true_list)
    y_pred_fold = np.array(y_pred_fold)
    y_prob_fold = np.array(y_prob_fold)
    y_pred_fvc = np.array(y_pred_fvc)
    y_prob_fvc = np.array(y_prob_fvc)
    
    # Compute metrics
    metrics_fold = compute_metrics(y_true_array, y_pred_fold, y_prob_fold, "prediction_fold")
    metrics_fvc = compute_metrics(y_true_array, y_pred_fvc, y_prob_fvc, "FVC@52")
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n📊 PREDICTION_FOLD (Slope-based):")
    for key in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']:
        val = metrics_fold.get(key, 0)
        print(f"   {key:15s}: {val:.4f}")
    
    print("\n📊 FVC@52 (CNN Embedding):")
    for key in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']:
        val = metrics_fvc.get(key, 0)
        print(f"   {key:15s}: {val:.4f}")
    
    # Summary table
    summary_df = pd.DataFrame([metrics_fold, metrics_fvc])
    summary_df = summary_df[['Model', 'N', 'Positives', 'Negatives', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']]
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('two_model_comparison.csv', index=False)
    print("\n✅ Summary saved to 'two_model_comparison.csv'")
    
    # Visualize
    print("\n[VISUALIZATION] Generating comparison dashboard...")
    
    models_data = {
        'prediction_fold': {
            'y_true': y_true_array,
            'y_pred': y_pred_fold,
            'y_prob': y_prob_fold,
            'metrics': metrics_fold
        },
        'FVC@52': {
            'y_true': y_true_array,
            'y_pred': y_pred_fvc,
            'y_prob': y_prob_fvc,
            'metrics': metrics_fvc
        }
    }
    
    plot_comparison('prediction_fold', models_data['prediction_fold'], 
                   'FVC@52', models_data['FVC@52'],
                   save_path='two_model_comparison.png')
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
