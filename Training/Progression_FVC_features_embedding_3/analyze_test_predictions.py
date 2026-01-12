"""
Analyze Test Predictions - Compute Classification Metrics
Uses existing test_predictions_final.csv to compute progression classification metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, 
    average_precision_score, precision_recall_curve,
    matthews_corrcoef
)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("FVC@52 TEST SET - PROGRESSION CLASSIFICATION ANALYSIS")
print("="*80)

# Load test predictions
test_df = pd.read_csv(r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Training\Progression_FVC_features_embedding_3\results\test_predictions_final.csv')
print(f"\n✓ Loaded {len(test_df)} test predictions")

# Load baseline FVC values
train_csv = pd.read_csv(r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Training\CNN_Slope_Prediction\train.csv')

baseline_fvc = {}
for patient_id in test_df['patient_id'].values:
    patient_weeks = train_csv[train_csv['Patient'] == patient_id]['Weeks'].values
    if len(patient_weeks) > 0:
        earliest_week_idx = np.argmin(patient_weeks)
        baseline_fvc_val = train_csv[train_csv['Patient'] == patient_id].iloc[earliest_week_idx]['FVC']
        baseline_fvc[patient_id] = baseline_fvc_val

print(f"✓ Loaded baseline FVC for {len(baseline_fvc)} patients")

# ============================================================================
# COMPUTE BINARY LABELS (≥10% DECLINE = PROGRESSION)
# ============================================================================

print("\n" + "="*80)
print("COMPUTING PROGRESSION LABELS")
print("="*80)

y_true = []
y_pred = []
y_prob = []
patient_ids_valid = []

for _, row in test_df.iterrows():
    pid = row['patient_id']
    
    if pid not in baseline_fvc:
        continue
    
    baseline = baseline_fvc[pid]
    true_fvc = row['true_fvc']
    pred_fvc = row['pred_fvc']
    
    # True label: actual progression
    true_decline_pct = 100 * (baseline - true_fvc) / baseline
    y_true.append(int(true_decline_pct >= 10.0))
    
    # Predicted label: predicted progression
    pred_decline_pct = 100 * (baseline - pred_fvc) / baseline
    y_pred.append(int(pred_decline_pct >= 10.0))
    
    # Probability: normalized decline
    prob = max(0.0, min(pred_decline_pct / 10.0, 1.0))
    y_prob.append(prob)
    
    patient_ids_valid.append(pid)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

print(f"\n✓ Valid patients: {len(y_true)}")
print(f"  True progressions: {y_true.sum()}")
print(f"  True stable: {len(y_true) - y_true.sum()}")
print(f"  Predicted progressions: {y_pred.sum()}")
print(f"  Predicted stable: {len(y_pred) - y_pred.sum()}")

# ============================================================================
# COMPUTE METRICS
# ============================================================================

print("\n" + "="*80)
print("CLASSIFICATION METRICS")
print("="*80)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

# Specificity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# AUC and AP
if len(np.unique(y_true)) == 2:
    auc_roc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
else:
    auc_roc = np.nan
    ap = np.nan
    fpr, tpr = None, None

mcc = matthews_corrcoef(y_true, y_pred)

print(f"\n📊 PROGRESSION CLASSIFICATION RESULTS:")
print(f"   Accuracy:    {accuracy:.4f}")
print(f"   Precision:   {precision:.4f}")
print(f"   Recall:      {recall:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   AUC-ROC:     {auc_roc:.4f}")
print(f"   AP:          {ap:.4f}")
print(f"   MCC:         {mcc:.4f}")

print(f"\n   Confusion Matrix:")
print(f"   {cm}")
print(f"   [[TN={tn}, FP={fp}]")
print(f"    [FN={fn}, TP={tp}]]")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save enhanced predictions with classification labels
enhanced_df = pd.DataFrame({
    'patient_id': patient_ids_valid,
    'baseline_fvc': [baseline_fvc[pid] for pid in patient_ids_valid],
    'true_fvc': [test_df[test_df['patient_id'] == pid]['true_fvc'].values[0] for pid in patient_ids_valid],
    'pred_fvc': [test_df[test_df['patient_id'] == pid]['pred_fvc'].values[0] for pid in patient_ids_valid],
    'error': [test_df[test_df['patient_id'] == pid]['error'].values[0] for pid in patient_ids_valid],
    'error_percent': [test_df[test_df['patient_id'] == pid]['error_percent'].values[0] for pid in patient_ids_valid],
    'true_progression': y_true,
    'pred_progression': y_pred,
    'probability': y_prob
})

enhanced_df.to_csv('results/test_predictions_with_classification.csv', index=False)
print(f"\n✅ Enhanced predictions saved to 'results/test_predictions_with_classification.csv'")

# Save metrics summary
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC', 'AP', 'MCC'],
    'Value': [accuracy, precision, recall, specificity, f1, auc_roc, ap, mcc]
})
metrics_df.to_csv('results/test_classification_metrics.csv', index=False)
print(f"✅ Metrics saved to 'results/test_classification_metrics.csv'")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
if fpr is not None and tpr is not None:
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC Curve (AUC = {auc_roc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Test Set', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['Stable', 'Progression'],
            yticklabels=['Stable', 'Progression'])
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title('Confusion Matrix - Test Set', fontsize=12, fontweight='bold')

# 3. Metrics Bar Chart
ax = axes[1, 0]
metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metric_values = [accuracy, precision, recall, specificity, f1]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Classification Metrics - Test Set', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
           f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Precision-Recall Curve
if len(np.unique(y_true)) == 2:
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ax = axes[1, 1]
    ax.plot(recall_vals, precision_vals, color='#e74c3c', lw=2.5, label=f'PR Curve (AP = {ap:.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve - Test Set', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.suptitle('FVC@52 Test Set - Progression Classification Analysis', 
            fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/test_progression_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✅ PLOTS SAVED: 'results/test_progression_evaluation.png'")
plt.show()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
