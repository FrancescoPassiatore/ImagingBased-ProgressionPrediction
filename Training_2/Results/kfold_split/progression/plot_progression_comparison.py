import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
csvs = {
    'DenseNet': 'D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\Results\\progression\\progression_metrics\\mlp_progression_prediction_summary_densenet.csv',
    'Robust Scaler': 'D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\Results\\progression\\progression_metrics\\mlp_progression_prediction_summary_robust_Scaler.csv',
    'Standard Scaler': 'D:\\FrancescoP\\ImagingBased-ProgressionPrediction\\Training_2\\Results\\progression\\progression_metrics\\mlp_progression_prediction_summary_standard_scaler.csv',
}

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC', 'AP']

# Load all data
all_data = []
for setting, path in csvs.items():
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        for metric in metrics:
            mean = float(str(row[metric]).split('±')[0])
            std = float(str(row[metric]).split('±')[1]) if '±' in str(row[metric]) else np.nan
            all_data.append({
                'Setting': setting,
                'Approach': row['Approach'],
                'Metric': metric,
                'Mean': mean,
                'Std': std
            })
plot_df = pd.DataFrame(all_data)

# Plot grouped bar charts for each metric
out_dir = Path('Training_2/Results/progression/progression_plots')
out_dir.mkdir(parents=True, exist_ok=True)
sns.set(style="whitegrid", font_scale=1.1)

for metric in metrics:
    plt.figure(figsize=(8,5))
    data = plot_df[plot_df['Metric'] == metric]
    ax = sns.barplot(
        data=data,
        x='Approach', y='Mean', hue='Setting',
        capsize=0.1, errorbar=None, err_kws={'linewidth': 1.5}
    )
    # Add error bars manually, grouped by approach and setting
    n_settings = data['Setting'].nunique()
    n_approaches = data['Approach'].nunique()
    for i, (idx, row) in enumerate(data.iterrows()):
        # Calculate the bar index
        approach_idx = list(data['Approach'].unique()).index(row['Approach'])
        setting_idx = list(data['Setting'].unique()).index(row['Setting'])
        bar_idx = approach_idx + setting_idx * n_approaches
        if bar_idx < len(ax.patches):
            bar = ax.patches[bar_idx]
            std = row['Std']
            bar_x = bar.get_x() + bar.get_width()/2
            bar_y = bar.get_height()
            if not np.isnan(std):
                ax.errorbar(bar_x, bar_y, yerr=std, color='black', capsize=4, fmt='none', lw=1.5)
    # Add value labels on bars
    for bar in ax.patches:
        bar_x = bar.get_x() + bar.get_width()/2
        bar_y = bar.get_height()
        ax.text(bar_x, bar_y + 0.01, f'{bar_y:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.title(f'Progression Prediction: {metric}')
    plt.ylabel(metric)
    plt.ylim(0, 1.05 if metric != 'AP' else 1.1)
    plt.legend(title='Setting')
    plt.tight_layout()
    save_path = out_dir / f'progression_comparison_{metric.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    plt.show()
