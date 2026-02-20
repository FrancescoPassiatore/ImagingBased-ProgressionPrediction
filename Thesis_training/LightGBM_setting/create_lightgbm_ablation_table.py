"""
Create comprehensive LightGBM ablation study table with all metrics

Usage:
    python create_lightgbm_ablation_table.py
    
Output:
    - lightgbm_ablation_table.csv
    - lightgbm_ablation_table.tex (LaTeX format)
    - lightgbm_ablation_table_formatted.txt (readable text)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} ± {std:.3f}"


def create_lightgbm_ablation_table(results_dir: Path, output_dir: Path = None):
    """
    Create comprehensive LightGBM ablation study table
    
    Args:
        results_dir: Path to results directory with experiment subdirectories
        output_dir: Path to save output files (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Find all experiment directories (those with fold_results.csv)
    experiment_dirs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / "fold_results.csv").exists():
            experiment_dirs.append(d)
    
    if not experiment_dirs:
        print(f"❌ No experiment directories with fold_results.csv found in {results_dir}")
        return
    
    print(f"\n{'='*80}")
    print("CREATING LIGHTGBM ABLATION STUDY TABLE")
    print(f"{'='*80}")
    print(f"Found {len(experiment_dirs)} experiments:")
    for d in experiment_dirs:
        print(f"  - {d.name}")
    
    # Collect results from all experiments
    results = []
    
    # Enhanced descriptions mapping
    description_map = {
        'hand_only': 'Hand-crafted features only',
        'demo_only': 'Demographics only',
        'hand_demo': 'Hand-crafted + Demographics',
        'cnn_mean': 'CNN mean pooling only',
        'cnn_max': 'CNN max pooling only',
        'cnn_max_mean': 'CNN max+mean pooling (concatenated)',
        'cnn_mean_pca': 'CNN mean pooling + PCA',
        'cnn_max_pca': 'CNN max pooling + PCA',
        'cnn_max_mean_pca': 'CNN max+mean pooling + PCA',
        'best_cnn_hand': 'Best CNN + Hand-crafted',
        'best_cnn_hand_demo': 'Best CNN + Hand-crafted + Demographics (FULL)',
    }
    
    for exp_dir in experiment_dirs:
        config_name = exp_dir.name
        fold_results_file = exp_dir / "fold_results.csv"
        
        # Read fold results
        df = pd.read_csv(fold_results_file)
        
        # Get description
        description = description_map.get(config_name, f"[{config_name}]")
        
        # Calculate statistics for all metrics
        metrics = {
            'Configuration': config_name,
            'Description': description,
            'Accuracy': format_metric(df['test_accuracy'].mean(), df['test_accuracy'].std()),
            'Precision': format_metric(df['test_precision'].mean(), df['test_precision'].std()),
            'Recall': format_metric(df['test_recall'].mean(), df['test_recall'].std()),
            'Specificity': format_metric(df['test_specificity'].mean(), df['test_specificity'].std()),
            'Sensitivity': format_metric(df['test_recall'].mean(), df['test_recall'].std()),  # Sensitivity = Recall
            'F1': format_metric(df['test_f1'].mean(), df['test_f1'].std()),
            'AUC': format_metric(df['test_auc'].mean(), df['test_auc'].std()),
        }
        
        # Store raw values for sorting/analysis
        metrics['_auc_mean'] = df['test_auc'].mean()
        metrics['_accuracy_mean'] = df['test_accuracy'].mean()
        
        results.append(metrics)
        
        print(f"\n✓ {config_name}:")
        print(f"    AUC: {metrics['AUC']}")
        print(f"    Accuracy: {metrics['Accuracy']}")
        print(f"    F1: {metrics['F1']}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by logical order (hand-crafted first, then CNN, then combined)
    def get_sort_key(config_name):
        # Define order: hand/demo -> cnn -> cnn_pca -> best combined
        order_priority = {
            'hand_only': 1,
            'demo_only': 2,
            'hand_demo': 3,
            'cnn_mean': 10,
            'cnn_max': 11,
            'cnn_max_mean': 12,
            'cnn_mean_pca': 20,
            'cnn_max_pca': 21,
            'cnn_max_mean_pca': 22,
            'best_cnn_hand': 30,
            'best_cnn_hand_demo': 31,
        }
        return order_priority.get(config_name, 100)
    
    results_df['_sort_key'] = results_df['Configuration'].apply(get_sort_key)
    results_df = results_df.sort_values('_sort_key')
    results_df = results_df.drop(columns=['_sort_key', '_auc_mean', '_accuracy_mean'])
    
    # === Save CSV ===
    csv_path = output_dir / "lightgbm_ablation_table.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to: {csv_path}")
    
    # === Save formatted text table ===
    txt_path = output_dir / "lightgbm_ablation_table_formatted.txt"
    with open(txt_path, 'w') as f:
        f.write("="*140 + "\n")
        f.write("LIGHTGBM ABLATION STUDY RESULTS - COMPREHENSIVE METRICS\n")
        f.write("="*140 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*140 + "\n")
        f.write("Note: All metrics computed on test set with optimal threshold from validation\n")
        f.write("Sensitivity = Recall (True Positive Rate)\n")
        f.write("="*140 + "\n")
    
    print(f"✓ Formatted text saved to: {txt_path}")
    
    # === Save LaTeX table ===
    latex_path = output_dir / "lightgbm_ablation_table.tex"
    
    # Create LaTeX table with better formatting
    latex_content = r"""\begin{table}[h]
\centering
\caption{LightGBM Ablation Study Results - Comprehensive Metrics}
\label{tab:lightgbm_ablation_results}
\small
\begin{tabular}{llccccccc}
\toprule
\textbf{Config} & \textbf{Description} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{Sensitivity} & \textbf{F1} & \textbf{AUC} \\
\midrule
"""
    
    # Add data rows
    for _, row in results_df.iterrows():
        config = row['Configuration'].replace('_', '\\_')
        desc = row['Description'].replace('_', '\\_')
        # Shorten description for LaTeX
        if len(desc) > 50:
            desc = desc[:47] + "..."
        
        latex_content += f"{config} & {desc} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['Specificity']} & {row['Sensitivity']} & {row['F1']} & {row['AUC']} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

% Note: All metrics computed on test set with optimal threshold from validation
% Sensitivity = Recall (True Positive Rate)
"""
    
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # === Print summary ===
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Show best performing configurations
    # Need to recompute raw values if dropped
    auc_values = []
    for _, row in results_df.iterrows():
        # Parse AUC from string format "X.XXX ± Y.YYY"
        auc_str = row['AUC'].split(' ± ')[0]
        auc_values.append((row['Configuration'], row['Description'], float(auc_str), row['AUC']))
    
    auc_values.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 3 by AUC:")
    for config, desc, auc_val, auc_str in auc_values[:3]:
        print(f"  {config:25s} - {auc_str} - {desc}")
    
    print(f"\n{'='*80}")
    print(f"✓ All tables saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == "__main__":
    # Results directory
    results_dir = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\LightGBM_setting\results\lightgbm_experiment4_best")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("\nPlease specify the results directory:")
        user_input = input("Results directory path: ").strip()
        if user_input:
            results_dir = Path(user_input)
        else:
            exit(1)
    
    print(f"\n📁 Using results directory: {results_dir}")
    
    # Create table
    create_lightgbm_ablation_table(results_dir)
