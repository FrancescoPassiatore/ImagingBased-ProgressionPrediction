"""
Create comprehensive ablation study table for third block analysis

Usage:
    python create_ablation_table.py
    
Output:
    - third_block_ablation_table.csv
    - third_block_ablation_table.tex (LaTeX format)
    - third_block_ablation_table_formatted.txt (readable text)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} ± {std:.3f}"


def create_ablation_table(results_dir: Path, output_dir: Path = None):
    """
    Create comprehensive ablation study table for third block experiments
    
    Args:
        results_dir: Path to third_block_analysis directory with ablation_* subdirectories
        output_dir: Path to save output files (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Find all ablation experiment directories
    ablation_dirs = sorted([d for d in results_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('ablation_3_')])
    
    if not ablation_dirs:
        print(f"❌ No ablation_3_* directories found in {results_dir}")
        return
    
    print(f"\n{'='*80}")
    print("CREATING THIRD BLOCK ABLATION STUDY TABLE")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Found {len(ablation_dirs)} experiments:")
    for d in ablation_dirs:
        print(f"  - {d.name}")
    
    # Collect results from all experiments
    results = []
    
    for ablation_dir in ablation_dirs:
        config_name = ablation_dir.name.replace('ablation_', '')
        detailed_results_file = ablation_dir / "detailed_fold_results.csv"
        
        if not detailed_results_file.exists():
            print(f"⚠️  Skipping {config_name}: detailed_fold_results.csv not found")
            continue
        
        # Read fold results
        df = pd.read_csv(detailed_results_file)
        
        # Calculate statistics for metrics with optimal threshold
        metrics = {
            'Configuration': config_name,
            'Accuracy': format_metric(df['test_accuracy_optimal'].mean(), df['test_accuracy_optimal'].std()),
            'Precision': format_metric(df['test_precision_optimal'].mean(), df['test_precision_optimal'].std()),
            'Recall': format_metric(df['test_recall_optimal'].mean(), df['test_recall_optimal'].std()),
            'Specificity': format_metric(df['test_specificity_optimal'].mean(), df['test_specificity_optimal'].std()),
            'Sensitivity': format_metric(df['test_recall_optimal'].mean(), df['test_recall_optimal'].std()),  # Sensitivity = Recall
            'F1': format_metric(df['test_f1_optimal'].mean(), df['test_f1_optimal'].std()),
            'AUC': format_metric(df['test_auc_optimal'].mean(), df['test_auc_optimal'].std()),
        }
        
        # Store raw values for sorting/analysis
        metrics['_auc_mean'] = df['test_auc_optimal'].mean()
        metrics['_accuracy_mean'] = df['test_accuracy_optimal'].mean()
        metrics['_n_folds'] = len(df)
        
        results.append(metrics)
        
        print(f"\n✓ {config_name}:")
        print(f"    Folds: {len(df)}")
        print(f"    AUC: {metrics['AUC']}")
        print(f"    Accuracy: {metrics['Accuracy']}")
        print(f"    F1: {metrics['F1']}")
    
    if not results:
        print(f"\n❌ No valid results found!")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by configuration name
    results_df = results_df.sort_values('Configuration')
    
    # Enhanced descriptions mapping for third block
    description_map = {
        '3_cnn_hand': 'CNN + Hand-crafted features',
        '3_cnn_hand_demo': 'CNN + Hand-crafted + Demographics (FULL)',
    }
    
    results_df['Description'] = results_df['Configuration'].map(description_map).fillna(results_df['Configuration'])
    
    # Reorder columns for better readability
    column_order = ['Configuration', 'Description', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F1', 'AUC']
    results_df = results_df[column_order + ['_auc_mean', '_accuracy_mean', '_n_folds']]
    
    # === Save CSV ===
    csv_path = output_dir / "third_block_ablation_table.csv"
    results_df.drop(columns=['_auc_mean', '_accuracy_mean', '_n_folds']).to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to: {csv_path}")
    
    # === Save formatted text table ===
    txt_path = output_dir / "third_block_ablation_table_formatted.txt"
    with open(txt_path, 'w') as f:
        f.write("="*140 + "\n")
        f.write("THIRD BLOCK ABLATION STUDY RESULTS - COMPREHENSIVE METRICS\n")
        f.write("="*140 + "\n\n")
        display_df = results_df.drop(columns=['_auc_mean', '_accuracy_mean', '_n_folds'])
        f.write(display_df.to_string(index=False))
        f.write("\n\n" + "="*140 + "\n")
        f.write("Note: All metrics use optimal threshold determined on validation set\n")
        f.write("Sensitivity = Recall (True Positive Rate)\n")
        f.write("Block 3 experiments combine CNN features with hand-crafted and demographic features\n")
        f.write("="*140 + "\n")
    
    print(f"✓ Formatted text saved to: {txt_path}")
    
    # === Save LaTeX table ===
    latex_path = output_dir / "third_block_ablation_table.tex"
    
    # Create LaTeX table with better formatting
    latex_content = r"""\begin{table}[h]
\centering
\caption{Third Block Ablation Study Results - Comprehensive Metrics}
\label{tab:third_block_ablation}
\small
\begin{tabular}{llccccccc}
\toprule
\textbf{Config} & \textbf{Description} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{Sensitivity} & \textbf{F1} & \textbf{AUC} \\
\midrule
"""
    
    # Add data rows
    for _, row in results_df.iterrows():
        config_short = row['Configuration'].replace('_', '\\_')
        desc = row['Description'].replace('_', '\\_').replace('+', '$+$')
        latex_content += f"{config_short} & {desc} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['Specificity']} & {row['Sensitivity']} & {row['F1']} & {row['AUC']} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

% Note: All metrics use optimal threshold determined on validation set
% Sensitivity = Recall (True Positive Rate)
% Block 3 experiments combine CNN features with hand-crafted and demographic features
"""
    
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # === Print summary statistics ===
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nBest performing configuration (by AUC):")
    best_idx = results_df['_auc_mean'].idxmax()
    best_config = results_df.loc[best_idx]
    print(f"  Configuration: {best_config['Configuration']}")
    print(f"  Description: {best_config['Description']}")
    print(f"  AUC: {best_config['AUC']}")
    print(f"  Accuracy: {best_config['Accuracy']}")
    print(f"  F1: {best_config['F1']}")
    
    print(f"\nAUC comparison:")
    for _, row in results_df.iterrows():
        print(f"  {row['Configuration']:30s}: {row['_auc_mean']:.4f}")
    
    if len(results_df) > 1:
        auc_diff = results_df['_auc_mean'].max() - results_df['_auc_mean'].min()
        print(f"\nAUC difference (max - min): {auc_diff:.4f}")
    
    print(f"\n{'='*80}")
    print("✓ ABLATION TABLE CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")
    print(f"  - {latex_path}")


def main():
    """Main execution"""
    # Set results directory
    results_dir = Path(__file__).parent
    
    print(f"\nWorking directory: {results_dir}")
    
    # Create ablation table
    create_ablation_table(results_dir)


if __name__ == '__main__':
    main()
