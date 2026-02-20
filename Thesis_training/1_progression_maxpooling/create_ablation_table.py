"""
Create comprehensive ablation study table with all metrics

Usage:
    python create_ablation_table.py
    
Output:
    - ablation_results_table.csv
    - ablation_results_table.tex (LaTeX format)
    - ablation_results_table_formatted.txt (readable text)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} ± {std:.3f}"


def create_ablation_table(results_dir: Path, output_dir: Path = None):
    """
    Create comprehensive ablation study table
    
    Args:
        results_dir: Path to results directory with ablation_* subdirectories
        output_dir: Path to save output files (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Find all ablation experiment directories
    ablation_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('ablation_')])
    
    if not ablation_dirs:
        print(f"❌ No ablation directories found in {results_dir}")
        return
    
    print(f"\n{'='*80}")
    print("CREATING ABLATION STUDY TABLE")
    print(f"{'='*80}")
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
        
        # Extract description from first line or use config name
        description = f"[{config_name}]"  # Default
        
        # Calculate statistics for metrics with optimal threshold
        metrics = {
            'Configuration': config_name,
            'Description': description,
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
        
        results.append(metrics)
        
        print(f"\n✓ {config_name}:")
        print(f"    AUC: {metrics['AUC']}")
        print(f"    Accuracy: {metrics['Accuracy']}")
        print(f"    F1: {metrics['F1']}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by block order (1_ first, then 2_, then 3_)
    def get_sort_key(config_name):
        if config_name.startswith('1_'):
            return (1, config_name)
        elif config_name.startswith('2_'):
            return (2, config_name)
        elif config_name.startswith('3_'):
            return (3, config_name)
        else:
            return (4, config_name)
    
    results_df['_sort_key'] = results_df['Configuration'].apply(get_sort_key)
    results_df = results_df.sort_values('_sort_key')
    results_df = results_df.drop(columns=['_sort_key', '_auc_mean', '_accuracy_mean'])
    
    # Enhanced descriptions mapping
    description_map = {
        '1_hand_only': '[BLOCK 1] Hand-crafted features only',
        '1_hand_demo': '[BLOCK 1] Hand-crafted + Demographics',
        '2_cnn_mean': '[BLOCK 2] CNN with mean pooling',
        '2_cnn_max': '[BLOCK 2] CNN with max pooling',
        '2_cnn_max_mean': '[BLOCK 2] CNN with max+mean pooling',
        '3_cnn_hand': '[BLOCK 3] CNN + Hand-crafted',
        '3_cnn_hand_demo': '[BLOCK 3] CNN + Hand-crafted + Demographics (FULL)',
    }
    
    results_df['Description'] = results_df['Configuration'].map(description_map).fillna(results_df['Description'])
    
    # === Save CSV ===
    csv_path = output_dir / "ablation_results_table.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved to: {csv_path}")
    
    # === Save formatted text table ===
    txt_path = output_dir / "ablation_results_table_formatted.txt"
    with open(txt_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("ABLATION STUDY RESULTS - COMPREHENSIVE METRICS\n")
        f.write("="*120 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*120 + "\n")
        f.write("Note: All metrics use optimal threshold determined on validation set\n")
        f.write("Sensitivity = Recall (True Positive Rate)\n")
        f.write("="*120 + "\n")
    
    print(f"✓ Formatted text saved to: {txt_path}")
    
    # === Save LaTeX table ===
    latex_path = output_dir / "ablation_results_table.tex"
    
    # Create LaTeX table with better formatting
    latex_content = r"""\begin{table}[h]
\centering
\caption{Ablation Study Results - Comprehensive Metrics}
\label{tab:ablation_results}
\small
\begin{tabular}{llccccccc}
\toprule
\textbf{Config} & \textbf{Description} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Specificity} & \textbf{Sensitivity} & \textbf{F1} & \textbf{AUC} \\
\midrule
"""
    
    # Add data rows
    for _, row in results_df.iterrows():
        config = row['Configuration'].replace('_', '\\_')
        desc = row['Description'].replace('_', '\\_').replace('[', '').replace(']', '')
        # Shorten description for LaTeX
        desc_parts = desc.split(']')
        if len(desc_parts) > 1:
            block = desc_parts[0]
            rest = desc_parts[1].strip()
            desc = f"{block}] {rest[:40]}..." if len(rest) > 40 else f"{block}] {rest}"
        
        latex_content += f"{config} & {desc} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['Specificity']} & {row['Sensitivity']} & {row['F1']} & {row['AUC']} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

% Note: All metrics use optimal threshold determined on validation set
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
    print("\nTop 3 by AUC:")
    for i, row in results_df.nlargest(3, '_auc_mean' if '_auc_mean' in results_df.columns else 'AUC').iterrows():
        print(f"  {row['Configuration']:20s} - {row['AUC']}")
    
    print(f"\n{'='*80}")
    print(f"✓ All tables saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == "__main__":
    # Determine results directory
    script_dir = Path(__file__).parent
    
    # Look for results directories
    possible_dirs = [
        script_dir / "two_blocks_analysis",
        script_dir / "three_blocks_analysis",
        script_dir / "ablation_results",
        script_dir / "results",
    ]
    
    results_dir = None
    for d in possible_dirs:
        if d.exists() and any(d.iterdir()):
            ablation_dirs = [x for x in d.iterdir() if x.is_dir() and x.name.startswith('ablation_')]
            if ablation_dirs:
                results_dir = d
                break
    
    if results_dir is None:
        print("❌ Could not find results directory with ablation experiments")
        print("Looking for directories with ablation_* subdirectories in:")
        for d in possible_dirs:
            print(f"  - {d}")
        print("\nPlease specify the results directory:")
        user_input = input("Results directory path: ").strip()
        if user_input:
            results_dir = Path(user_input)
        else:
            exit(1)
    
    print(f"\n📁 Using results directory: {results_dir}")
    
    # Create table
    create_ablation_table(results_dir)
