"""
Analyze Fold Distribution
Check if some folds have impossible test sets
"""

import pandas as pd
import pickle
from pathlib import Path

def analyze_fold_distribution(kfold_splits_path, ground_truth_path):
    """
    Analyze the distribution of classes in each fold
    """
    
    # Load splits
    with open(kfold_splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path)
    
    print("\n" + "="*70)
    print("FOLD DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for fold_name, fold_data in sorted(splits.items()):
        fold_idx = fold_name
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*70}")
        
        for split_name in ['train', 'val', 'test']:
            patient_ids = fold_data[split_name]
            
            # Get labels for these patients
            labels = []
            for pid in patient_ids:
                label = gt_df[gt_df['PatientID'] == pid]['has_progressed'].values
                if len(label) > 0:
                    labels.append(label[0])
            
            n_progression = sum(labels)
            n_no_progression = len(labels) - n_progression
            
            print(f"\n{split_name.upper()}:")
            print(f"  Total patients: {len(patient_ids)}")
            print(f"  Progression: {n_progression} ({n_progression/len(labels)*100:.1f}%)")
            print(f"  No Progression: {n_no_progression} ({n_no_progression/len(labels)*100:.1f}%)")
            
            # Check for problems
            if split_name == 'test':
                if len(patient_ids) < 15:
                    print(f"  ⚠️  WARNING: Only {len(patient_ids)} test patients - very small!")
                
                if n_progression == 0 or n_no_progression == 0:
                    print(f"  🚨 CRITICAL: Only one class in test set!")
                
                if n_progression < 3 or n_no_progression < 3:
                    print(f"  ⚠️  WARNING: Very few samples of one class")
                
                # Check balance
                balance_ratio = min(n_progression, n_no_progression) / max(n_progression, n_no_progression)
                if balance_ratio < 0.3:
                    print(f"  ⚠️  WARNING: Severe class imbalance (ratio: {balance_ratio:.2f})")


def compare_fold_results_with_distribution(results_csv, kfold_splits_path, ground_truth_path):
    """
    Compare fold results with their class distribution
    """
    
    # Load results
    results_df = pd.read_csv(results_csv)
    
    # Load splits and GT
    with open(kfold_splits_path, 'rb') as f:
        splits = pickle.load(f)
    gt_df = pd.read_csv(ground_truth_path)
    
    print("\n" + "="*70)
    print("FOLD RESULTS vs DISTRIBUTION")
    print("="*70)
    
    analysis = []
    
    for _, row in results_df.iterrows():
        fold_idx = int(row['fold_idx'])
        fold_data = splits[fold_idx]
        
        # Get test set distribution
        test_ids = fold_data['test']
        test_labels = []
        for pid in test_ids:
            label = gt_df[gt_df['PatientID'] == pid]['has_progressed'].values
            if len(label) > 0:
                test_labels.append(label[0])
        
        n_prog = sum(test_labels)
        n_no_prog = len(test_labels) - n_prog
        balance = min(n_prog, n_no_prog) / max(n_prog, n_no_prog) if max(n_prog, n_no_prog) > 0 else 0
        
        analysis.append({
            'fold': fold_idx,
            'val_auc': row['val_auc'],
            'test_auc': row['test_auc'],
            'test_acc': row['test_acc'],
            'test_size': len(test_ids),
            'test_prog': n_prog,
            'test_no_prog': n_no_prog,
            'balance_ratio': balance
        })
    
    analysis_df = pd.DataFrame(analysis)
    
    print("\nFold Analysis:")
    print(analysis_df.to_string(index=False))
    
    # Identify problematic folds
    print("\n" + "="*70)
    print("PROBLEMATIC FOLDS")
    print("="*70)
    
    for _, row in analysis_df.iterrows():
        issues = []
        
        if row['test_auc'] < 0.4:
            issues.append(f"Very low test AUC ({row['test_auc']:.3f})")
        
        if row['test_acc'] < 0.35 or row['test_acc'] > 0.75:
            issues.append(f"Extreme accuracy ({row['test_acc']:.3f}) - may predict only one class")
        
        if row['balance_ratio'] < 0.35:
            issues.append(f"Severe imbalance (ratio: {row['balance_ratio']:.2f})")
        
        if row['test_size'] < 15:
            issues.append(f"Very small test set ({int(row['test_size'])} patients)")
        
        if abs(row['val_auc'] - row['test_auc']) > 0.3:
            issues.append(f"Large val-test gap ({abs(row['val_auc'] - row['test_auc']):.3f})")
        
        if issues:
            print(f"\nFold {row['fold']}:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
            print(f"  Test distribution: {int(row['test_prog'])} prog / {int(row['test_no_prog'])} no-prog")


def recommend_actions(analysis_df):
    """
    Recommend what to do based on analysis
    """
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    
   
    
    # Check variance
    val_auc_std = analysis_df['val_auc'].std()
    test_auc_std = analysis_df['test_auc'].std()
    
    if val_auc_std > 0.1 or test_auc_std > 0.15:
        print("\n⚠️  HIGH VARIANCE ACROSS FOLDS")
        print(f"   Val AUC std: {val_auc_std:.3f}")
        print(f"   Test AUC std: {test_auc_std:.3f}")
        print("\n   POSSIBLE CAUSES:")
        print("   - Dataset too small")
        print("   - Heterogeneous patient population")
        print("   - Some folds have outlier patients")
        print("\n   RECOMMENDED ACTIONS:")
        print("   1. Use repeated k-fold CV (repeat 3-5 times)")
        print("   2. Report mean ± std")
        print("   3. Analyze outlier patients")
        print("   4. Consider patient clustering before splitting")


# Usage
if __name__ == "__main__":
    
    CONFIG = {
        'kfold_splits_path': Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits.pkl"),
        'gt_path': Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv"),
        'results_csv': Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Results_2\kfold_summary.csv")
    }
    
    # Analyze distribution
    analyze_fold_distribution(
        CONFIG['kfold_splits_path'],
        CONFIG['gt_path']
    )
    
    # Compare with results
    compare_fold_results_with_distribution(
        CONFIG['results_csv'],
        CONFIG['kfold_splits_path'],
        CONFIG['gt_path']
    )
    
    # Get recommendations
    results_df = pd.read_csv(CONFIG['results_csv'])
    recommend_actions(results_df)