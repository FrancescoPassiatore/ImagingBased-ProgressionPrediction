"""
Disciplined Cox Survival Analysis Experiment
Fixed hyperparameters: penalizer=0.5, l1_ratio=0.0 (Ridge)
"""

import pickle
from pathlib import Path
import pandas as pd
from cox_survival_analysis import SurvivalDataLoader, CoxSurvivalAnalyzer, CONFIG


def run_experiment(experiment_name, use_cnn, use_hand, use_demo, n_select=None, data_loader_shared=None):
    """Run a single experiment configuration"""
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"# CNN: {use_cnn}, Handcrafted: {use_hand}, Demographics: {use_demo}")
    if n_select:
        print(f"# Features to select: {n_select}")
    print(f"{'#'*80}")
    
    # Update config
    config = CONFIG.copy()
    if n_select:
        config['n_select'] = n_select
    
    # Use shared data loader if provided (to reuse CNN cache)
    if data_loader_shared:
        data_loader = data_loader_shared
        data_loader.config = config  # Update config for this experiment
    else:
        data_loader = SurvivalDataLoader(config)
    
    # Load K-fold splits
    with open(config['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    # Prepare dataset
    df = data_loader.prepare_full_dataset(
        use_cnn=use_cnn,
        use_hand=use_hand,
        use_demo=use_demo
    )
    
    # Run Cox analysis
    analyzer = CoxSurvivalAnalyzer(config)
    results = analyzer.run_cross_validation(
        df, kfold_splits,
        use_cnn=use_cnn,
        use_hand=use_hand,
        use_demo=use_demo,
        experiment_name=experiment_name
    )
    
    # Extract validation C-index
    val_ci_list = [r['val_ci'] for r in results]
    mean_val_ci = sum(val_ci_list) / len(val_ci_list) if val_ci_list else None
    
    return {
        'experiment': experiment_name,
        'mean_val_ci': mean_val_ci,
        'val_ci_list': val_ci_list
    }


def main():
    """Run all 5 disciplined experiments"""
    
    print(f"\n{'='*80}")
    print(f"DISCIPLINED COX SURVIVAL ANALYSIS EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Fixed Hyperparameters:")
    print(f"  - penalizer: 0.5")
    print(f"  - l1_ratio: 0.0 (Pure Ridge/L2)")
    print(f"{'='*80}")
    
    # Extract CNN features ONCE (reused across all CNN experiments)
    print("\n" + "="*80)
    print("PRE-EXTRACTING CNN FEATURES (will be reused)")
    print("="*80)
    data_loader_shared = SurvivalDataLoader(CONFIG)
    survival_df = data_loader_shared.load_survival_data()
    patient_ids = survival_df['Patient'].tolist()
    
    # Extract CNN features
    cnn_df = data_loader_shared.extract_cnn_features(patient_ids)
    print(f"✓ CNN features extracted for {len(cnn_df)} patients")
    
    # Store for reuse
    data_loader_shared.cnn_features_cache = cnn_df
    
    results_summary = []
    
    # Experiment 1: Handcrafted only (baseline)
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="1_handcrafted_only",
        use_cnn=False,
        use_hand=True,
        use_demo=False,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)
    

    # Experiment 1: Handcrafted + Demographics
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="1_handcrafted_demo",
        use_cnn=False,
        use_hand=True,
        use_demo=True,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)

    # Experiment 2: CNN univariate selection (30 features)
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="2_cnn_select30",
        use_cnn=True,
        use_hand=False,
        use_demo=False,
        n_select=30,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)
    
    # Experiment 3: CNN univariate selection (50 features)
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="3_cnn_select50",
        use_cnn=True,
        use_hand=False,
        use_demo=False,
        n_select=50,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)
    
    # Experiment 4: CNN univariate selection (70 features)
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="4_cnn_select70",
        use_cnn=True,
        use_hand=False,
        use_demo=False,
        n_select=70,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)
    
    # Experiment 5: CNN select 50 + Handcrafted + Demo
    print("\n" + "="*80)
    result = run_experiment(
        experiment_name="5_cnn_select50_hand_demo",
        use_cnn=True,
        use_hand=True,
        use_demo=True,
        n_select=50,
        data_loader_shared=data_loader_shared
    )
    results_summary.append(result)
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame([
        {
            'Experiment': r['experiment'],
            'Mean Val C-index': f"{r['mean_val_ci']:.4f}" if r['mean_val_ci'] else "N/A"
        }
        for r in results_summary
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = CONFIG['output_dir'] / 'experiment_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Results saved to: {comparison_path}")
    
    # Find best
    valid_results = [r for r in results_summary if r['mean_val_ci'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['mean_val_ci'])
        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION: {best['experiment']}")
        print(f"   Mean Val C-index: {best['mean_val_ci']:.4f}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
