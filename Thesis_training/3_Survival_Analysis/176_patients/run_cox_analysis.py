"""
Run Cox Survival Analysis with different feature configurations
"""

from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))

from cox_survival_analysis import (
    SurvivalDataLoader, CoxSurvivalAnalyzer, CONFIG
)


def run_ablation_study():
    """Run Cox analysis with different feature combinations"""
    
    print(f"\n{'='*80}")
    print(f"COX SURVIVAL ANALYSIS - ABLATION STUDY")
    print(f"{'='*80}")
    
    # Feature configurations
    configs = {
        'demo_only': {
            'use_cnn': False,
            'use_hand': False,
            'use_demo': True,
            'description': 'Demographics only'
        },
        'hand_only': {
            'use_cnn': False,
            'use_hand': True,
            'use_demo': False,
            'description': 'Handcrafted features only'
        },
        'hand_demo': {
            'use_cnn': False,
            'use_hand': True,
            'use_demo': True,
            'description': 'Handcrafted + Demographics'
        },
        'cnn_only': {
            'use_cnn': True,
            'use_hand': False,
            'use_demo': False,
            'description': 'CNN features only'
        },
        'cnn_demo': {
            'use_cnn': True,
            'use_hand': False,
            'use_demo': True,
            'description': 'CNN + Demographics'
        },
        'cnn_hand': {
            'use_cnn': True,
            'use_hand': True,
            'use_demo': False,
            'description': 'CNN + Handcrafted'
        },
        'full': {
            'use_cnn': True,
            'use_hand': True,
            'use_demo': True,
            'description': 'All features (CNN + Handcrafted + Demographics)'
        }
    }
    
    # Load K-fold splits
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    # Run each configuration
    all_results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n{'='*80}")
        print(f"CONFIGURATION: {config_params['description']}")
        print(f"{'='*80}")
        
        # Load data
        data_loader = SurvivalDataLoader(CONFIG)
        df = data_loader.prepare_full_dataset(
            use_cnn=config_params['use_cnn'],
            use_hand=config_params['use_hand'],
            use_demo=config_params['use_demo']
        )
        
        # Run analysis
        analyzer = CoxSurvivalAnalyzer(CONFIG)
        results = analyzer.run_cross_validation(
            df, kfold_splits,
            use_cnn=config_params['use_cnn'],
            use_hand=config_params['use_hand'],
            use_demo=config_params['use_demo']
        )
        
        all_results[config_name] = results
    
    print(f"\n{'='*80}")
    print(f"✓ ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {CONFIG['output_dir']}")
    
    return all_results


def run_single_configuration(use_cnn=True, use_hand=True, use_demo=True):
    """Run Cox analysis with a single feature configuration"""
    
    print(f"\n{'='*80}")
    print(f"COX SURVIVAL ANALYSIS")
    print(f"Features: CNN={use_cnn}, Hand={use_hand}, Demo={use_demo}")
    print(f"{'='*80}")
    
    # Load K-fold splits
    with open(CONFIG['kfold_splits_path'], 'rb') as f:
        kfold_splits = pickle.load(f)
    
    # Load data
    data_loader = SurvivalDataLoader(CONFIG)
    df = data_loader.prepare_full_dataset(
        use_cnn=use_cnn,
        use_hand=use_hand,
        use_demo=use_demo
    )
    
    # Run analysis
    analyzer = CoxSurvivalAnalyzer(CONFIG)
    results = analyzer.run_cross_validation(
        df, kfold_splits,
        use_cnn=use_cnn,
        use_hand=use_hand,
        use_demo=use_demo
    )
    
    print(f"\n{'='*80}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Cox Survival Analysis')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'ablation'],
                        help='Run mode: single configuration or ablation study')
    parser.add_argument('--cnn', type=int, default=1, help='Use CNN features (0/1)')
    parser.add_argument('--hand', type=int, default=1, help='Use handcrafted features (0/1)')
    parser.add_argument('--demo', type=int, default=1, help='Use demographics (0/1)')
    
    args = parser.parse_args()
    
    if args.mode == 'ablation':
        run_ablation_study()
    else:
        run_single_configuration(
            use_cnn=bool(args.cnn),
            use_hand=bool(args.hand),
            use_demo=bool(args.demo)
        )
