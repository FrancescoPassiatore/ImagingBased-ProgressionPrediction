"""
Master Script - Run All Training Steps
=======================================

This script executes all training steps in sequence:
1. Train CNN (5-fold CV)
2. Train Correctors (4 approaches × 5 folds)
3. Predict FVC@52
4. Compare Approaches

Usage:
    python run_all.py                  # Run all steps
    python run_all.py --steps 1 2      # Run only steps 1 and 2
    python run_all.py --skip-cnn       # Skip CNN training (use existing)
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

STEPS = {
    1: {
        'name': 'Train CNN (5-Fold CV)',
        'script': 'train_cnn_kfold.py',
        'description': 'Trains CNN slope predictor with 5-fold cross-validation',
        'estimated_time': '2-4 hours'
    },
    2: {
        'name': 'Train Correctors (4 Approaches)',
        'script': 'train_corrector_kfold.py',
        'description': 'Trains slope correctors for 4 different approaches',
        'estimated_time': '1-2 hours'
    },
    3: {
        'name': 'Predict FVC@52',
        'script': 'predict_fvc52.py',
        'description': 'Predicts FVC at week 52 and evaluates all approaches',
        'estimated_time': '30 minutes'
    },
    4: {
        'name': 'Compare Approaches',
        'script': 'compare_approaches.py',
        'description': 'Creates comprehensive comparison of all approaches',
        'estimated_time': '5 minutes'
    }
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_step_info(step_num):
    """Print step information"""
    step = STEPS[step_num]
    print(f"\n{'='*80}")
    print(f"STEP {step_num}/4: {step['name']}")
    print(f"{'='*80}")
    print(f"Description: {step['description']}")
    print(f"Estimated time: {step['estimated_time']}")
    print(f"Script: {step['script']}")
    print(f"{'='*80}\n")


def run_step(step_num):
    """Run a single training step"""
    step = STEPS[step_num]
    script_path = Path(__file__).parent / step['script']
    
    if not script_path.exists():
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    print(f"▶️  Starting: {step['name']}...")
    print(f"   Running: python {step['script']}\n")
    
    start_time = time.time()
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            check=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n✅ COMPLETED: {step['name']}")
        print(f"   Time elapsed: {hours}h {minutes}m {seconds}s\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {step['name']} failed with exit code {e.returncode}")
        print(f"   Please check the output above for error details.\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  INTERRUPTED: {step['name']} was interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error in {step['name']}")
        print(f"   {type(e).__name__}: {e}\n")
        return False


def check_prerequisites():
    """Check if required files exist"""
    print_header("CHECKING PREREQUISITES")
    
    required_files = [
        'Training/CNN_Slope_Prediction/train_with_coefs.csv',
        'Training/CNN_Slope_Prediction/patient_features.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    # Check NPY directory
    npy_dir = Path(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy\extracted_npy')
    if npy_dir.exists():
        n_patients = len(list(npy_dir.iterdir()))
        print(f"✓ Found NPY directory with {n_patients} patient folders")
    else:
        print(f"❌ Missing: NPY directory")
        all_exist = False
    
    print()
    return all_exist


def print_summary(completed_steps, failed_steps, total_time):
    """Print execution summary"""
    print_header("EXECUTION SUMMARY")
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Completed steps: {len(completed_steps)}/4")
    
    if completed_steps:
        print("\n✅ Completed:")
        for step_num in completed_steps:
            print(f"   {step_num}. {STEPS[step_num]['name']}")
    
    if failed_steps:
        print("\n❌ Failed:")
        for step_num in failed_steps:
            print(f"   {step_num}. {STEPS[step_num]['name']}")
    
    print("\n" + "="*80)
    
    if len(completed_steps) == 4:
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
        print("\nNext steps:")
        print("1. Check results/ folder for predictions and metrics")
        print("2. Check plots/ folder for visualizations")
        print("3. Review fvc52_summary.csv for final results")
        print()
    elif failed_steps:
        print("\n⚠️  Some steps failed. Please review errors above.")
    
    print("="*80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run all training steps in sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                  # Run all steps
  python run_all.py --steps 1 2      # Run only steps 1 and 2
  python run_all.py --steps 3 4      # Run only steps 3 and 4 (skip training)
  python run_all.py --skip-cnn       # Skip CNN training (use existing models)
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4],
        help='Specific steps to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-cnn',
        action='store_true',
        help='Skip CNN training (equivalent to --steps 2 3 4)'
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.skip_cnn:
        steps_to_run = [2, 3, 4]
    elif args.steps:
        steps_to_run = sorted(args.steps)
    else:
        steps_to_run = [1, 2, 3, 4]
    
    # Print welcome message
    print_header("5-FOLD CROSS-VALIDATION RETRAINING")
    print("This script will execute the following steps:")
    for step_num in steps_to_run:
        print(f"  {step_num}. {STEPS[step_num]['name']} (~{STEPS[step_num]['estimated_time']})")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please ensure all required files exist.")
        return 1
    
    # Confirm execution
    response = input("Do you want to continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Execution cancelled.")
        return 0
    
    # Execute steps
    start_time = time.time()
    completed_steps = []
    failed_steps = []
    
    for step_num in steps_to_run:
        print_step_info(step_num)
        
        success = run_step(step_num)
        
        if success:
            completed_steps.append(step_num)
        else:
            failed_steps.append(step_num)
            
            # Ask if user wants to continue
            if step_num < max(steps_to_run):
                response = input("\nStep failed. Continue with next step? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Execution stopped by user.")
                    break
    
    # Print summary
    total_time = time.time() - start_time
    print_summary(completed_steps, failed_steps, total_time)
    
    return 0 if len(failed_steps) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
