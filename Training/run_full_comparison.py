"""
Quick launcher: Run prediction_fold -> Run comparison
"""
import subprocess
import sys
import os

print("\n" + "="*80)
print("LAUNCHING COMPARISON PIPELINE")
print("="*80)

# Step 1: Run prediction_fold
print("\n[1/2] Running prediction_fold.py...")
print("-"*80)

try:
    result = subprocess.run(
        [sys.executable, r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\Progression_prediction_slope_1\prediction_fold.py'],
        cwd=r'D:\FrancescoP\ImagingBased-ProgressionPrediction',
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ prediction_fold.py failed with code {result.returncode}")
        sys.exit(1)
    
    print("\n✅ prediction_fold.py completed successfully")
    
except Exception as e:
    print(f"\n❌ Error running prediction_fold.py: {e}")
    sys.exit(1)

# Step 2: Run comparison
print("\n[2/2] Running compare_two_models.py...")
print("-"*80)

try:
    result = subprocess.run(
        [sys.executable, r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training\compare_two_models.py'],
        cwd=r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Training',
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ compare_two_models.py failed with code {result.returncode}")
        sys.exit(1)
    
    print("\n✅ compare_two_models.py completed successfully")
    
except Exception as e:
    print(f"\n❌ Error running compare_two_models.py: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE!")
print("="*80)
print("\nOutput files generated:")
print("  1. prediction_fold_final.csv")
print("  2. two_model_comparison.csv")
print("  3. two_model_comparison.png")
