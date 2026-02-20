"""
Test data loading with updated IPFDataLoader
"""
import sys
from pathlib import Path

# Add utilities path
sys.path.append(str(Path(__file__).parent.parent / '1_progression_maxpooling'))

from utilities import IPFDataLoader

print("="*80)
print("TESTING UPDATED IPFDataLoader")
print("="*80)

# Paths
ground_truth_path = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv'
features_path = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\patient_features_30_60.csv'
demographics_path = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\train.csv'
npy_dir = r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\extracted_npy_full_dataset'

print("\n1. Initializing IPFDataLoader...")
try:
    data_loader = IPFDataLoader(
        ground_truth_path=ground_truth_path,
        features_path=features_path,
        npy_dir=npy_dir,
        demographics_path=demographics_path
    )
    print("   ✓ IPFDataLoader initialized successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n2. Loading patient data and features...")
try:
    patient_data, features_data = data_loader.get_patient_data()
    print(f"   ✓ Loaded successfully")
    print(f"   - Patients with complete data: {len(patient_data)}")
    print(f"   - Patients with features: {len(features_data)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Checking data structure...")
if len(patient_data) > 0:
    # Show first patient
    first_patient_id = list(patient_data.keys())[0]
    print(f"\n   Example patient: {first_patient_id}")
    print(f"   Patient data keys: {list(patient_data[first_patient_id].keys())}")
    print(f"   - Label: {patient_data[first_patient_id]['gt_has_progressed']}")
    print(f"   - Slices: {patient_data[first_patient_id]['n_slices']}")
    
    if first_patient_id in features_data:
        print(f"\n   Features data keys: {list(features_data[first_patient_id].keys())}")
        print(f"   - Age: {features_data[first_patient_id]['age']}")
        print(f"   - Sex: {features_data[first_patient_id]['sex']}")
        print(f"   - Smoking: {features_data[first_patient_id]['smoking_status']}")
        print(f"   - ApproxVol: {features_data[first_patient_id]['approx_vol']:.2f}")

print("\n4. Checking progression distribution...")
progression_counts = sum(1 for p in patient_data.values() if p['gt_has_progressed'] == 1)
no_progression_counts = len(patient_data) - progression_counts
print(f"   Progression: {progression_counts} ({progression_counts/len(patient_data)*100:.1f}%)")
print(f"   No Progression: {no_progression_counts} ({no_progression_counts/len(patient_data)*100:.1f}%)")

print("\n" + "="*80)
print("TEST COMPLETED SUCCESSFULLY ✓")
print("="*80)
print("\nYou can now run: train_ensemble.py")
